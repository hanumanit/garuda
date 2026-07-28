//! Multi-user scheduler: priority queue, bounded concurrency, cancellation, timeouts.
//!
//! Three invariants drive the design.
//!
//! **Slots are released by dropping them.** A per-user concurrency permit lives
//! inside the queued request, so it comes back whether the request completes,
//! fails, times out, is cancelled, or is dropped because the client hung up. The
//! previous version incremented a counter on submit and decremented it on a
//! success path that a disconnected SSE client never reached; ten disconnects
//! locked the API out permanently.
//!
//! **Priority is only meaningful under contention.** Requests wait in a heap and
//! are pulled from it when a decode slot frees up, so a high-priority request that
//! arrives while the machine is busy runs before the low-priority ones queued
//! ahead of it. Sorting a batch and then immediately spawning every entry — which
//! is what the old loop did — orders nothing.
//!
//! **Cancellation is checked between tokens.** Generation is a loop the scheduler
//! drives, not an opaque call, so a dropped client stops the work rather than
//! paying for it to finish.

use crate::core::{GarudaError, Token};
use crate::runtime::{InferenceRuntime, SamplingParams, StopReason};
use parking_lot::Mutex;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Priority {
    Low,
    #[default]
    Normal,
    High,
}

impl std::str::FromStr for Priority {
    type Err = GarudaError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "low" => Ok(Priority::Low),
            "normal" | "" => Ok(Priority::Normal),
            "high" => Ok(Priority::High),
            other => Err(GarudaError::Config(format!(
                "unknown priority '{other}' (expected low, normal or high)"
            ))),
        }
    }
}

/// What the caller wants run.
#[derive(Debug, Clone)]
pub struct RequestSpec {
    pub user_id: String,
    pub prompt: Vec<Token>,
    pub params: SamplingParams,
    pub priority: Priority,
    pub timeout: Duration,
}

/// One event on a request's output stream.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token(Token),
    Done(StopReason),
    Error(GarudaError),
}

/// Set when the client goes away, checked between tokens.
#[derive(Debug, Clone, Default)]
struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// The caller's end of a submitted request.
///
/// Dropping it cancels the work. That is the whole cancellation story for HTTP:
/// axum drops the response stream when the client disconnects, which drops this,
/// which stops generation on the next token boundary.
#[derive(Debug)]
pub struct Handle {
    pub id: Uuid,
    pub events: mpsc::UnboundedReceiver<StreamEvent>,
    cancel: CancelFlag,
}

impl Handle {
    /// Stop generation explicitly. Idempotent.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

struct Request {
    id: Uuid,
    spec: RequestSpec,
    events: mpsc::UnboundedSender<StreamEvent>,
    cancel: CancelFlag,
    submitted: Instant,
    /// Returned to the user's pool when this request is dropped, on every path.
    _user_permit: OwnedSemaphorePermit,
}

/// Heap entry: highest priority first, FIFO within a priority.
struct Queued {
    request: Request,
    seq: u64,
}

impl PartialEq for Queued {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for Queued {}
impl PartialOrd for Queued {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Queued {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap: higher priority wins, and among equals the
        // lower sequence number (older request) wins, so we reverse that half.
        self.request
            .spec
            .priority
            .cmp(&other.request.spec.priority)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SchedulerConfig {
    /// Sequences decoding at once.
    pub max_concurrent: usize,
    /// Tokens to guess ahead when only one sequence is decoding. `0` disables it.
    ///
    /// Speculation and cross-request batching buy the same thing — more tokens per
    /// pass over the weights — so there is no point doing both. With several
    /// sequences in flight the batch already provides it; with one, nothing else can.
    pub speculative_lookahead: usize,
    /// Requests that may wait for a decode slot before submissions are refused.
    pub queue_capacity: usize,
    /// Concurrent requests one user may have in the system.
    pub max_concurrent_per_user: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 4,
            speculative_lookahead: 4,
            queue_capacity: 256,
            max_concurrent_per_user: 8,
        }
    }
}

#[derive(Debug, Default)]
struct Counters {
    submitted: AtomicU64,
    completed: AtomicU64,
    cancelled: AtomicU64,
    timed_out: AtomicU64,
    failed: AtomicU64,
    rejected_busy: AtomicU64,
    rejected_rate_limit: AtomicU64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SchedulerStats {
    pub submitted: u64,
    pub completed: u64,
    pub cancelled: u64,
    pub timed_out: u64,
    pub failed: u64,
    pub rejected_busy: u64,
    pub rejected_rate_limit: u64,
}

pub struct Scheduler {
    tx: mpsc::Sender<Queued>,
    user_slots: Mutex<HashMap<String, Arc<Semaphore>>>,
    config: SchedulerConfig,
    seq: AtomicU64,
    counters: Counters,
}

/// Beyond this many tracked users, drop the ones with nothing outstanding. Without
/// this, a client that invents a fresh user id per request grows the map forever.
const USER_TABLE_HIGH_WATER: usize = 1024;

impl Scheduler {
    pub fn new(runtime: Arc<InferenceRuntime>, config: SchedulerConfig) -> Arc<Self> {
        let (tx, rx) = mpsc::channel::<Queued>(config.queue_capacity.max(1));

        let scheduler = Arc::new(Self {
            tx,
            user_slots: Mutex::new(HashMap::new()),
            config,
            seq: AtomicU64::new(0),
            counters: Counters::default(),
        });

        tokio::spawn(run_loop(rx, runtime, scheduler.clone(), config));
        scheduler
    }

    pub fn stats(&self) -> SchedulerStats {
        let c = &self.counters;
        SchedulerStats {
            submitted: c.submitted.load(Ordering::Relaxed),
            completed: c.completed.load(Ordering::Relaxed),
            cancelled: c.cancelled.load(Ordering::Relaxed),
            timed_out: c.timed_out.load(Ordering::Relaxed),
            failed: c.failed.load(Ordering::Relaxed),
            rejected_busy: c.rejected_busy.load(Ordering::Relaxed),
            rejected_rate_limit: c.rejected_rate_limit.load(Ordering::Relaxed),
        }
    }

    fn user_semaphore(&self, user: &str) -> Arc<Semaphore> {
        let mut slots = self.user_slots.lock();

        if slots.len() > USER_TABLE_HIGH_WATER {
            let max = self.config.max_concurrent_per_user;
            slots.retain(|_, sem| sem.available_permits() < max);
        }

        slots
            .entry(user.to_owned())
            .or_insert_with(|| Arc::new(Semaphore::new(self.config.max_concurrent_per_user)))
            .clone()
    }

    /// Queue `spec`. The returned [`Handle`] streams events; dropping it cancels.
    ///
    /// Fails with [`GarudaError::RateLimit`] if the user is already at their
    /// concurrency limit, or [`GarudaError::Busy`] if the queue is full.
    pub fn submit(&self, spec: RequestSpec) -> Result<Handle, GarudaError> {
        let permit = self
            .user_semaphore(&spec.user_id)
            .try_acquire_owned()
            .map_err(|_| {
                self.counters
                    .rejected_rate_limit
                    .fetch_add(1, Ordering::Relaxed);
                GarudaError::RateLimit
            })?;

        let id = Uuid::new_v4();
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        let cancel = CancelFlag::default();

        let queued = Queued {
            seq: self.seq.fetch_add(1, Ordering::Relaxed),
            request: Request {
                id,
                spec,
                events: events_tx,
                cancel: cancel.clone(),
                submitted: Instant::now(),
                _user_permit: permit,
            },
        };

        // `try_send` rather than `send`: a full queue must be refused now, not
        // absorbed into unbounded memory. The permit drops with `queued` on failure.
        self.tx.try_send(queued).map_err(|e| {
            self.counters.rejected_busy.fetch_add(1, Ordering::Relaxed);
            match e {
                mpsc::error::TrySendError::Full(_) => GarudaError::Busy,
                mpsc::error::TrySendError::Closed(_) => {
                    GarudaError::Scheduler("scheduler has shut down".into())
                }
            }
        })?;

        self.counters.submitted.fetch_add(1, Ordering::Relaxed);
        Ok(Handle {
            id,
            events: events_rx,
            cancel,
        })
    }
}

/// Prompt tokens absorbed per scheduler iteration before anything is known about
/// what they cost, and the range the measured estimate is held to.
///
/// Prefill is unbounded in a way decode is not — its cost is whatever prompt the
/// caller sent. Absorbing a long one in a single burst stalls every request already
/// streaming for its whole duration, which is precisely the latency spike batching
/// was supposed to remove. Feeding it in a piece at a time bounds that stall to one
/// piece.
///
/// How big a piece should be is not a constant, though: 32 tokens is imperceptible
/// on a small model and seconds on a large one. [`Pacing`] measures both sides and
/// sizes each chunk to cost about one decode step, so the two interleave evenly
/// whatever the model. These bounds only keep a wild estimate in check.
const PREFILL_TOKENS_INITIAL: usize = 32;
const PREFILL_TOKENS_MIN: usize = 1;
const PREFILL_TOKENS_MAX: usize = 512;

/// Running estimates of what a decode step and a prefill token cost.
///
/// The scheduler alternates between the two, so the chunk that keeps them balanced
/// is the one whose work matches a decode step's: a newly admitted request then gets
/// roughly half the machine while the requests already streaming keep the other
/// half, instead of one starving the other. Both are exponential moving averages —
/// the per-token costs drift as a batch grows and shrinks.
#[derive(Debug, Default, Clone, Copy)]
struct Pacing {
    decode_step: Option<f64>,
    prefill_token: Option<f64>,
}

impl Pacing {
    /// Weight on the newest sample. Responsive enough to follow a batch changing
    /// size, damped enough not to chase one slow step.
    const ALPHA: f64 = 0.25;

    fn blend(slot: &mut Option<f64>, sample: f64) {
        *slot = Some(match *slot {
            Some(prev) => prev * (1.0 - Self::ALPHA) + sample * Self::ALPHA,
            None => sample,
        });
    }

    fn observe_decode(&mut self, elapsed: Duration) {
        Self::blend(&mut self.decode_step, elapsed.as_secs_f64());
    }

    fn observe_prefill(&mut self, elapsed: Duration, tokens: usize) {
        if tokens > 0 {
            Self::blend(
                &mut self.prefill_token,
                elapsed.as_secs_f64() / tokens as f64,
            );
        }
    }

    /// Prompt tokens to absorb this iteration, given something is decoding.
    fn chunk(&self) -> usize {
        match (self.decode_step, self.prefill_token) {
            (Some(step), Some(per_token)) if per_token > 0.0 => {
                ((step / per_token) as usize).clamp(PREFILL_TOKENS_MIN, PREFILL_TOKENS_MAX)
            }
            _ => PREFILL_TOKENS_INITIAL,
        }
    }
}

/// What the decode worker owns between iterations.
struct Batch {
    active: Vec<Active>,
    prefilling: Vec<Prefilling>,
    pacing: Pacing,
}

/// A request that has been admitted and whose prompt is still going in.
struct Prefilling {
    request: Request,
    pending: crate::runtime::Pending,
    deadline: Instant,
}

/// A request that has been admitted and is now decoding.
struct Active {
    request: Request,
    session: crate::runtime::Session,
    deadline: Instant,
}

/// Pull requests into a priority heap, admit up to `max_concurrent` of them, and
/// step every admitted sequence together.
///
/// The sequences share one pass over the weights per step, which for a large model
/// is most of what a token costs — so `max_concurrent` concurrent requests cost far
/// less than `max_concurrent` times one. With `max_concurrent = 1` the batch is
/// always a single sequence and this is exactly what it replaced.
async fn run_loop(
    mut rx: mpsc::Receiver<Queued>,
    runtime: Arc<InferenceRuntime>,
    scheduler: Arc<Scheduler>,
    config: SchedulerConfig,
) {
    let max_active = config.max_concurrent.max(1);
    let mut heap: BinaryHeap<Queued> = BinaryHeap::new();
    let mut batch = Batch {
        active: Vec::new(),
        prefilling: Vec::new(),
        pacing: Pacing::default(),
    };

    loop {
        // Only ever block when there is genuinely nothing to do. Waiting here rather
        // than admitting eagerly is what gives priority its meaning: the heap keeps
        // filling while the current batch decodes.
        if batch.active.is_empty() && batch.prefilling.is_empty() && heap.is_empty() {
            match rx.recv().await {
                Some(q) => heap.push(q),
                None => break, // All senders dropped: the scheduler is gone.
            }
        }
        while let Ok(q) = rx.try_recv() {
            heap.push(q);
        }

        // Drop queued requests whose client has already gone, rather than waiting for
        // a decode slot to free up first. Their user's concurrency permit lives
        // inside the entry, so holding a dead request holds a live caller's slot —
        // which is how a burst of disconnects used to lock a user out. Re-collecting
        // restores the heap order.
        if heap
            .iter()
            .any(|q| q.request.cancel.is_cancelled() || q.request.events.is_closed())
        {
            let counters = &scheduler.counters;
            heap = heap
                .drain()
                .filter(|q| {
                    let gone = q.request.cancel.is_cancelled() || q.request.events.is_closed();
                    if gone {
                        counters.cancelled.fetch_add(1, Ordering::Relaxed);
                    }
                    !gone
                })
                .collect();
        }

        let mut admit = Vec::new();
        while batch.active.len() + batch.prefilling.len() + admit.len() < max_active {
            let Some(Queued { request, .. }) = heap.pop() else {
                break;
            };
            // Cancelled or abandoned while queued: skip it. Dropping `request`
            // returns the user's permit.
            if request.cancel.is_cancelled() || request.events.is_closed() {
                scheduler.counters.cancelled.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            admit.push(request);
        }

        // Admission prefills and the step is a forward pass: both are CPU-bound and
        // must not run on an async executor thread.
        let rt = runtime.clone();
        let sch = scheduler.clone();
        let joined = tokio::task::spawn_blocking(move || {
            let mut batch = batch;
            for request in admit {
                admit_one(&rt, &sch, &mut batch.prefilling, request);
            }
            // Nothing is streaming, so there is no one to stall: take the whole
            // prompt rather than dribbling it in over many iterations.
            let budget = if batch.active.is_empty() {
                usize::MAX
            } else {
                batch.pacing.chunk()
            };

            let started = Instant::now();
            let absorbed =
                advance_prefills(&rt, &sch, &mut batch.prefilling, &mut batch.active, budget);
            if absorbed > 0 {
                batch.pacing.observe_prefill(started.elapsed(), absorbed);
            }

            let started = Instant::now();
            if step_batch(&rt, &sch, &mut batch.active, config.speculative_lookahead) {
                batch.pacing.observe_decode(started.elapsed());
            }
            batch
        })
        .await;

        batch = match joined {
            Ok(b) => b,
            Err(e) => {
                tracing::error!(error = %e, "decode worker died; dropping its batch");
                Batch {
                    active: Vec::new(),
                    prefilling: Vec::new(),
                    pacing: Pacing::default(),
                }
            }
        };
    }
}

/// Accept one request, or report why it could not start. No forward pass happens
/// here — the prompt goes in through [`advance_prefills`].
fn admit_one(
    runtime: &InferenceRuntime,
    scheduler: &Scheduler,
    prefilling: &mut Vec<Prefilling>,
    request: Request,
) {
    match runtime.start_incremental(&request.spec.prompt, &request.spec.params) {
        Ok(pending) => {
            let deadline = request.submitted + request.spec.timeout;
            prefilling.push(Prefilling {
                request,
                pending,
                deadline,
            });
        }
        Err(e) => {
            scheduler.counters.failed.fetch_add(1, Ordering::Relaxed);
            let _ = request.events.send(StreamEvent::Error(e));
        }
    }
}

/// Feed each pending prompt a little further, promoting the ones that finish.
///
/// Cancellation and timeouts are checked here too, so a client that hangs up during
/// a long prefill stops it rather than paying for the rest of the prompt.
fn advance_prefills(
    runtime: &InferenceRuntime,
    scheduler: &Scheduler,
    prefilling: &mut Vec<Prefilling>,
    active: &mut Vec<Active>,
    budget: usize,
) -> usize {
    if prefilling.is_empty() {
        return 0;
    }
    let counters = &scheduler.counters;
    let now = Instant::now();
    let mut still = Vec::with_capacity(prefilling.len());
    let mut absorbed = 0;

    for mut p in std::mem::take(prefilling) {
        if p.request.cancel.is_cancelled() {
            counters.cancelled.fetch_add(1, Ordering::Relaxed);
            let _ = p
                .request
                .events
                .send(StreamEvent::Error(GarudaError::Cancelled));
            continue;
        }
        if p.request.events.is_closed() {
            counters.cancelled.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        if now >= p.deadline {
            counters.timed_out.fetch_add(1, Ordering::Relaxed);
            let _ = p
                .request
                .events
                .send(StreamEvent::Error(GarudaError::Timeout));
            continue;
        }

        let before = p.pending.remaining();
        let outcome = runtime.advance_prefill(&mut p.pending, budget);
        absorbed += before.saturating_sub(p.pending.remaining());
        match outcome {
            Ok(true) => active.push(Active {
                session: runtime.finish_prefill(p.pending),
                request: p.request,
                deadline: p.deadline,
            }),
            Ok(false) => still.push(p),
            Err(e) => {
                counters.failed.fetch_add(1, Ordering::Relaxed);
                let _ = p.request.events.send(StreamEvent::Error(e));
            }
        }
    }
    *prefilling = still;
    absorbed
}

/// Advance every active sequence by one token, retiring those that are done.
fn step_batch(
    runtime: &InferenceRuntime,
    scheduler: &Scheduler,
    active: &mut Vec<Active>,
    lookahead: usize,
) -> bool {
    let counters = &scheduler.counters;

    // Cancellation and timeouts are checked between tokens, so a dropped client stops
    // the work rather than paying for it to finish.
    let now = Instant::now();
    active.retain(|a| {
        if a.request.cancel.is_cancelled() {
            counters.cancelled.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(request = %a.request.id, "cancelled");
            let _ = a
                .request
                .events
                .send(StreamEvent::Error(GarudaError::Cancelled));
            return false;
        }
        // A closed receiver means the client is gone. Stop; do not spend the rest of
        // the budget generating tokens nobody will read.
        if a.request.events.is_closed() {
            counters.cancelled.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        if now >= a.deadline {
            counters.timed_out.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(request = %a.request.id, "timed out");
            let _ = a
                .request
                .events
                .send(StreamEvent::Error(GarudaError::Timeout));
            return false;
        }
        true
    });
    if active.is_empty() {
        return false;
    }

    // One sequence has nothing to share a pass with, so it guesses ahead instead;
    // several already share one, and speculating on top would only add work.
    if active.len() == 1 && lookahead > 0 {
        let a = &mut active[0];
        let params = a.request.spec.params;
        let mut tokens = Vec::new();
        let outcome =
            runtime.next_tokens_speculative(&mut a.session, &params, lookahead, &mut tokens);

        let mut gone = false;
        for t in tokens {
            if a.request.events.send(StreamEvent::Token(t)).is_err() {
                counters.cancelled.fetch_add(1, Ordering::Relaxed);
                gone = true;
                break;
            }
        }
        if let Err(reason) = outcome {
            if !gone {
                counters.completed.fetch_add(1, Ordering::Relaxed);
                let _ = a.request.events.send(StreamEvent::Done(reason));
            }
            gone = true;
        }
        if gone {
            active.clear();
        }
        return true;
    }

    let params: Vec<SamplingParams> = active.iter().map(|a| a.request.spec.params).collect();
    let mut sessions: Vec<&mut crate::runtime::Session> =
        active.iter_mut().map(|a| &mut a.session).collect();
    let outcomes = runtime.next_token_batch(&mut sessions, &params);
    drop(sessions);

    let mut done = Vec::with_capacity(outcomes.len());
    for (a, outcome) in active.iter().zip(outcomes) {
        done.push(match outcome {
            Ok(token) => {
                if a.request.events.send(StreamEvent::Token(token)).is_err() {
                    counters.cancelled.fetch_add(1, Ordering::Relaxed);
                    true
                } else {
                    false
                }
            }
            Err(reason) => {
                counters.completed.fetch_add(1, Ordering::Relaxed);
                let _ = a.request.events.send(StreamEvent::Done(reason));
                true
            }
        });
    }

    let mut keep = done.into_iter().map(|d| !d);
    active.retain(|_| keep.next().expect("one flag per active request"));
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KvConfig;
    use crate::core::{Expert, ModelDims, StorageBackend};
    use crate::memory::MemoryManager;
    use crate::moe::MoeEngine;
    use crate::router::{Router, RouterType};
    use crate::storage::LocalStorageBackend;
    use crate::tokenizer::Tokenizer;
    use crate::weights::ModelWeights;

    fn runtime(tag: &str) -> (Arc<InferenceRuntime>, std::path::PathBuf) {
        let dims = ModelDims::default();
        let dir = std::env::temp_dir().join(format!("garuda_sched_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let l2: Arc<dyn StorageBackend> = Arc::new(LocalStorageBackend::new(&dir));
        let budget = Expert::n_params(&dims) * 4 * dims.n_experts;
        let mm = Arc::new(MemoryManager::new(dims, budget, l2, None).unwrap());
        let weights = Arc::new(ModelWeights::synthesize(dims).unwrap());
        let router = Router::new(RouterType::Mixtral, dims).unwrap();
        let engine = Arc::new(MoeEngine::new(dims, weights, router, mm, None).unwrap());

        let kv = KvConfig::mha(dims, 256, 64, None, None);
        let rt = InferenceRuntime::new(Arc::new(Tokenizer::new()), engine, kv, 16);
        (Arc::new(rt), dir)
    }

    fn spec(user: &str, prompt: &str, max_tokens: usize) -> RequestSpec {
        RequestSpec {
            user_id: user.to_owned(),
            prompt: Tokenizer::new().encode(prompt),
            params: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                max_tokens,
                seed: Some(1),
            },
            priority: Priority::Normal,
            timeout: Duration::from_secs(10),
        }
    }

    async fn collect(mut h: Handle) -> (Vec<Token>, Option<StreamEvent>) {
        let mut tokens = Vec::new();
        let mut last = None;
        while let Some(ev) = h.events.recv().await {
            match ev {
                StreamEvent::Token(t) => tokens.push(t),
                other => {
                    last = Some(other);
                    break;
                }
            }
        }
        (tokens, last)
    }

    #[tokio::test]
    async fn a_request_streams_tokens_and_then_finishes() {
        let (rt, dir) = runtime("basic");
        let s = Scheduler::new(rt, SchedulerConfig::default());

        let h = s.submit(spec("u1", "hello", 6)).unwrap();
        let (tokens, last) = collect(h).await;

        assert!(!tokens.is_empty(), "no tokens were produced");
        assert!(tokens.len() <= 6);
        assert!(matches!(last, Some(StreamEvent::Done(_))), "got {last:?}");
        assert_eq!(s.stats().completed, 1);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn output_length_is_set_by_max_tokens_not_by_the_prompt() {
        let (rt, dir) = runtime("len");
        let s = Scheduler::new(rt, SchedulerConfig::default());

        let h = s.submit(spec("u1", "a", 5)).unwrap();
        let (short_prompt_out, _) = collect(h).await;

        let h = s
            .submit(spec("u1", "a considerably longer prompt here", 5))
            .unwrap();
        let (long_prompt_out, _) = collect(h).await;

        assert!(short_prompt_out.len() <= 5 && long_prompt_out.len() <= 5);
        assert!(
            long_prompt_out.len() < "a considerably longer prompt here".len(),
            "output still tracks prompt length"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn per_user_slots_come_back_when_a_client_disconnects() {
        // The bug this pins: dropping the handle mid-stream used to leak the slot,
        // and after `max_concurrent_per_user` disconnects the user was locked out
        // for the life of the process.
        let (rt, dir) = runtime("leak");
        let s = Scheduler::new(
            rt,
            SchedulerConfig {
                max_concurrent_per_user: 2,
                ..Default::default()
            },
        );

        // Disconnect as fast as the limiter allows. A `RateLimit` in here is the
        // documented answer while the user already has `max_concurrent_per_user` in
        // flight, and how quickly a permit comes back is a timing detail — the bug
        // being pinned is not that, it is never recovering at all.
        let mut disconnects = 0;
        for _ in 0..2000 {
            if let Ok(h) = s.submit(spec("victim", "a long prompt to generate from", 64)) {
                // Take the response and walk away, exactly as a disconnecting client does.
                drop(h);
                disconnects += 1;
                if disconnects == 20 {
                    break;
                }
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(disconnects, 20, "never got 20 requests in to disconnect");

        // Give the workers a moment to notice and release.
        for _ in 0..200 {
            if s.submit(spec("victim", "hi", 1)).is_ok() {
                let _ = std::fs::remove_dir_all(dir);
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("user is still locked out after disconnecting repeatedly");
    }

    #[tokio::test]
    async fn a_user_over_their_concurrency_limit_is_rate_limited() {
        let (rt, dir) = runtime("ratelimit");
        let s = Scheduler::new(
            rt,
            SchedulerConfig {
                max_concurrent: 1,
                max_concurrent_per_user: 2,
                ..Default::default()
            },
        );

        // Hold the handles so the permits stay taken.
        let _a = s
            .submit(spec("u1", "long prompt for slow work", 512))
            .unwrap();
        let _b = s
            .submit(spec("u1", "long prompt for slow work", 512))
            .unwrap();

        let err = s.submit(spec("u1", "third", 8)).unwrap_err();
        assert_eq!(err, GarudaError::RateLimit);
        assert_eq!(s.stats().rejected_rate_limit, 1);

        // A different user is unaffected.
        assert!(s.submit(spec("u2", "hello", 4)).is_ok());

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_full_queue_is_refused_rather_than_buffered() {
        let (rt, dir) = runtime("busy");
        let s = Scheduler::new(
            rt,
            SchedulerConfig {
                max_concurrent: 1,
                queue_capacity: 2,
                max_concurrent_per_user: 1000,
                ..Default::default()
            },
        );

        let mut held = Vec::new();
        let mut saw_busy = false;
        for i in 0..64 {
            match s.submit(spec(&format!("u{i}"), "a long prompt to keep it busy", 512)) {
                Ok(h) => held.push(h),
                Err(GarudaError::Busy) => {
                    saw_busy = true;
                    break;
                }
                Err(e) => panic!("unexpected {e:?}"),
            }
        }
        assert!(saw_busy, "an unbounded queue accepted everything");
        assert!(s.stats().rejected_busy >= 1);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn cancelling_stops_generation_early() {
        let (rt, dir) = runtime("cancel");
        let s = Scheduler::new(rt, SchedulerConfig::default());

        let mut h = s
            .submit(spec("u1", "generate a lot of tokens please", 4096))
            .unwrap();

        // Take a couple of tokens, then cancel.
        let mut seen = 0;
        while let Some(ev) = h.events.recv().await {
            if let StreamEvent::Token(_) = ev {
                seen += 1;
                if seen == 2 {
                    h.cancel();
                    break;
                }
            }
        }
        assert_eq!(seen, 2);

        // The stream ends promptly rather than running to 4096 tokens.
        let mut extra = 0;
        while let Some(ev) = h.events.recv().await {
            match ev {
                StreamEvent::Token(_) => extra += 1,
                StreamEvent::Error(GarudaError::Cancelled) => break,
                other => panic!("unexpected {other:?}"),
            }
            assert!(extra < 500, "cancellation was ignored");
        }
        assert!(s.stats().cancelled >= 1);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn pacing_sizes_a_chunk_to_cost_about_one_decode_step() {
        let mut p = Pacing::default();
        // Nothing measured yet: fall back to the starting guess.
        assert_eq!(p.chunk(), PREFILL_TOKENS_INITIAL);

        // A decode step costing 30 ms and prefill tokens costing 10 ms each means
        // three prompt tokens are worth about one step.
        p.observe_decode(Duration::from_millis(30));
        p.observe_prefill(Duration::from_millis(100), 10);
        assert_eq!(p.chunk(), 3);

        // A model ten times faster per prefill token takes ten times as many.
        let mut q = Pacing::default();
        q.observe_decode(Duration::from_millis(30));
        q.observe_prefill(Duration::from_millis(10), 10);
        assert_eq!(q.chunk(), 30);

        // And the estimate is held to sane bounds whatever the ratio says.
        let mut wild = Pacing::default();
        wild.observe_decode(Duration::from_secs(60));
        wild.observe_prefill(Duration::from_nanos(1), 1000);
        assert_eq!(wild.chunk(), PREFILL_TOKENS_MAX);

        let mut tiny = Pacing::default();
        tiny.observe_decode(Duration::from_nanos(1));
        tiny.observe_prefill(Duration::from_secs(1), 1);
        assert_eq!(tiny.chunk(), PREFILL_TOKENS_MIN);
    }

    #[test]
    fn pacing_smooths_rather_than_chasing_one_sample() {
        let mut p = Pacing::default();
        p.observe_decode(Duration::from_millis(100));
        // One outlier must not move the estimate all the way to it.
        p.observe_decode(Duration::from_millis(500));
        let blended = p.decode_step.unwrap();
        assert!(
            blended > 0.100 && blended < 0.250,
            "one sample moved the average to {blended}"
        );
    }

    #[tokio::test]
    async fn concurrent_requests_decode_together_and_each_gets_its_own_answer() {
        // The batching loop steps every admitted sequence in one pass. Sharing that
        // pass must not let sequences bleed into one another: each has its own cache,
        // sampler and budget, so a pinned seed must still reproduce what it produces
        // alone.
        let (rt, dir) = runtime("batched");
        let s = Scheduler::new(
            rt,
            SchedulerConfig {
                max_concurrent: 4,
                ..Default::default()
            },
        );

        // Run one alone first to get the reference answer.
        let solo = collect(s.submit(spec("u0", "alpha", 8)).unwrap()).await.0;

        // Now the same request alongside three different ones, so it is decoded as
        // part of a batch rather than by itself.
        let mut handles = vec![("alpha", s.submit(spec("u0", "alpha", 8)).unwrap())];
        for (i, prompt) in ["beta", "gamma", "delta"].iter().enumerate() {
            handles.push((
                *prompt,
                s.submit(spec(&format!("u{}", i + 1), prompt, 8)).unwrap(),
            ));
        }

        let mut outputs = Vec::new();
        for (name, h) in handles {
            let (tokens, last) = collect(h).await;
            assert!(
                matches!(last, Some(StreamEvent::Done(_))),
                "{name}: {last:?}"
            );
            outputs.push((name, tokens));
        }

        let batched_alpha = &outputs.iter().find(|(n, _)| *n == "alpha").unwrap().1;
        assert_eq!(
            *batched_alpha, solo,
            "a sequence decoded in a batch differs from the same one decoded alone"
        );
        // And the different prompts did not all collapse to the same answer.
        let beta = &outputs.iter().find(|(n, _)| *n == "beta").unwrap().1;
        assert_ne!(beta, batched_alpha, "two prompts produced identical output");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_request_that_outruns_its_timeout_is_stopped() {
        let (rt, dir) = runtime("timeout");
        let s = Scheduler::new(rt, SchedulerConfig::default());

        let mut sp = spec("u1", "a prompt", 100_000);
        sp.timeout = Duration::from_millis(50);

        let h = s.submit(sp).unwrap();
        let (_, last) = collect(h).await;

        assert!(
            matches!(last, Some(StreamEvent::Error(GarudaError::Timeout))),
            "got {last:?}"
        );
        assert_eq!(s.stats().timed_out, 1);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn high_priority_work_overtakes_low_priority_work_in_the_queue() {
        let (rt, dir) = runtime("priority");
        // A single decode slot, so everything after the first request must queue.
        let s = Scheduler::new(
            rt,
            SchedulerConfig {
                max_concurrent: 1,
                queue_capacity: 64,
                max_concurrent_per_user: 64,
                ..Default::default()
            },
        );

        // Occupy the only slot, and wait until it is genuinely running: submitting is
        // synchronous, so without this the scheduler loop may not have started yet and
        // nothing would actually be queued behind anything.
        let mut blocker = s
            .submit(spec("blocker", "a long running prompt", 100_000))
            .unwrap();
        match blocker.events.recv().await {
            Some(StreamEvent::Token(_)) => {}
            other => panic!("blocker did not start: {other:?}"),
        }

        // Queue four low-priority requests, then one high-priority one behind them.
        let mut pending = Vec::new();
        for i in 0..4 {
            let mut sp = spec(&format!("low{i}"), "hello", 2);
            sp.priority = Priority::Low;
            pending.push(("low", s.submit(sp).unwrap()));
        }
        let mut hi = spec("high", "hello", 2);
        hi.priority = Priority::High;
        pending.push(("high", s.submit(hi).unwrap()));

        // Free the slot.
        drop(blocker);

        // Drain them all concurrently and record the order they finish in.
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut tasks = Vec::new();
        for (label, handle) in pending {
            let order = order.clone();
            tasks.push(tokio::spawn(async move {
                let _ = collect(handle).await;
                order.lock().push(label);
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }

        let finished = order.lock().clone();
        assert_eq!(
            finished.first().copied(),
            Some("high"),
            "high priority did not overtake the queue: {finished:?}"
        );

        let _ = std::fs::remove_dir_all(dir);
    }
}

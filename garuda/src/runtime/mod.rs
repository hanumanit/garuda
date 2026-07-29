//! The inference runtime: prompt handling, the decode loop, and sampling.
//!
//! A [`Session`] holds one sequence. The scheduler drives it one token at a time
//! via [`InferenceRuntime::next_token`], which is what makes cancellation, timeouts
//! and streaming possible: control returns to the caller between every token.

use crate::cache::{KvConfig, PromptCache, SeqState};
use crate::core::{GarudaError, InferenceBackend, ModelDims, Tensor, Token};
use crate::tokenizer::Tokenize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// How to turn logits into a token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    /// `0.0` means greedy (argmax); the other knobs are then ignored.
    pub temperature: f32,
    /// Nucleus sampling: keep the smallest set of tokens whose probability sums to
    /// `top_p`. `1.0` disables it.
    pub top_p: f32,
    /// Keep only the `top_k` most likely tokens. `0` disables it.
    pub top_k: usize,
    pub max_tokens: usize,
    /// `None` draws a seed from the sequence id, so a request is reproducible only
    /// if the caller pins it.
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            max_tokens: 128,
            seed: None,
        }
    }
}

impl SamplingParams {
    pub fn validate(&self) -> Result<(), GarudaError> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(GarudaError::Config(format!(
                "temperature must be a non-negative number, got {}",
                self.temperature
            )));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(GarudaError::Config(format!(
                "top_p must be in (0, 1], got {}",
                self.top_p
            )));
        }
        if self.max_tokens == 0 {
            return Err(GarudaError::Config("max_tokens must be at least 1".into()));
        }
        Ok(())
    }
}

/// Small deterministic PRNG, so a seeded request replays exactly.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the zero state, which splitmix handles but which reads as "unseeded".
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Pick a token from `logits` under `params`.
pub fn sample(
    logits: &Tensor,
    params: &SamplingParams,
    rng: &mut Rng,
) -> Result<Token, GarudaError> {
    let data = logits.data();
    if data.is_empty() {
        return Err(GarudaError::Inference(
            "cannot sample from empty logits".into(),
        ));
    }

    if params.temperature == 0.0 {
        let (idx, _) =
            data.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv { (i, v) } else { (bi, bv) }
                });
        return Ok(idx as Token);
    }

    let candidates = candidates(data, params);
    let total: f32 = candidates.iter().map(|(_, p)| p).sum();
    if total <= 0.0 {
        // Every surviving candidate has zero mass (possible after an extreme
        // temperature). Fall back to the most likely token rather than to token 0.
        return Ok(candidates[0].0);
    }

    let mut point = rng.next_f32() * total;
    for (tok, p) in &candidates {
        point -= p;
        if point <= 0.0 {
            return Ok(*tok);
        }
    }
    Ok(candidates.last().expect("non-empty").0)
}

/// The candidates `sample` chooses between: temperature, then top-k, then top-p.
///
/// Weights are the softmax probabilities of the survivors, so they sum to at most
/// one — truncation removes mass rather than redistributing it. Returned rather than
/// consumed on the spot because verifying a speculated token needs to *read* this
/// distribution, not just draw from it.
fn candidates(data: &[f32], params: &SamplingParams) -> Vec<(Token, f32)> {
    // Temperature, then softmax to probabilities.
    let mut scaled: Vec<f32> = data.iter().map(|&v| v / params.temperature).collect();
    crate::simd::softmax(&mut scaled);

    let mut candidates: Vec<(Token, f32)> = scaled
        .iter()
        .enumerate()
        .map(|(i, &p)| (i as Token, p))
        .collect();

    // Most likely first; ties break on token id so the order is a strict total
    // order (ids are unique), which is what makes the selection below deterministic.
    let by_likelihood = |a: &(Token, f32), b: &(Token, f32)| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    };

    // Only the top-k survive, so partition to find them in O(vocab) rather than
    // sorting all of it: at a real model's 32k vocabulary a full sort per token is
    // ~480k comparisons to then throw all but 40 of the results away.
    if params.top_k > 0 && params.top_k < candidates.len() {
        candidates.select_nth_unstable_by(params.top_k - 1, by_likelihood);
        candidates.truncate(params.top_k);
    }
    candidates.sort_by(by_likelihood);

    if params.top_p < 1.0 {
        let mut cumulative = 0.0;
        let mut keep = 0;
        for (i, (_, p)) in candidates.iter().enumerate() {
            cumulative += p;
            keep = i + 1;
            if cumulative >= params.top_p {
                break;
            }
        }
        candidates.truncate(keep.max(1));
    }

    candidates
}

/// Decide a speculated token against the distribution the caller actually asked for.
///
/// Returns whether the guess survived, and the token to emit either way.
///
/// This is what makes guessing safe for a sampled request. The drafter is
/// deterministic — it proposes one token with certainty — so the standard
/// speculative-sampling rule reduces to: keep the guess with probability `p(guess)`,
/// and on rejection draw from `p` with the guess removed. Over many steps that emits
/// exactly `p`, which an equality test against the argmax does not: that would
/// silently hand a caller who asked for `temperature = 0.8` the greedy answer.
fn verify_drafted(cands: &[(Token, f32)], drafted: Token, rng: &mut Rng) -> (bool, Token) {
    let total: f32 = cands.iter().map(|(_, p)| p).sum();
    if total <= 0.0 {
        return (false, cands[0].0);
    }
    // Zero if top-k or top-p cut the guess away, which then always rejects — correct,
    // since the caller's distribution gives it no mass at all.
    let drafted_mass = cands
        .iter()
        .find(|(t, _)| *t == drafted)
        .map(|(_, p)| *p)
        .unwrap_or(0.0);

    if rng.next_f32() < drafted_mass / total {
        return (true, drafted);
    }

    let residual = total - drafted_mass;
    if residual <= 0.0 {
        // `p` was a point mass on the guess, so there is nothing else to draw.
        return (true, drafted);
    }
    let mut point = rng.next_f32() * residual;
    for (tok, p) in cands {
        if *tok == drafted {
            continue;
        }
        point -= p;
        if point <= 0.0 {
            return (false, *tok);
        }
    }
    (
        false,
        cands
            .iter()
            .rev()
            .find(|(t, _)| *t != drafted)
            .map(|(t, _)| *t)
            .unwrap_or(drafted),
    )
}

/// A sequence whose prompt is still going in.
///
/// Held between [`InferenceRuntime::start_incremental`] and
/// [`InferenceRuntime::finish_prefill`], so a scheduler can absorb a long prompt in
/// pieces rather than in one unbounded burst.
#[derive(Debug)]
pub struct Pending {
    seq: SeqState,
    /// The full prompt. Everything but its last token is prefilled.
    context: Vec<Token>,
    /// Prompt positions already in `seq`.
    consumed: usize,
    seed: u64,
    /// Whether the finished prefix is worth caching — false for one that came *from*
    /// the cache.
    cache_on_finish: bool,
}

impl Pending {
    /// Prompt tokens still to absorb before this can start decoding.
    pub fn remaining(&self) -> usize {
        (self.context.len() - 1).saturating_sub(self.consumed)
    }
}

/// Context tokens that must match for a guess to be drawn from an earlier passage.
///
/// Short enough to fire often, long enough that a match means something: two tokens
/// repeat everywhere, and eight almost never repeat outside genuinely echoed text.
const NGRAM: usize = 4;

/// Guess the next few tokens by finding where the recent context occurred earlier
/// and copying whatever followed it.
///
/// No draft model, no extra weights, no extra memory — which matters most on exactly
/// the machine speculation is for, one already running a checkpoint bigger than its
/// RAM. It only fires when the output echoes the input, so it does nothing for
/// open-ended prose and a great deal for summarising, editing, extraction and
/// anything grounded in a long prompt.
///
/// Returns an empty guess when it has no basis for one. That is the honest answer:
/// a wrong guess costs a wasted slot in the verification pass, and a guess made up
/// from nothing is wrong nearly always.
pub fn draft_from_context(context: &[Token], max_tokens: usize, ngram: usize) -> Vec<Token> {
    if max_tokens == 0 || ngram == 0 || context.len() <= ngram {
        return Vec::new();
    }
    let needle = &context[context.len() - ngram..];
    // Latest match first: recent context is the better predictor, and it also keeps
    // the scan short on the common case of a repeated phrase nearby.
    let search_end = context.len() - ngram;
    for start in (0..search_end).rev() {
        if &context[start..start + ngram] != needle {
            continue;
        }
        let from = start + ngram;
        let take = max_tokens.min(context.len() - from);
        if take > 0 {
            return context[from..from + take].to_vec();
        }
    }
    Vec::new()
}

/// One in-flight sequence.
#[derive(Debug)]
pub struct Session {
    seq: SeqState,
    /// Prompt followed by everything generated so far.
    context: Vec<Token>,
    prompt_len: usize,
    rng: Rng,
    finished: bool,
    /// Extra tokens the last few guesses actually won, as a moving average.
    ///
    /// A guess is not free: the verification pass computes every drafted position,
    /// so a batch of six wrong guesses does six positions of expert arithmetic to
    /// produce one token. Measured on Mixtral that is ~1.6x *slower* than not
    /// guessing. Where the guesses land it is ~3x faster. The gap is far too wide to
    /// settle with a fixed policy, so the sequence keeps score and stops drafting
    /// when it is losing.
    spec_gain: f64,
    /// Rounds since drafting was last attempted, so a sequence that gave up gets
    /// another look rather than being written off for good — text turns repetitive
    /// halfway through often enough to be worth rechecking.
    spec_idle: usize,
}

impl Session {
    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    pub fn generated(&self) -> usize {
        self.context.len() - self.prompt_len
    }

    pub fn generated_tokens(&self) -> &[Token] {
        &self.context[self.prompt_len..]
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The model emitted end-of-sequence.
    Eos,
    /// `max_tokens` was reached.
    Length,
    /// The context window is full.
    ContextFull,
}

impl StopReason {
    /// The `finish_reason` string OpenAI clients expect.
    pub fn as_openai(&self) -> &'static str {
        match self {
            StopReason::Eos => "stop",
            StopReason::Length | StopReason::ContextFull => "length",
        }
    }
}

pub struct InferenceRuntime {
    pub tokenizer: Arc<dyn Tokenize>,
    backend: Arc<dyn InferenceBackend>,
    prompt_cache: PromptCache,
    kv_template: KvConfig,
    max_context: usize,
    next_seq: AtomicU64,
}

impl InferenceRuntime {
    pub fn new(
        tokenizer: Arc<dyn Tokenize>,
        backend: Arc<dyn InferenceBackend>,
        kv_template: KvConfig,
        prompt_cache_capacity: usize,
        prompt_cache_bytes: usize,
    ) -> Self {
        let max_context = kv_template.max_positions;
        Self {
            tokenizer,
            backend,
            prompt_cache: PromptCache::new(prompt_cache_capacity, prompt_cache_bytes),
            kv_template,
            max_context,
            next_seq: AtomicU64::new(1),
        }
    }

    pub fn dims(&self) -> ModelDims {
        self.backend.dims()
    }

    pub fn max_context(&self) -> usize {
        self.max_context
    }

    pub fn prompt_cache_stats(&self) -> crate::cache::CacheStats {
        self.prompt_cache.stats()
    }

    fn fresh_seq_id(&self) -> u64 {
        self.next_seq.fetch_add(1, Ordering::Relaxed)
    }

    /// Begin a sequence, prefilling everything but the final prompt token.
    ///
    /// The prefix is what gets cached: the last token is always run fresh, because
    /// that is the step that produces the logits for the first sampled token.
    ///
    /// This absorbs the whole prompt before returning. A scheduler with other
    /// requests already streaming wants [`Self::start_incremental`] instead, so a
    /// long prompt does not stall them.
    pub fn start(&self, prompt: &[Token], params: &SamplingParams) -> Result<Session, GarudaError> {
        let mut pending = self.start_incremental(prompt, params)?;
        while !self.advance_prefill(&mut pending, usize::MAX)? {}
        Ok(self.finish_prefill(pending))
    }

    /// Begin a sequence without absorbing the prompt yet.
    ///
    /// Prefill is the one part of serving a request whose cost is set by the caller:
    /// a long prompt is many forward passes, and running them all before the next
    /// decode step stalls every request already streaming. Returning the half-built
    /// state lets a scheduler feed the prompt in a piece at a time and keep decoding
    /// in between — see [`Self::advance_prefill`].
    pub fn start_incremental(
        &self,
        prompt: &[Token],
        params: &SamplingParams,
    ) -> Result<Pending, GarudaError> {
        params.validate()?;

        if prompt.is_empty() {
            return Err(GarudaError::Inference("prompt is empty".into()));
        }
        if prompt.len() >= self.max_context {
            return Err(GarudaError::Inference(format!(
                "prompt of {} tokens does not fit the {}-token context window",
                prompt.len(),
                self.max_context
            )));
        }

        let seq_id = self.fresh_seq_id();
        let prefix = &prompt[..prompt.len() - 1];

        // A `match` rather than an `else if let` chain: under the 2024 edition the
        // latter would drop the `get` temporary at a different point, and the
        // explicit form keeps the prompt-cache lookup scope unambiguous.
        let (seq, consumed, cache_on_finish) = if prefix.is_empty() {
            (SeqState::new(self.kv_template.clone(), seq_id), 0, false)
        } else {
            match self.prompt_cache.get(prefix, seq_id) {
                // A hit arrives with the whole prefix already in it, and re-inserting
                // what was just read would be pure churn.
                Some(cached) => (cached, prefix.len(), false),
                None => (SeqState::new(self.kv_template.clone(), seq_id), 0, true),
            }
        };

        Ok(Pending {
            seq,
            context: prompt.to_vec(),
            consumed,
            seed: params.seed.unwrap_or(seq_id),
            cache_on_finish,
        })
    }

    /// Absorb up to `max_tokens` more of a pending prompt. `Ok(true)` once it is all
    /// in and the sequence is ready for [`Self::finish_prefill`].
    ///
    /// `usize::MAX` absorbs the rest in one call, which is what a scheduler with
    /// nothing else running should do — there is no one to stall.
    pub fn advance_prefill(
        &self,
        pending: &mut Pending,
        max_tokens: usize,
    ) -> Result<bool, GarudaError> {
        let prefix_len = pending.context.len() - 1;
        if pending.consumed >= prefix_len {
            return Ok(true);
        }

        let target = prefix_len.min(pending.consumed.saturating_add(max_tokens.max(1)));
        // `hidden` rather than `logits`: the prefix's logits are thrown away, so the
        // output head would be a `vocab x d_model` matmul done for nothing. The
        // backend consumes only the positions it has not seen, so feeding a growing
        // prefix is how a prompt goes in a piece at a time.
        self.backend
            .hidden(&pending.context[..target], &mut pending.seq)?;
        pending.consumed = target;

        let done = pending.consumed >= prefix_len;
        if done && pending.cache_on_finish && !pending.seq.has_spill() {
            self.prompt_cache
                .insert(&pending.context[..prefix_len], pending.seq.clone());
        }
        Ok(done)
    }

    /// Turn a fully-absorbed prompt into the session that decodes from it.
    pub fn finish_prefill(&self, pending: Pending) -> Session {
        Session {
            seq: pending.seq,
            prompt_len: pending.context.len(),
            context: pending.context,
            rng: Rng::new(pending.seed),
            finished: false,
            // Start optimistic: one round of drafting is cheap to try and tells us
            // more than any guess about the workload could.
            spec_gain: 1.0,
            spec_idle: 0,
        }
    }

    /// The model's pooled representation of `tokens`, L2-normalised.
    ///
    /// This is the real final hidden state, not a placeholder vector. It is still
    /// useless for semantic search — the weights are untrained, so two similar
    /// sentences have no reason to land near each other. It is the right *shape* of
    /// answer, computed the right way, from a model that knows nothing.
    pub fn embed(&self, tokens: &[Token]) -> Result<Vec<f32>, GarudaError> {
        if tokens.is_empty() {
            return Err(GarudaError::Inference("cannot embed an empty input".into()));
        }
        if tokens.len() > self.max_context {
            return Err(GarudaError::Inference(format!(
                "input of {} tokens does not fit the {}-token context window",
                tokens.len(),
                self.max_context
            )));
        }

        let mut seq = SeqState::new(self.kv_template.clone(), self.fresh_seq_id());
        let hidden = self.backend.hidden(tokens, &mut seq)?;

        let mut v = hidden.into_data();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        Ok(v)
    }

    /// Produce the next token, or `Err(reason)` when the sequence is done.
    ///
    /// Returns `Ok(None)` never — a finished sequence reports *why* it finished, so
    /// the API layer can set `finish_reason` honestly instead of guessing "stop".
    pub fn next_token(
        &self,
        session: &mut Session,
        params: &SamplingParams,
    ) -> Result<Token, StopReason> {
        if session.finished {
            return Err(StopReason::Length);
        }
        if session.generated() >= params.max_tokens {
            session.finished = true;
            return Err(StopReason::Length);
        }
        if session.context.len() >= self.max_context {
            session.finished = true;
            return Err(StopReason::ContextFull);
        }

        let logits = match self.backend.logits(&session.context, &mut session.seq) {
            Ok(l) => l,
            Err(e) => {
                session.finished = true;
                tracing::warn!(error = %e, "forward pass failed");
                return Err(StopReason::ContextFull);
            }
        };

        let token = match sample(&logits, params, &mut session.rng) {
            Ok(t) => t,
            Err(e) => {
                session.finished = true;
                tracing::warn!(error = %e, "sampling failed");
                return Err(StopReason::ContextFull);
            }
        };

        session.context.push(token);

        if token == self.tokenizer.eos() {
            session.finished = true;
            return Err(StopReason::Eos);
        }
        Ok(token)
    }

    /// Produce the next token, and as many after it as a guess gets right.
    ///
    /// Identical in output to [`Self::next_token`] — a guess is only ever kept where
    /// the model would have chosen it anyway — but a run of `j` accepted guesses
    /// costs one pass over the weights instead of `j + 1`. On a checkpoint larger
    /// than RAM a pass is gigabytes of paging, which is the whole difference between
    /// one token per read and several.
    ///
    /// Greedy requests get back exactly what plain decoding produces, token for
    /// token. Sampled ones get the same *distribution*: a guess is kept with the
    /// probability the caller's distribution assigns it, and otherwise replaced by a
    /// draw from what remains — the standard speculative-sampling rule, reduced by
    /// the drafter being deterministic. What that does not preserve
    /// is the particular sequence a given seed produces — speculating consumes the
    /// generator differently, so a seeded sampled request reproduces itself but not
    /// its non-speculative twin.
    ///
    /// Tokens land in `out`. `Err` means the sequence finished, exactly as
    /// `next_token` reports it, and any tokens produced before that are still in
    /// `out`.
    pub fn next_tokens_speculative(
        &self,
        session: &mut Session,
        params: &SamplingParams,
        lookahead: usize,
        out: &mut Vec<Token>,
    ) -> Result<(), StopReason> {
        if lookahead == 0 || !self.backend.speculation_supported() {
            return self.next_token(session, params).map(|t| out.push(t));
        }
        if session.finished {
            return Err(StopReason::Length);
        }
        if session.generated() >= params.max_tokens {
            session.finished = true;
            return Err(StopReason::Length);
        }
        if session.context.len() >= self.max_context {
            session.finished = true;
            return Err(StopReason::ContextFull);
        }

        // How many guesses are worth making: no more than the budget or the window
        // can take, since a token accepted past either would have to be thrown away.
        let room = params
            .max_tokens
            .saturating_sub(session.generated())
            .min(self.max_context.saturating_sub(session.context.len()))
            .saturating_sub(1);
        // Ask for a little more than has been landing rather than for the maximum
        // every time. The verification pass computes every position drafted, so six
        // guesses that win one token do six positions of expert arithmetic for it —
        // measured on Mixtral, ~1.6x slower than not guessing, against ~3x faster
        // where the guesses land. Sizing the request to the observed yield is what
        // keeps the bad case from being paid for.
        const WORTH_IT: f64 = 0.35;
        const RETRY_AFTER: usize = 16;

        let budget = if session.spec_gain < WORTH_IT && session.spec_idle < RETRY_AFTER {
            session.spec_idle += 1;
            0
        } else {
            session.spec_idle = 0;
            (session.spec_gain.ceil() as usize + 1).min(lookahead)
        };
        let draft = draft_from_context(&session.context, budget.min(room), NGRAM);

        // One pass over `context ++ draft`. Answer `i` is the token the model would
        // produce having seen the first `i` guesses — so answer 0 is the real next
        // token, and answer `i` is only reachable if guesses `0..i` were all right.
        let n = draft.len() + 1;
        let mut probe = session.context.clone();
        probe.extend_from_slice(&draft);
        let answers = match self.backend.logits_multi(&probe, &mut session.seq, n) {
            Ok(a) => a,
            Err(e) => {
                session.finished = true;
                tracing::warn!(error = %e, "forward pass failed");
                return Err(StopReason::ContextFull);
            }
        };

        let mut stop = None;
        for (i, answer) in answers.iter().enumerate() {
            // Answer `i` is the model's verdict on guess `i`; the last has no guess to
            // check and simply supplies the next token.
            let (kept, token) = match draft.get(i) {
                None => match sample(answer, params, &mut session.rng) {
                    Ok(t) => (false, t),
                    Err(e) => {
                        tracing::warn!(error = %e, "sampling failed");
                        stop = Some(StopReason::ContextFull);
                        break;
                    }
                },
                Some(&guess) if params.temperature == 0.0 => {
                    // Greedy: the model's choice either is the guess or is not, and
                    // keeping it when it matches cannot change the answer at all.
                    match sample(answer, params, &mut session.rng) {
                        Ok(t) => (t == guess, t),
                        Err(e) => {
                            tracing::warn!(error = %e, "sampling failed");
                            stop = Some(StopReason::ContextFull);
                            break;
                        }
                    }
                }
                Some(&guess) => {
                    let cands = candidates(answer.data(), params);
                    if cands.is_empty() {
                        stop = Some(StopReason::ContextFull);
                        break;
                    }
                    verify_drafted(&cands, guess, &mut session.rng)
                }
            };

            session.context.push(token);
            if token == self.tokenizer.eos() {
                stop = Some(StopReason::Eos);
                break;
            }
            out.push(token);
            if !kept {
                break;
            }
            if session.generated() >= params.max_tokens {
                stop = Some(StopReason::Length);
                break;
            }
        }

        // Score the round: how many tokens beyond the one a plain step would have
        // produced. Smoothed, so a single unlucky guess does not switch drafting off
        // and a single lucky one does not switch it back on.
        if !draft.is_empty() {
            let won = (out.len() + usize::from(stop.is_some())).saturating_sub(1) as f64;
            session.spec_gain = session.spec_gain * 0.7 + won * 0.3;
        }

        // The cache consumed every guess. Give back the positions belonging to tokens
        // that were never produced, restoring the invariant the ordinary path keeps:
        // one fewer cache position than context tokens.
        let keep = session.context.len().saturating_sub(1);
        if session.seq.truncate(keep).is_err() {
            session.finished = true;
            return Err(StopReason::ContextFull);
        }

        match stop {
            Some(reason) => {
                session.finished = true;
                Err(reason)
            }
            None => Ok(()),
        }
    }

    /// One decode step for several sequences at once, one result each in order.
    ///
    /// The sequences are independent — separate caches, separate samplers, separate
    /// budgets — so this produces exactly what stepping each one alone produces. What
    /// it buys is that they share a single pass over the weights, which for a large
    /// model is most of the cost of a token.
    ///
    /// Sessions that are already finished, out of budget or out of context never
    /// reach the backend; only the live ones are batched.
    pub fn next_token_batch(
        &self,
        sessions: &mut [&mut Session],
        params: &[SamplingParams],
    ) -> Vec<Result<Token, StopReason>> {
        let n = sessions.len();
        debug_assert_eq!(n, params.len());

        // Decide who is still running, and retire the rest, before any work.
        let mut stopped: Vec<Option<StopReason>> = Vec::with_capacity(n);
        for (s, p) in sessions.iter_mut().zip(params) {
            stopped.push(if s.finished {
                Some(StopReason::Length)
            } else if s.generated() >= p.max_tokens {
                s.finished = true;
                Some(StopReason::Length)
            } else if s.context.len() >= self.max_context {
                s.finished = true;
                Some(StopReason::ContextFull)
            } else {
                None
            });
        }

        let mut contexts: Vec<&[Token]> = Vec::new();
        let mut seqs: Vec<&mut crate::cache::SeqState> = Vec::new();
        let mut live: Vec<usize> = Vec::new();
        for (i, s) in sessions.iter_mut().enumerate() {
            if stopped[i].is_some() {
                continue;
            }
            // Disjoint fields of the same session: the context is read while the
            // cache is written, which is the whole shape of a decode step.
            let Session { context, seq, .. } = &mut **s;
            contexts.push(context.as_slice());
            seqs.push(seq);
            live.push(i);
        }

        if live.is_empty() {
            return stopped
                .into_iter()
                .map(|r| Err(r.expect("all stopped")))
                .collect();
        }

        let batched = self.backend.logits_batch(&contexts, &mut seqs);
        drop(contexts);
        drop(seqs);

        let logits = match batched {
            Ok(l) => l,
            Err(e) => {
                // One bad sequence must not be reported as everyone's failure, and a
                // batch is only ever an optimisation — so fall back to stepping each
                // live sequence alone, which attributes the error where it belongs.
                tracing::warn!(error = %e, "batched forward failed; stepping individually");
                let mut out: Vec<Option<Result<Token, StopReason>>> = vec![None; n];
                for (i, r) in stopped.iter().enumerate() {
                    if let Some(reason) = r {
                        out[i] = Some(Err(*reason));
                    }
                }
                for &i in &live {
                    out[i] = Some(self.next_token(sessions[i], &params[i]));
                }
                return out
                    .into_iter()
                    .map(|r| r.expect("every slot filled"))
                    .collect();
            }
        };

        let mut out: Vec<Option<Result<Token, StopReason>>> = vec![None; n];
        for (i, r) in stopped.iter().enumerate() {
            if let Some(reason) = r {
                out[i] = Some(Err(*reason));
            }
        }
        for (k, &i) in live.iter().enumerate() {
            let s = &mut *sessions[i];
            out[i] = Some(match sample(&logits[k], &params[i], &mut s.rng) {
                Ok(token) => {
                    s.context.push(token);
                    if token == self.tokenizer.eos() {
                        s.finished = true;
                        Err(StopReason::Eos)
                    } else {
                        Ok(token)
                    }
                }
                Err(e) => {
                    s.finished = true;
                    tracing::warn!(error = %e, "sampling failed");
                    Err(StopReason::ContextFull)
                }
            });
        }
        out.into_iter()
            .map(|r| r.expect("every slot filled"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Expert, StorageBackend};
    use crate::memory::MemoryManager;
    use crate::moe::MoeEngine;
    use crate::router::{Router, RouterType};
    use crate::storage::LocalStorageBackend;
    use crate::weights::ModelWeights;

    fn runtime(tag: &str) -> (InferenceRuntime, std::path::PathBuf) {
        let dims = ModelDims::default();
        let dir = std::env::temp_dir().join(format!("garuda_rt_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let l2: Arc<dyn StorageBackend> = Arc::new(LocalStorageBackend::new(&dir));
        let budget = Expert::n_params(&dims) * 4 * dims.n_experts;
        let mm = Arc::new(MemoryManager::new(dims, budget, l2, None).unwrap());
        let weights = Arc::new(ModelWeights::synthesize(dims).unwrap());
        let router = Router::new(RouterType::Mixtral, dims).unwrap();
        let engine = Arc::new(MoeEngine::new(dims, weights, router, mm, None).unwrap());

        let kv = KvConfig::mha(dims, 128, 64, None, None);
        let tk = Arc::new(crate::tokenizer::Tokenizer::new());
        (InferenceRuntime::new(tk, engine, kv, 8, 64 << 20), dir)
    }

    fn greedy(max_tokens: usize) -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            max_tokens,
            seed: Some(7),
        }
    }

    fn drain(
        rt: &InferenceRuntime,
        s: &mut Session,
        p: &SamplingParams,
    ) -> (Vec<Token>, StopReason) {
        let mut out = Vec::new();
        loop {
            match rt.next_token(s, p) {
                Ok(t) => out.push(t),
                Err(r) => return (out, r),
            }
        }
    }

    #[test]
    fn greedy_sampling_takes_the_argmax() {
        let logits = Tensor::vector(vec![0.1, 5.0, -2.0, 4.9]).clone();
        let mut rng = Rng::new(1);
        let p = SamplingParams {
            temperature: 0.0,
            ..greedy(1)
        };
        assert_eq!(sample(&logits, &p, &mut rng).unwrap(), 1);
    }

    #[test]
    fn top_k_of_one_is_deterministic_regardless_of_seed() {
        let logits = Tensor::vector(vec![0.1, 5.0, -2.0, 4.9]);
        let p = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            max_tokens: 1,
            seed: None,
        };
        for seed in 0..20 {
            let mut rng = Rng::new(seed);
            assert_eq!(sample(&logits, &p, &mut rng).unwrap(), 1);
        }
    }

    #[test]
    fn top_k_selection_picks_the_same_tokens_a_full_sort_would() {
        // `sample` partitions instead of sorting the whole vocabulary. The surviving
        // set, and the order within it, must be exactly what the old full sort gave.
        let vocab = 4096;
        let logits: Vec<f32> = (0..vocab)
            .map(|i| ((i * 7919 % 1000) as f32 / 100.0).sin() * 6.0)
            .collect();

        let mut scaled = logits.clone();
        crate::simd::softmax(&mut scaled);
        let mut sorted: Vec<(Token, f32)> = scaled
            .iter()
            .enumerate()
            .map(|(i, &p)| (i as Token, p))
            .collect();
        sorted.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        // top_k = 1 is the head of that ordering, whatever the seed.
        let tensor = Tensor::vector(logits);
        let p = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            max_tokens: 1,
            seed: None,
        };
        for seed in 0..50 {
            let mut rng = Rng::new(seed);
            assert_eq!(sample(&tensor, &p, &mut rng).unwrap(), sorted[0].0);
        }

        // With a wider k, every token the sampler can reach must be inside the true
        // top-k — the partition must not leak a lower-ranked token into the set.
        let k = 40;
        let allowed: std::collections::HashSet<Token> =
            sorted[..k].iter().map(|&(t, _)| t).collect();
        let p = SamplingParams { top_k: k, ..p };
        for seed in 0..500 {
            let mut rng = Rng::new(seed);
            let t = sample(&tensor, &p, &mut rng).unwrap();
            assert!(
                allowed.contains(&t),
                "sampled {t}, outside the true top-{k}"
            );
        }
    }

    #[test]
    fn sampling_never_returns_a_token_outside_the_vocabulary() {
        let logits = Tensor::vector((0..260).map(|i| (i as f32 * 0.01).sin()).collect());
        let p = SamplingParams {
            temperature: 1.5,
            top_p: 0.9,
            top_k: 40,
            max_tokens: 1,
            seed: None,
        };
        for seed in 0..200 {
            let mut rng = Rng::new(seed);
            let t = sample(&logits, &p, &mut rng).unwrap();
            assert!((t as usize) < 260, "sampled {t}");
        }
    }

    #[test]
    fn same_seed_reproduces_the_same_sequence() {
        let (rt, dir) = runtime("seeded");
        let prompt = rt.tokenizer.encode("hello");
        let p = SamplingParams {
            temperature: 0.9,
            top_p: 0.95,
            top_k: 40,
            max_tokens: 12,
            seed: Some(1234),
        };

        let mut a = rt.start(&prompt, &p).unwrap();
        let mut b = rt.start(&prompt, &p).unwrap();
        assert_eq!(drain(&rt, &mut a, &p).0, drain(&rt, &mut b, &p).0);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn generation_stops_at_max_tokens_and_reports_length() {
        let (rt, dir) = runtime("maxtok");
        let prompt = rt.tokenizer.encode("hi");
        let p = greedy(5);

        let mut s = rt.start(&prompt, &p).unwrap();
        let (tokens, reason) = drain(&rt, &mut s, &p);

        // Greedy decoding could legitimately hit EOS first; if it did not, the run
        // must be capped at exactly max_tokens.
        if reason == StopReason::Length {
            assert_eq!(tokens.len(), 5);
        }
        assert!(tokens.len() <= 5, "generated past max_tokens");
        assert_eq!(
            s.generated(),
            tokens.len() + usize::from(reason == StopReason::Eos)
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn output_length_is_independent_of_prompt_length() {
        // The old scheduler emitted exactly one token per prompt token. It was
        // echoing the input, not generating.
        let (rt, dir) = runtime("indep");
        let p = greedy(6);

        let short = rt.tokenizer.encode("a");
        let long = rt
            .tokenizer
            .encode("a much, much longer prompt than the other one");

        let mut a = rt.start(&short, &p).unwrap();
        let mut b = rt.start(&long, &p).unwrap();
        let (ta, _) = drain(&rt, &mut a, &p);
        let (tb, _) = drain(&rt, &mut b, &p);

        assert!(ta.len() <= 6 && tb.len() <= 6);
        assert_ne!(
            tb.len(),
            long.len(),
            "output length tracked the prompt length"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn generated_tokens_are_not_the_prompt_shifted_by_one() {
        let (rt, dir) = runtime("noecho");
        let p = greedy(8);
        let prompt = rt.tokenizer.encode("Explain Mixture of Experts.");

        let mut s = rt.start(&prompt, &p).unwrap();
        let (out, _) = drain(&rt, &mut s, &p);

        let echo: Vec<Token> = prompt.iter().map(|t| t + 1).collect();
        assert_ne!(
            out,
            echo[..out.len().min(echo.len())],
            "still echoing the prompt"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn empty_prompt_is_rejected() {
        let (rt, dir) = runtime("emptyprompt");
        assert!(rt.start(&[], &greedy(4)).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn oversized_prompt_is_rejected_before_any_work() {
        let (rt, dir) = runtime("toolong");
        let prompt: Vec<Token> = (0..rt.max_context() + 1)
            .map(|i| 4 + (i % 200) as Token)
            .collect();
        let err = rt.start(&prompt, &greedy(4)).unwrap_err();
        assert!(matches!(err, GarudaError::Inference(_)));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn invalid_sampling_params_are_rejected() {
        let (rt, dir) = runtime("badparams");
        let prompt = rt.tokenizer.encode("x");

        for bad in [
            SamplingParams {
                temperature: -1.0,
                ..Default::default()
            },
            SamplingParams {
                top_p: 0.0,
                ..Default::default()
            },
            SamplingParams {
                top_p: 1.5,
                ..Default::default()
            },
            SamplingParams {
                max_tokens: 0,
                ..Default::default()
            },
        ] {
            assert!(rt.start(&prompt, &bad).is_err(), "accepted {bad:?}");
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Absorbing a prompt a few tokens at a time must build exactly the sequence
    /// absorbing it in one go builds. The scheduler relies on this to interleave a
    /// long prefill with other requests' decoding.
    #[test]
    fn a_chunked_prefill_produces_the_same_session_as_a_single_pass() {
        // Separate runtimes, so each prefills cold: sharing one would let the first
        // run populate the prompt cache and the second hit it, testing nothing. The
        // weights are deterministic and the seed is pinned, so the two agree.
        let (rt, dir) = runtime("chunked_whole");
        let (rt2, dir2) = runtime("chunked_pieces");
        let p = greedy(6);
        let prompt = rt
            .tokenizer
            .encode("a prompt long enough to need several prefill chunks to absorb");

        let mut whole = rt.start(&prompt, &p).unwrap();

        let mut pending = rt2.start_incremental(&prompt, &p).unwrap();
        let mut rounds = 0;
        while !rt2.advance_prefill(&mut pending, 3).unwrap() {
            rounds += 1;
            assert!(rounds < prompt.len(), "prefill made no progress");
        }
        assert!(rounds > 1, "fixture too short to exercise chunking");
        let mut piecewise = rt2.finish_prefill(pending);

        assert_eq!(piecewise.prompt_len(), whole.prompt_len());
        assert_eq!(
            drain(&rt, &mut whole, &p).0,
            drain(&rt2, &mut piecewise, &p).0
        );

        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_dir_all(dir2);
    }

    #[test]
    fn a_partly_absorbed_prompt_reports_what_is_left() {
        let (rt, dir) = runtime("remaining");
        let prompt = rt.tokenizer.encode("count me down");
        let prefix = prompt.len() - 1;

        let mut pending = rt.start_incremental(&prompt, &greedy(2)).unwrap();
        assert_eq!(pending.remaining(), prefix);
        rt.advance_prefill(&mut pending, 2).unwrap();
        assert_eq!(pending.remaining(), prefix - 2);
        while !rt.advance_prefill(&mut pending, 4).unwrap() {}
        assert_eq!(pending.remaining(), 0);

        let _ = std::fs::remove_dir_all(dir);
    }

    /// A backend whose next token is a fixed function of the last, so the sequence
    /// it produces is perfectly predictable and eventually repeats.
    ///
    /// Untrained weights cannot demonstrate this: the drafter proposes what came
    /// after an earlier occurrence of the context, and a model with random weights
    /// has no reason to agree, so nothing is ever accepted and an equivalence test
    /// over it compares plain decoding against plain decoding. This one cycles with
    /// a period short enough that the n-gram match fires.
    struct CyclicBackend {
        dims: ModelDims,
        /// When set, the next token depends on the *position* through a hash rather
        /// than on the last token, so the sequence never settles into a pattern the
        /// n-gram lookup can find. Any deterministic function of the last token
        /// cycles, and a cycle is exactly what the drafter predicts perfectly — so
        /// this is the only way to build a fixture whose guesses keep missing.
        chaotic: bool,
    }

    impl CyclicBackend {
        /// Eight ids well clear of the tokenizer's special range, so nothing here can
        /// be mistaken for end-of-sequence.
        fn next(&self, last: Token, prefix_len: usize) -> Token {
            if !self.chaotic {
                return 8 + (last.wrapping_add(1) % 8);
            }
            let mut z = (prefix_len as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            8 + ((z ^ (z >> 31)) % 8) as Token
        }
    }

    impl InferenceBackend for CyclicBackend {
        fn dims(&self) -> ModelDims {
            self.dims
        }

        fn hidden(&self, context: &[Token], seq: &mut SeqState) -> Result<Tensor, GarudaError> {
            let _ = self.logits_multi(context, seq, 1)?;
            Ok(Tensor::zeros(vec![self.dims.d_model]))
        }

        fn logits(&self, context: &[Token], seq: &mut SeqState) -> Result<Tensor, GarudaError> {
            Ok(self
                .logits_multi(context, seq, 1)?
                .pop()
                .expect("one tensor"))
        }

        fn speculation_supported(&self) -> bool {
            true
        }

        fn logits_multi(
            &self,
            context: &[Token],
            seq: &mut SeqState,
            n: usize,
        ) -> Result<Vec<Tensor>, GarudaError> {
            let already = seq.len();
            if already > context.len() {
                return Err(GarudaError::Inference("sequence is ahead".into()));
            }
            let new = context.len() - already;
            if new == 0 {
                return Err(GarudaError::Inference("no new tokens".into()));
            }
            if n > new {
                return Err(GarudaError::Inference(format!(
                    "asked for {n} of {new} new positions"
                )));
            }
            // One cache position per new token, as the contract requires.
            let zero = vec![0.0; self.dims.d_model];
            for _ in 0..new {
                seq.kv().append(&zero, &zero)?;
            }
            Ok((0..n)
                .map(|k| {
                    let at = context.len() - n + k;
                    let mut v = vec![0.0; self.dims.vocab_size];
                    v[self.next(context[at], at + 1) as usize] = 1.0;
                    Tensor::vector(v)
                })
                .collect())
        }
    }

    fn cyclic_runtime() -> InferenceRuntime {
        stub_runtime(false)
    }

    fn stub_runtime(chaotic: bool) -> InferenceRuntime {
        let dims = ModelDims::default();
        InferenceRuntime::new(
            Arc::new(crate::tokenizer::Tokenizer::new()),
            Arc::new(CyclicBackend { dims, chaotic }),
            KvConfig::mha(dims, 256, 64, None, None),
            8,
            64 << 20,
        )
    }

    /// The whole justification for speculation: it must produce exactly what plain
    /// greedy decoding produces. A guess is only ever accepted where the model would
    /// have chosen it anyway, so any difference is a bug, not a tradeoff.
    #[test]
    fn speculative_decoding_produces_exactly_what_greedy_decoding_does() {
        let rt = cyclic_runtime();
        let rt2 = cyclic_runtime();
        let p = greedy(40);
        let prompt: Vec<Token> = vec![8, 9, 10, 11, 12];

        let mut plain = rt2.start(&prompt, &p).unwrap();
        let (want, want_reason) = drain(&rt2, &mut plain, &p);

        let mut spec = rt.start(&prompt, &p).unwrap();
        let mut got = Vec::new();
        let reason;
        let mut passes = 0;
        let mut accepted_extra = 0;
        loop {
            let mut batch = Vec::new();
            let outcome = rt.next_tokens_speculative(&mut spec, &p, 4, &mut batch);
            passes += 1;
            accepted_extra += batch.len().saturating_sub(1);
            got.extend(batch);
            if let Err(r) = outcome {
                reason = r;
                break;
            }
        }

        // Guards the test itself: if nothing was ever guessed and accepted, this
        // would be comparing plain decoding against plain decoding.
        assert!(
            accepted_extra > 0,
            "speculation never accepted a token, so this proves nothing"
        );
        assert!(
            passes < got.len(),
            "{} passes for {} tokens — no pass produced more than one",
            passes,
            got.len()
        );

        assert_eq!(got, want, "speculation changed the output");
        assert_eq!(reason, want_reason);
        assert_eq!(
            spec.generated(),
            plain.generated(),
            "generated count differs"
        );
        assert_eq!(spec.seq.len(), plain.seq.len(), "the caches diverged");
    }

    /// The property the whole sampled path rests on: over many steps, keeping a
    /// guess with probability `p(guess)` and otherwise drawing from the remainder
    /// emits exactly `p`. Get this wrong and a caller who asked for `temperature =
    /// 0.8` quietly receives something else — a bias no single request could reveal.
    #[test]
    fn verifying_a_guess_reproduces_the_callers_distribution() {
        let dist = [(10u32, 0.5f32), (11, 0.3), (12, 0.2)];

        for drafted in [11u32, 10, 12, 99] {
            let mut rng = Rng::new(0xD15 + drafted as u64);
            let mut seen = [0u32; 3];
            const N: u32 = 60_000;
            for _ in 0..N {
                let (_, tok) = verify_drafted(&dist, drafted, &mut rng);
                seen[(tok - 10) as usize] += 1;
            }
            for (i, (tok, want)) in dist.iter().enumerate() {
                let got = seen[i] as f32 / N as f32;
                assert!(
                    (got - want).abs() < 0.01,
                    "drafting {drafted}: token {tok} came out at {got:.4}, \
                     but the caller's distribution says {want}"
                );
            }
        }
    }

    /// A token the caller's own truncation removed has no mass, so a guess of it can
    /// never be kept — and the emitted token must still follow `p`.
    #[test]
    fn a_guess_outside_the_candidate_set_is_always_rejected() {
        let dist = [(10u32, 0.6f32), (11, 0.4)];
        let mut rng = Rng::new(7);
        for _ in 0..1000 {
            let (kept, tok) = verify_drafted(&dist, 99, &mut rng);
            assert!(!kept, "kept a guess with no probability mass");
            assert!(
                tok == 10 || tok == 11,
                "emitted {tok}, outside the candidates"
            );
        }
    }

    /// Sampled requests speculate now. They did not before — the equality test that
    /// works for greedy would have handed them the greedy answer.
    #[test]
    fn a_sampled_request_can_win_more_than_one_token_from_a_pass() {
        let rt = cyclic_runtime();
        // Low temperature so the model's own choice dominates and guesses drawn from
        // the repeating context are usually kept.
        let p = SamplingParams {
            temperature: 0.05,
            top_p: 1.0,
            top_k: 0,
            max_tokens: 80,
            seed: Some(3),
        };
        let prompt: Vec<Token> = vec![8, 9, 10, 11, 12];

        let mut s = rt.start(&prompt, &p).unwrap();
        let mut multi = 0;
        for _ in 0..30 {
            let mut batch = Vec::new();
            let done = rt
                .next_tokens_speculative(&mut s, &p, 6, &mut batch)
                .is_err();
            if batch.len() > 1 {
                multi += 1;
            }
            if done {
                break;
            }
        }
        assert!(
            multi > 0,
            "a sampled request never won more than one token per pass"
        );
    }

    #[test]
    fn the_drafter_only_guesses_when_the_context_repeats() {
        // No repetition, no basis for a guess.
        assert!(draft_from_context(&[1, 2, 3, 4, 5, 6, 7, 8], 4, 4).is_empty());
        // Too short to hold an n-gram plus a match.
        assert!(draft_from_context(&[1, 2, 3], 4, 4).is_empty());
        assert!(draft_from_context(&[1, 2, 3, 4, 5], 0, 4).is_empty());

        // "1 2 3 4" occurred before and was followed by 9, 9, 9.
        let ctx = [1, 2, 3, 4, 9, 9, 9, 7, 1, 2, 3, 4];
        assert_eq!(draft_from_context(&ctx, 3, 4), vec![9, 9, 9]);
        // Bounded by what was asked for.
        assert_eq!(draft_from_context(&ctx, 2, 4), vec![9, 9]);

        // The most recent match wins: the later occurrence was followed by 5.
        let ctx = [7, 7, 7, 7, 9, 1, 2, 7, 7, 7, 7, 5, 1, 2, 7, 7, 7, 7];
        assert_eq!(draft_from_context(&ctx, 1, 4), vec![5]);
    }

    #[test]
    fn a_repeated_prompt_hits_the_prefix_cache() {
        let (rt, dir) = runtime("prefix");
        let p = greedy(3);
        let prompt = rt.tokenizer.encode("the same prompt twice");

        let _ = rt.start(&prompt, &p).unwrap();
        assert_eq!(rt.prompt_cache_stats().hits, 0, "first run cannot hit");

        let _ = rt.start(&prompt, &p).unwrap();
        assert_eq!(
            rt.prompt_cache_stats().hits,
            1,
            "second run should reuse the prefill"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_cache_hit_produces_the_same_tokens_as_a_cold_run() {
        let (rt, dir) = runtime("prefix_correct");
        let p = greedy(6);
        let prompt = rt.tokenizer.encode("consistency check");

        let mut cold = rt.start(&prompt, &p).unwrap();
        let (a, ra) = drain(&rt, &mut cold, &p);

        let mut warm = rt.start(&prompt, &p).unwrap();
        let (b, rb) = drain(&rt, &mut warm, &p);

        assert_eq!(a, b, "the prefix cache changed the output");
        assert_eq!(ra, rb);

        let _ = std::fs::remove_dir_all(dir);
    }
}

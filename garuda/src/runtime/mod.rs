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

    // Temperature, then softmax to probabilities.
    let mut scaled: Vec<f32> = data.iter().map(|&v| v / params.temperature).collect();
    crate::simd::softmax(&mut scaled);

    let mut candidates: Vec<(Token, f32)> = scaled
        .iter()
        .enumerate()
        .map(|(i, &p)| (i as Token, p))
        .collect();
    candidates.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    if params.top_k > 0 && params.top_k < candidates.len() {
        candidates.truncate(params.top_k);
    }

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

/// One in-flight sequence.
#[derive(Debug)]
pub struct Session {
    seq: SeqState,
    /// Prompt followed by everything generated so far.
    context: Vec<Token>,
    prompt_len: usize,
    rng: Rng,
    finished: bool,
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
    ) -> Self {
        let max_context = kv_template.max_positions;
        Self {
            tokenizer,
            backend,
            prompt_cache: PromptCache::new(prompt_cache_capacity),
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
    pub fn start(&self, prompt: &[Token], params: &SamplingParams) -> Result<Session, GarudaError> {
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

        let seq = if prefix.is_empty() {
            SeqState::new(self.kv_template.clone(), seq_id)
        } else {
            // A `match` rather than an `else if let` chain: under the 2024 edition the
            // latter would drop the `get` temporary at a different point, and the
            // explicit form keeps the prompt-cache lookup scope unambiguous.
            match self.prompt_cache.get(prefix, seq_id) {
                Some(cached) => cached,
                None => {
                    let mut fresh = SeqState::new(self.kv_template.clone(), seq_id);
                    // Prefill. The logits from the prefix are not needed; the KV state is.
                    self.backend.logits(prefix, &mut fresh)?;
                    if !fresh.has_spill() {
                        self.prompt_cache.insert(prefix, fresh.clone());
                    }
                    fresh
                }
            }
        };

        let seed = params.seed.unwrap_or(seq_id);
        Ok(Session {
            seq,
            context: prompt.to_vec(),
            prompt_len: prompt.len(),
            rng: Rng::new(seed),
            finished: false,
        })
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
        (InferenceRuntime::new(tk, engine, kv, 8), dir)
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

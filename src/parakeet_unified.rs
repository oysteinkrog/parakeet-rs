use crate::audio::{self, load_audio};
use crate::config::PreprocessorConfig;
use crate::decoder::{TimedToken, TranscriptionResult};
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_unified::{ParakeetUnifiedModel, UnifiedModelConfig};
use crate::nemotron::SentencePieceVocab;
use crate::timestamps::{process_timestamps, TimestampMode};
use crate::transcriber::Transcriber;
use ndarray::Array3;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const SAMPLE_RATE: usize = 16000;
const FEATURE_SIZE: usize = 128;
const HOP_LENGTH: usize = 160;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const PREEMPHASIS: f32 = 0.97;
const DECODER_LSTM_DIM: usize = 640;
const DECODER_LSTM_LAYERS: usize = 2;
const SUBSAMPLING_FACTOR: usize = 8;
const MAX_SYMBOLS_PER_STEP: usize = 10;
const TSALLIS_ALPHA: f64 = 0.33;

/// Number of most likely tokens kept per decoder call in [`TokenDetail::top`].
pub const TOKEN_DETAIL_TOP_K: usize = 5;
/// Number of non-blank candidates kept per blank frame in [`FrameSignal::top`].
pub const FRAME_SIGNAL_TOP_K: usize = 3;

/// A summary of the joint network's distribution at one decoder call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenDetail {
    /// The most likely tokens, most likely first, as (token id, natural log
    /// probability). The blank id can appear here.
    pub top: [(usize, f32); TOKEN_DETAIL_TOP_K],
    /// Natural log probability of blank.
    pub logp_blank: f32,
    /// Shannon entropy in nats.
    pub entropy: f32,
    /// NeMo's `entropy_tsallis_exp` confidence with alpha 0.33: 1 for a
    /// one-hot distribution, 0 for a uniform one.
    pub tsallis_confidence: f32,
    /// Probability of the most likely token minus that of the second.
    pub margin: f32,
}

/// The decoder call that ended one encoder frame by emitting blank.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameSignal {
    /// Encoder frame from the start of the stream (80 ms per frame).
    pub abs_frame: usize,
    /// log10(1 - P(blank)) at that call: how close the frame came to a token.
    pub log10_nonblank: f32,
    /// The most likely non-blank tokens, most likely first, as (token id,
    /// natural log probability in the full distribution).
    pub top: [(usize, f32); FRAME_SIGNAL_TOP_K],
}

/// A token decoded from the right-context frames of the last step. It is
/// never committed and is replaced by the next step.
#[derive(Debug, Clone, PartialEq)]
pub struct TentativeToken {
    pub id: usize,
    pub text: String,
    /// Encoder frame from the start of the stream.
    pub abs_frame: usize,
    /// Encoder frames between this token's frame and the right edge of the
    /// encoded window; 0 for the last frame.
    pub frames_to_edge: usize,
    /// The model's probability for this token, like `token_confidences`.
    pub confidence: f32,
    pub detail: TokenDetail,
}

/// Time and decoder calls spent on the last streaming step.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StepStats {
    /// Feature extraction and the encoder run.
    pub encoder: Duration,
    /// Greedy decode of the chunk frames, signals included.
    pub decode: Duration,
    /// Part of `decode` and `tentative` spent computing the signals.
    pub signals: Duration,
    /// Greedy decode of the right-context frames, signals included.
    pub tentative: Duration,
    pub joint_calls: usize,
    pub tentative_joint_calls: usize,
}

/// Everything one call to `greedy_decode` produced.
#[derive(Default)]
struct Decoded {
    /// (token id, absolute encoder frame) of each emitted token.
    tokens: Vec<(usize, usize)>,
    scores: Vec<f32>,
    details: Vec<TokenDetail>,
    frames: Vec<FrameSignal>,
    joint_calls: usize,
    signal_time: Duration,
}

/// One pass over the logits: log-softmax summaries for `TokenDetail` and
/// `FrameSignal`. Sums run in f64 so that 1 - P(blank) keeps its precision
/// when blank is nearly certain.
fn summarize_logits(logits: &[f32], blank_id: usize) -> (TokenDetail, FrameSignal) {
    fn insert<const K: usize>(top: &mut [(usize, f64); K], id: usize, value: f64) {
        if value <= top[K - 1].1 {
            return;
        }
        let mut i = K - 1;
        while i > 0 && value > top[i - 1].1 {
            top[i] = top[i - 1];
            i -= 1;
        }
        top[i] = (id, value);
    }

    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut sum = 0.0f64;
    let mut nonblank_sum = 0.0f64;
    for (i, &l) in logits.iter().enumerate() {
        let e = (l as f64 - max).exp();
        sum += e;
        if i != blank_id {
            nonblank_sum += e;
        }
    }
    let log_sum = sum.ln();

    let mut top = [(0usize, f64::NEG_INFINITY); TOKEN_DETAIL_TOP_K];
    let mut top_nonblank = [(0usize, f64::NEG_INFINITY); FRAME_SIGNAL_TOP_K];
    let mut entropy = 0.0f64;
    let mut alpha_sum = 0.0f64;
    for (i, &l) in logits.iter().enumerate() {
        let logp = l as f64 - max - log_sum;
        let p = logp.exp();
        if p > 0.0 {
            entropy -= p * logp;
        }
        alpha_sum += (TSALLIS_ALPHA * logp).exp();
        insert(&mut top, i, logp);
        if i != blank_id {
            insert(&mut top_nonblank, i, logp);
        }
    }

    // NeMo asr_confidence_utils: entropy_tsallis_exp(x, v, t)
    let v = logits.len() as f64;
    let exp_neg_max_ent = ((1.0 - v.powf(1.0 - TSALLIS_ALPHA)) / (1.0 - TSALLIS_ALPHA)).exp();
    let tsallis_confidence = (((1.0 - alpha_sum) / (1.0 - TSALLIS_ALPHA)).exp() - exp_neg_max_ent)
        / (1.0 - exp_neg_max_ent);

    let logp_blank = logits
        .get(blank_id)
        .map_or(f64::NEG_INFINITY, |&l| l as f64 - max - log_sum);
    let detail = TokenDetail {
        top: top.map(|(id, lp)| (id, lp as f32)),
        logp_blank: logp_blank as f32,
        entropy: entropy as f32,
        tsallis_confidence: tsallis_confidence as f32,
        margin: (top[0].1.exp() - top[1].1.exp()) as f32,
    };
    let frame = FrameSignal {
        abs_frame: 0,
        log10_nonblank: (nonblank_sum / sum).log10() as f32,
        top: top_nonblank.map(|(id, lp)| (id, lp as f32)),
    };
    (detail, frame)
}

/// Greedy RNN-T decode of `start_frame..end_frame`, advancing the given
/// decoder state. The token choice is the argmax of the joint logits; the
/// signals are read off the same logits and never change it.
#[allow(clippy::too_many_arguments)]
fn greedy_decode(
    model: &mut ParakeetUnifiedModel,
    blank_id: usize,
    encoder_out: &Array3<f32>,
    start_frame: usize,
    end_frame: usize,
    absolute_frame_offset: usize,
    state_1: &mut Array3<f32>,
    state_2: &mut Array3<f32>,
    last_token: &mut i32,
) -> Result<Decoded> {
    let mut out = Decoded::default();
    let hidden_dim = encoder_out.shape()[1];
    let end_frame = end_frame.min(encoder_out.shape()[2]);

    for frame_idx in start_frame..end_frame {
        let frame = encoder_out
            .slice(ndarray::s![0, .., frame_idx])
            .to_owned()
            .to_shape((1, hidden_dim, 1))
            .map_err(|e| Error::Model(format!("Failed to reshape encoder frame: {e}")))?
            .to_owned();

        let absolute_frame = absolute_frame_offset + (frame_idx - start_frame);

        for _ in 0..MAX_SYMBOLS_PER_STEP {
            let (logits, new_state_1, new_state_2) =
                model.run_decoder(&frame, *last_token, state_1, state_2)?;
            out.joint_calls += 1;

            let (token_id, max_logit) = logits
                .iter()
                .copied()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((blank_id, 0.0));

            let signal_start = Instant::now();
            let (detail, mut frame_signal) = match logits.as_slice() {
                Some(slice) => summarize_logits(slice, blank_id),
                None => summarize_logits(&logits.to_vec(), blank_id),
            };
            out.signal_time += signal_start.elapsed();

            if token_id == blank_id {
                frame_signal.abs_frame = absolute_frame;
                out.frames.push(frame_signal);
                break;
            }

            // Softmax probability of the emitted token.
            let sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
            out.scores.push(1.0 / sum);
            out.details.push(detail);
            out.tokens.push((token_id, absolute_frame));
            *last_token = token_id as i32;
            *state_1 = new_state_1;
            *state_2 = new_state_2;
        }
    }

    Ok(out)
}

#[derive(Debug, Clone, Copy)]
pub struct UnifiedStreamingConfig {
    pub left_context_secs: f32,
    pub chunk_secs: f32,
    pub right_context_secs: f32,
}

impl Default for UnifiedStreamingConfig {
    fn default() -> Self {
        Self {
            left_context_secs: 5.6,
            chunk_secs: 0.56,
            right_context_secs: 0.56,
        }
    }
}

impl UnifiedStreamingConfig {
    fn frames_from_secs(secs: f32) -> usize {
        ((secs * SAMPLE_RATE as f32) / HOP_LENGTH as f32).round() as usize
    }

    pub fn validate(self) -> Result<Self> {
        let left_frames = self.left_context_frames();
        let chunk_frames = self.chunk_frames();
        let right_frames = self.right_context_frames();

        if chunk_frames == 0 {
            return Err(Error::Config(
                "Unified streaming chunk size must be greater than zero".to_string(),
            ));
        }

        for (name, frames) in [
            ("left_context_secs", left_frames),
            ("chunk_secs", chunk_frames),
            ("right_context_secs", right_frames),
        ] {
            if frames % SUBSAMPLING_FACTOR != 0 {
                return Err(Error::Config(format!(
                    "{name} must map to a mel-frame count divisible by {SUBSAMPLING_FACTOR}"
                )));
            }
        }

        Ok(self)
    }

    pub fn left_context_frames(self) -> usize {
        Self::frames_from_secs(self.left_context_secs)
    }

    pub fn chunk_frames(self) -> usize {
        Self::frames_from_secs(self.chunk_secs)
    }

    pub fn right_context_frames(self) -> usize {
        Self::frames_from_secs(self.right_context_secs)
    }

    pub fn total_window_frames(self) -> usize {
        self.left_context_frames() + self.chunk_frames() + self.right_context_frames()
    }

    pub fn left_context_samples(self) -> usize {
        self.left_context_frames() * HOP_LENGTH
    }

    pub fn chunk_samples(self) -> usize {
        self.chunk_frames() * HOP_LENGTH
    }

    pub fn right_context_samples(self) -> usize {
        self.right_context_frames() * HOP_LENGTH
    }

    pub fn total_window_samples(self) -> usize {
        self.total_window_frames() * HOP_LENGTH
    }

    pub fn chunk_encoder_frames(self) -> usize {
        self.chunk_frames() / SUBSAMPLING_FACTOR
    }

    pub fn left_context_encoder_frames(self) -> usize {
        self.left_context_frames() / SUBSAMPLING_FACTOR
    }
}

/// Shared handle to a loaded ParakeetUnified model.
/// The ONNX session is loaded once and reference-counted.
///
/// Use [`ParakeetUnifiedHandle::load`] to load from disk, then
/// [`ParakeetUnified::from_shared`] to spawn each stream with its own state.
#[derive(Clone)]
pub struct ParakeetUnifiedHandle {
    model: Arc<Mutex<ParakeetUnifiedModel>>,
    vocab: Arc<SentencePieceVocab>,
    preprocessor_config: Arc<PreprocessorConfig>,
    blank_id: usize,
}

pub struct ParakeetUnified {
    model: Arc<Mutex<ParakeetUnifiedModel>>,
    vocab: Arc<SentencePieceVocab>,
    preprocessor_config: Arc<PreprocessorConfig>,
    state_1: Array3<f32>,
    state_2: Array3<f32>,
    last_token: i32,
    blank_id: usize,
    streaming_config: UnifiedStreamingConfig,
    audio_buffer: Vec<f32>,
    buffer_start_sample: usize,
    next_chunk_start_sample: usize,
    accumulated_tokens: Vec<usize>,
    accumulated_timed_tokens: Vec<TimedToken>,
    /// Probability of each token in `accumulated_timed_tokens`: the softmax
    /// of the joint logits at the emitted token.
    accumulated_token_scores: Vec<f32>,
    /// Scores of the tokens from the last `decode_encoder_frames` call.
    pending_scores: Vec<f32>,
    /// Distribution summary of each token in `accumulated_timed_tokens`.
    accumulated_token_details: Vec<TokenDetail>,
    /// Details of the tokens from the last `decode_encoder_frames` call.
    pending_details: Vec<TokenDetail>,
    /// One entry per committed encoder frame that ended in blank.
    frame_signals: Vec<FrameSignal>,
    tentative_enabled: bool,
    tentative: Vec<TentativeToken>,
    last_step_stats: StepStats,
    /// Time spent on the last `decode_tail`: `encoder` and `tentative`.
    last_tail_stats: StepStats,
}

impl ParakeetUnifiedHandle {
    /// Load the ParakeetUnified model, vocabulary, and preprocessor config
    /// from a directory.
    pub fn load<P: AsRef<Path>>(
        path: P,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let vocab = SentencePieceVocab::from_file(path.join("tokenizer.model"))?;
        let blank_id = vocab.size();

        let model_config = UnifiedModelConfig {
            vocab_size: vocab.size() + 1,
            blank_id,
            decoder_lstm_dim: DECODER_LSTM_DIM,
            decoder_lstm_layers: DECODER_LSTM_LAYERS,
            subsampling_factor: SUBSAMPLING_FACTOR,
        };

        let model = ParakeetUnifiedModel::from_pretrained(
            path,
            exec_config.unwrap_or_default(),
            model_config,
        )?;

        let preprocessor_config = PreprocessorConfig {
            feature_extractor_type: "ParakeetFeatureExtractor".to_string(),
            feature_size: FEATURE_SIZE,
            hop_length: HOP_LENGTH,
            n_fft: N_FFT,
            padding_side: "right".to_string(),
            padding_value: 0.0,
            preemphasis: PREEMPHASIS,
            processor_class: "ParakeetProcessor".to_string(),
            return_attention_mask: true,
            sampling_rate: SAMPLE_RATE,
            win_length: WIN_LENGTH,
        };

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            vocab: Arc::new(vocab),
            preprocessor_config: Arc::new(preprocessor_config),
            blank_id,
        })
    }
}

impl ParakeetUnified {
    pub fn from_pretrained<P: AsRef<Path>>(
        path: P,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        Self::from_pretrained_with_streaming_config(
            path,
            exec_config,
            UnifiedStreamingConfig::default(),
        )
    }

    pub fn from_pretrained_with_streaming_config<P: AsRef<Path>>(
        path: P,
        exec_config: Option<ExecutionConfig>,
        streaming_config: UnifiedStreamingConfig,
    ) -> Result<Self> {
        let handle = ParakeetUnifiedHandle::load(path, exec_config)?;
        Self::from_shared_with_streaming_config(&handle, streaming_config)
    }

    /// Spawn a new ParakeetUnified instance bound to a shared model, using the
    /// default streaming profile.
    pub fn from_shared(handle: &ParakeetUnifiedHandle) -> Self {
        // default config is pre-validated, so unwrap is safe
        Self::from_shared_with_streaming_config(handle, UnifiedStreamingConfig::default())
            .expect("default UnifiedStreamingConfig is always valid")
    }

    /// Spawn a new ParakeetUnified instance bound to a shared model with a
    /// custom streaming profile. Each instance owns independent decoder and
    /// audio-buffer state; the ONNX session is shared through the handle.
    pub fn from_shared_with_streaming_config(
        handle: &ParakeetUnifiedHandle,
        streaming_config: UnifiedStreamingConfig,
    ) -> Result<Self> {
        let streaming_config = streaming_config.validate()?;
        let blank_id = handle.blank_id;

        Ok(Self {
            model: Arc::clone(&handle.model),
            vocab: Arc::clone(&handle.vocab),
            preprocessor_config: Arc::clone(&handle.preprocessor_config),
            state_1: Array3::zeros((DECODER_LSTM_LAYERS, 1, DECODER_LSTM_DIM)),
            state_2: Array3::zeros((DECODER_LSTM_LAYERS, 1, DECODER_LSTM_DIM)),
            last_token: blank_id as i32,
            blank_id,
            streaming_config,
            audio_buffer: Vec::new(),
            buffer_start_sample: 0,
            next_chunk_start_sample: 0,
            accumulated_tokens: Vec::new(),
            accumulated_timed_tokens: Vec::new(),
            accumulated_token_scores: Vec::new(),
            pending_scores: Vec::new(),
            accumulated_token_details: Vec::new(),
            pending_details: Vec::new(),
            frame_signals: Vec::new(),
            tentative_enabled: true,
            tentative: Vec::new(),
            last_tail_stats: StepStats::default(),
            last_step_stats: StepStats::default(),
        })
    }

    pub fn streaming_config(&self) -> UnifiedStreamingConfig {
        self.streaming_config
    }

    pub fn preprocessor_config(&self) -> &PreprocessorConfig {
        &self.preprocessor_config
    }

    pub fn reset(&mut self) {
        self.state_1.fill(0.0);
        self.state_2.fill(0.0);
        self.last_token = self.blank_id as i32;
        self.audio_buffer.clear();
        self.buffer_start_sample = 0;
        self.next_chunk_start_sample = 0;
        self.accumulated_tokens.clear();
        self.accumulated_timed_tokens.clear();
        self.accumulated_token_scores.clear();
        self.accumulated_token_details.clear();
        self.frame_signals.clear();
        self.tentative.clear();
        self.last_step_stats = StepStats::default();
    }

    pub fn get_timed_transcript(&self, mode: TimestampMode) -> TranscriptionResult {
        let text = self.get_transcript();
        let tokens = process_timestamps(&self.accumulated_timed_tokens, mode);
        TranscriptionResult { text, tokens }
    }

    /// Confidence of each token returned by `get_timed_transcript` in
    /// `TimestampMode::Tokens`, in the same order: the model's probability
    /// for the emitted token, 0 to 1.
    pub fn token_confidences(&self) -> &[f32] {
        &self.accumulated_token_scores
    }

    /// Distribution summary of each token returned by `get_timed_transcript`
    /// in `TimestampMode::Tokens`, in the same order as `token_confidences`.
    pub fn token_details(&self) -> &[TokenDetail] {
        &self.accumulated_token_details
    }

    /// One entry per committed encoder frame, in order, taken from the decoder
    /// call that emitted blank to end the frame. A frame that hit the
    /// per-frame symbol limit has no entry.
    pub fn frame_signals(&self) -> &[FrameSignal] {
        &self.frame_signals
    }

    /// Tokens decoded from the right-context frames of the last streaming
    /// step, from a copy of the decoder state. They are never committed: the
    /// next step replaces them, and the committed tokens are the same whether
    /// or not this runs.
    pub fn tentative(&self) -> &[TentativeToken] {
        &self.tentative
    }

    /// The committed transcript followed by the tentative tokens.
    pub fn transcript_with_tentative(&self) -> String {
        let valid: Vec<usize> = self
            .accumulated_tokens
            .iter()
            .copied()
            .filter(|&token| token < self.blank_id)
            .chain(self.tentative.iter().map(|t| t.id))
            .collect();
        self.vocab.decode(&valid)
    }

    /// Decode the right-context frames each step (on by default). Off saves
    /// the extra decoder calls and leaves `tentative` empty.
    pub fn set_tentative_enabled(&mut self, enabled: bool) {
        self.tentative_enabled = enabled;
        if !enabled {
            self.tentative.clear();
        }
    }

    /// Time and decoder calls of the last streaming step.
    pub fn last_step_stats(&self) -> StepStats {
        self.last_step_stats
    }

    /// Time of the last `decode_tail`: the encoder run and the tentative
    /// decode.
    pub fn last_tail_stats(&self) -> StepStats {
        self.last_tail_stats
    }

    /// Samples of the stream the committed transcript covers: where the
    /// next streaming step starts.
    pub fn committed_samples(&self) -> usize {
        self.next_chunk_start_sample
    }

    /// Decode a new tentative tail from all the audio received after the
    /// committed part, between two streaming steps: one encoder pass over the
    /// window that ends at the newest sample and a greedy decode of every
    /// frame after the committed ones, from a copy of the decoder state.
    /// Replaces `tentative`. The committed state is not touched, so the
    /// committed tokens are the same whether or not this runs. Returns false
    /// when tentative decoding is off or no audio is past the committed part.
    pub fn decode_tail(&mut self) -> Result<bool> {
        let total_received = self.buffer_start_sample + self.audio_buffer.len();
        if !self.tentative_enabled || total_received <= self.next_chunk_start_sample {
            return Ok(false);
        }
        let start = self.next_chunk_start_sample;
        // As a flush builds it: the window ends at the newest sample, with no
        // padding after it.
        let (window_audio, left_encoder_frames, chunk_encoder_frames) =
            self.build_window_audio(start, total_received, true);
        if chunk_encoder_frames == 0 {
            return Ok(false);
        }
        let encoder_start = Instant::now();
        let features = audio::extract_features_raw(
            window_audio,
            SAMPLE_RATE as u32,
            1,
            &self.preprocessor_config,
        )?;
        let (encoded, encoded_len) = {
            let mut model = self
                .model
                .lock()
                .map_err(|e| Error::Model(format!("Failed to acquire model lock: {e}")))?;
            model.run_encoder(&features)?
        };
        let encoder = encoder_start.elapsed();
        let available_frames = (encoded_len as usize).min(encoded.shape()[2]);
        let start_frame = left_encoder_frames.min(available_frames);
        let absolute_frame_offset = start / (HOP_LENGTH * SUBSAMPLING_FACTOR);
        // decode_tentative writes its time into the step stats; keep those
        // for the step and take the tail's own.
        let step = self.last_step_stats;
        self.decode_tentative(&encoded, start_frame, available_frames, absolute_frame_offset)?;
        self.last_tail_stats = StepStats {
            encoder,
            tentative: self.last_step_stats.tentative,
            tentative_joint_calls: self.last_step_stats.tentative_joint_calls,
            ..StepStats::default()
        };
        self.last_step_stats = step;
        Ok(true)
    }

    pub fn get_transcript(&self) -> String {
        let valid: Vec<usize> = self
            .accumulated_tokens
            .iter()
            .copied()
            .filter(|&token| token < self.blank_id)
            .collect();
        self.vocab.decode(&valid)
    }

    pub fn transcribe_audio(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
    ) -> Result<String> {
        self.transcribe_offline(audio, sample_rate, channels, None)
            .map(|result| result.text)
    }

    pub fn transcribe_file<P: AsRef<Path>>(&mut self, audio_path: P) -> Result<String> {
        let (audio, spec) = load_audio(audio_path)?;
        self.transcribe_audio(audio, spec.sample_rate, spec.channels)
    }

    pub fn transcribe_chunk(&mut self, audio_chunk: &[f32]) -> Result<String> {
        self.audio_buffer.extend_from_slice(audio_chunk);
        self.process_ready_chunks(false)
    }

    pub fn flush(&mut self) -> Result<String> {
        self.process_ready_chunks(true)
    }

    fn process_ready_chunks(&mut self, flush: bool) -> Result<String> {
        let mut emitted = String::new();
        let chunk_samples = self.streaming_config.chunk_samples();
        let right_context_samples = self.streaming_config.right_context_samples();

        loop {
            let total_received = self.buffer_start_sample + self.audio_buffer.len();
            let ready = if flush {
                total_received > self.next_chunk_start_sample
            } else {
                total_received
                    >= self.next_chunk_start_sample + chunk_samples + right_context_samples
            };

            if !ready {
                break;
            }

            let (window_audio, left_encoder_frames, chunk_encoder_frames) =
                self.build_window_audio(self.next_chunk_start_sample, total_received, flush);
            if chunk_encoder_frames == 0 {
                break;
            }

            let encoder_start = Instant::now();
            let features = audio::extract_features_raw(
                window_audio,
                SAMPLE_RATE as u32,
                1,
                &self.preprocessor_config,
            )?;
            let (encoded, encoded_len) = {
                let mut model = self.model.lock().map_err(|e| {
                    Error::Model(format!("Failed to acquire model lock: {e}"))
                })?;
                model.run_encoder(&features)?
            };

            let available_frames = (encoded_len as usize).min(encoded.shape()[2]);
            let start_frame = left_encoder_frames.min(available_frames);
            let end_frame = (start_frame + chunk_encoder_frames).min(available_frames);

            let encoder_time = encoder_start.elapsed();

            let absolute_frame_offset =
                self.next_chunk_start_sample / (HOP_LENGTH * SUBSAMPLING_FACTOR);
            let decode_start = Instant::now();
            let tokens =
                self.decode_encoder_frames(&encoded, start_frame, end_frame, absolute_frame_offset)?;
            self.accumulated_tokens
                .extend(tokens.iter().map(|(id, _)| *id));
            self.accumulated_timed_tokens
                .extend(self.tokens_to_timed(&tokens));
            let scores = self.take_scores(&tokens);
            self.accumulated_token_scores.extend(scores);
            let details = self.take_details(&tokens);
            self.accumulated_token_details.extend(details);
            emitted.push_str(&self.decode_incremental_tokens(&tokens));
            self.last_step_stats.encoder = encoder_time;
            self.last_step_stats.decode = decode_start.elapsed();

            let tentative_offset = absolute_frame_offset + (end_frame - start_frame);
            self.decode_tentative(&encoded, end_frame, available_frames, tentative_offset)?;

            self.next_chunk_start_sample += chunk_samples;
            self.trim_audio_buffer();

            if flush && total_received <= self.next_chunk_start_sample {
                break;
            }
        }

        Ok(emitted)
    }

    fn build_window_audio(
        &self,
        chunk_start_sample: usize,
        total_received: usize,
        flush: bool,
    ) -> (Vec<f32>, usize, usize) {
        let left_context_samples = self.streaming_config.left_context_samples();
        let chunk_samples = self.streaming_config.chunk_samples();
        let right_context_samples = self.streaming_config.right_context_samples();

        let available_left = chunk_start_sample.saturating_sub(self.buffer_start_sample);
        let available_left = available_left.min(left_context_samples);
        let available_main = total_received.saturating_sub(chunk_start_sample).min(chunk_samples);
        let available_right = if flush {
            total_received
                .saturating_sub(chunk_start_sample + available_main)
                .min(right_context_samples)
        } else {
            right_context_samples
        };

        let window_start = chunk_start_sample.saturating_sub(available_left);
        let window_end = chunk_start_sample + available_main + available_right;
        let total_window_samples = window_end.saturating_sub(window_start);

        let left_encoder_frames = (available_left / HOP_LENGTH) / SUBSAMPLING_FACTOR;
        let chunk_encoder_frames = (available_main / HOP_LENGTH) / SUBSAMPLING_FACTOR;

        let mut window = vec![0.0f32; total_window_samples];
        let buffer_end = self.buffer_start_sample + self.audio_buffer.len();
        let copy_start = window_start.max(self.buffer_start_sample);
        let copy_end = window_end.min(buffer_end);

        if copy_end > copy_start {
            let src_start = copy_start - self.buffer_start_sample;
            let dst_start = copy_start - window_start;
            let len = copy_end - copy_start;
            window[dst_start..dst_start + len]
                .copy_from_slice(&self.audio_buffer[src_start..src_start + len]);
        }

        (window, left_encoder_frames, chunk_encoder_frames)
    }

    fn trim_audio_buffer(&mut self) {
        let keep_from = self
            .next_chunk_start_sample
            .saturating_sub(self.streaming_config.left_context_samples());
        if keep_from <= self.buffer_start_sample {
            return;
        }

        let drop = keep_from - self.buffer_start_sample;
        if drop == 0 {
            return;
        }

        if drop >= self.audio_buffer.len() {
            self.audio_buffer.clear();
            self.buffer_start_sample = keep_from;
            return;
        }

        self.audio_buffer.drain(0..drop);
        self.buffer_start_sample = keep_from;
    }

    fn decode_encoder_frames(
        &mut self,
        encoder_out: &Array3<f32>,
        start_frame: usize,
        end_frame: usize,
        absolute_frame_offset: usize,
    ) -> Result<Vec<(usize, usize)>> {
        // Hold the lock once across the decoder loop to avoid per-step acquire/release.
        let mut model = self
            .model
            .lock()
            .map_err(|e| Error::Model(format!("Failed to acquire model lock: {e}")))?;

        let decoded = greedy_decode(
            &mut model,
            self.blank_id,
            encoder_out,
            start_frame,
            end_frame,
            absolute_frame_offset,
            &mut self.state_1,
            &mut self.state_2,
            &mut self.last_token,
        )?;
        self.pending_scores = decoded.scores;
        self.pending_details = decoded.details;
        self.frame_signals.extend(decoded.frames);
        self.last_step_stats.joint_calls = decoded.joint_calls;
        self.last_step_stats.signals = decoded.signal_time;
        Ok(decoded.tokens)
    }

    /// Greedy decode of the right-context frames from a copy of the decoder
    /// state, into `self.tentative`. The committed state is not touched.
    fn decode_tentative(
        &mut self,
        encoder_out: &Array3<f32>,
        start_frame: usize,
        available_frames: usize,
        absolute_frame_offset: usize,
    ) -> Result<()> {
        self.tentative.clear();
        self.last_step_stats.tentative = Duration::ZERO;
        self.last_step_stats.tentative_joint_calls = 0;
        if !self.tentative_enabled || start_frame >= available_frames {
            return Ok(());
        }

        let tentative_start = Instant::now();
        let mut state_1 = self.state_1.clone();
        let mut state_2 = self.state_2.clone();
        let mut last_token = self.last_token;
        let decoded = {
            let mut model = self
                .model
                .lock()
                .map_err(|e| Error::Model(format!("Failed to acquire model lock: {e}")))?;
            greedy_decode(
                &mut model,
                self.blank_id,
                encoder_out,
                start_frame,
                available_frames,
                absolute_frame_offset,
                &mut state_1,
                &mut state_2,
                &mut last_token,
            )?
        };

        let edge = available_frames.min(encoder_out.shape()[2]);
        self.tentative = decoded
            .tokens
            .iter()
            .zip(decoded.scores)
            .zip(decoded.details)
            .map(|((&(id, abs_frame), confidence), detail)| {
                let frame_idx = start_frame + (abs_frame - absolute_frame_offset);
                TentativeToken {
                    id,
                    text: self.vocab.decode_single(id),
                    abs_frame,
                    frames_to_edge: edge.saturating_sub(frame_idx + 1),
                    confidence,
                    detail,
                }
            })
            .collect();
        self.last_step_stats.tentative = tentative_start.elapsed();
        self.last_step_stats.tentative_joint_calls = decoded.joint_calls;
        self.last_step_stats.signals += decoded.signal_time;
        Ok(())
    }

    fn encoder_frame_to_seconds(frame: usize) -> f32 {
        (frame * SUBSAMPLING_FACTOR * HOP_LENGTH) as f32 / SAMPLE_RATE as f32
    }

    /// Scores of the last decoded tokens, filtered like `tokens_to_timed`.
    fn take_scores(&mut self, tokens: &[(usize, usize)]) -> Vec<f32> {
        let scores = std::mem::take(&mut self.pending_scores);
        tokens
            .iter()
            .zip(scores)
            .filter(|((id, _), _)| *id < self.blank_id)
            .map(|(_, s)| s)
            .collect()
    }

    /// Details of the last decoded tokens, filtered like `tokens_to_timed`.
    fn take_details(&mut self, tokens: &[(usize, usize)]) -> Vec<TokenDetail> {
        let details = std::mem::take(&mut self.pending_details);
        tokens
            .iter()
            .zip(details)
            .filter(|((id, _), _)| *id < self.blank_id)
            .map(|(_, d)| d)
            .collect()
    }

    fn tokens_to_timed(&self, tokens: &[(usize, usize)]) -> Vec<TimedToken> {
        tokens
            .iter()
            .filter(|(id, _)| *id < self.blank_id)
            .map(|&(id, frame)| TimedToken {
                text: self.vocab.decode_single(id),
                start: Self::encoder_frame_to_seconds(frame),
                end: Self::encoder_frame_to_seconds(frame + 1),
            })
            .collect()
    }

    fn decode_incremental_tokens(&self, tokens: &[(usize, usize)]) -> String {
        let mut text = String::new();
        for &(token, _) in tokens {
            if token < self.blank_id {
                text.push_str(&self.vocab.decode_single(token));
            }
        }
        text
    }

    fn transcribe_offline(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult> {
        self.reset();

        let features = audio::extract_features_raw(audio, sample_rate, channels, &self.preprocessor_config)?;
        let (encoded, encoded_len) = {
            let mut model = self
                .model
                .lock()
                .map_err(|e| Error::Model(format!("Failed to acquire model lock: {e}")))?;
            model.run_encoder(&features)?
        };
        let frame_count = (encoded_len as usize).min(encoded.shape()[2]);
        let tokens = self.decode_encoder_frames(&encoded, 0, frame_count, 0)?;
        self.accumulated_tokens = tokens.iter().map(|(id, _)| *id).collect();
        self.accumulated_timed_tokens = self.tokens_to_timed(&tokens);
        self.accumulated_token_scores = self.take_scores(&tokens);
        self.accumulated_token_details = self.take_details(&tokens);

        let text = self.get_transcript();
        let timed = match mode {
            Some(m) => process_timestamps(&self.accumulated_timed_tokens, m),
            None => self.accumulated_timed_tokens.clone(),
        };

        Ok(TranscriptionResult {
            text,
            tokens: timed,
        })
    }
}

impl Transcriber for ParakeetUnified {
    fn transcribe_samples(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult> {
        self.transcribe_offline(audio, sample_rate, channels, mode)
    }
}

#[cfg(test)]
mod tests {
    use super::{summarize_logits, UnifiedStreamingConfig};

    const BLANK: usize = 1024;

    fn logits(blank_logit: f32) -> Vec<f32> {
        let mut l: Vec<f32> = (0..=BLANK)
            .map(|i| ((i as f64 * 0.37).sin() * 3.0) as f32)
            .collect();
        l[BLANK] = blank_logit;
        l
    }

    fn close(a: f32, b: f64, tol: f64) {
        assert!((a as f64 - b).abs() < tol, "{a} vs {b}");
    }

    // Expected values come from the formulas in probe_stream.py (float64 numpy
    // softmax of the same f32 logits) and NeMo's entropy_tsallis_exp.
    #[test]
    fn signals_match_reference_when_blank_is_nearly_certain() {
        let (detail, frame) = summarize_logits(&logits(14.0), BLANK);
        close(frame.log10_nonblank, -2.381127631313727, 1e-5);
        close(detail.logp_blank, -0.004166552033409276, 1e-6);
        close(detail.entropy, 0.05226803183753447, 1e-6);
        close(detail.margin, 0.9958254838676014, 1e-6);
        close(detail.tsallis_confidence, 5.594661568947102e-09, 1e-10);
        let top: Vec<usize> = detail.top.iter().map(|t| t.0).collect();
        assert_eq!(top, [1024, 225, 242, 208, 259]);
        close(detail.top[1].1, -11.004173943009361, 1e-5);
        let nb: Vec<usize> = frame.top.iter().map(|t| t.0).collect();
        assert_eq!(nb, [225, 242, 208]);
        close(frame.top[2].1, -11.004288622345909, 1e-5);
    }

    #[test]
    fn signals_match_reference_when_blank_is_unlikely() {
        let (detail, frame) = summarize_logits(&logits(2.5), BLANK);
        close(frame.log10_nonblank, -0.0010524211645798337, 1e-6);
        close(detail.logp_blank, -6.0238408529392995, 1e-5);
        close(detail.entropy, 6.092414688349319, 1e-5);
        close(detail.margin, 9.799302771833218e-08, 1e-7);
        let top: Vec<usize> = detail.top.iter().map(|t| t.0).collect();
        assert_eq!(top, [225, 242, 208, 259, 191]);
        close(detail.top[0].1, -5.523848243915252, 1e-5);
    }

    #[test]
    fn tsallis_confidence_spans_zero_to_one() {
        let (uniform, _) = summarize_logits(&[0.0; BLANK + 1], BLANK);
        close(uniform.tsallis_confidence, 0.0, 1e-6);
        let mut peaked = vec![-100.0f32; BLANK + 1];
        peaked[7] = 100.0;
        let (peaked, frame) = summarize_logits(&peaked, BLANK);
        close(peaked.tsallis_confidence, 1.0, 1e-6);
        assert_eq!(peaked.top[0].0, 7);
        assert_eq!(frame.top[0].0, 7);
        close(frame.log10_nonblank, 0.0, 1e-6);
    }

    #[test]
    fn default_streaming_profile_aligns_to_subsampling() {
        let config = UnifiedStreamingConfig::default().validate().unwrap();
        assert_eq!(config.left_context_frames(), 560);
        assert_eq!(config.chunk_frames(), 56);
        assert_eq!(config.right_context_frames(), 56);
        assert_eq!(config.left_context_encoder_frames(), 70);
        assert_eq!(config.chunk_encoder_frames(), 7);
    }
}

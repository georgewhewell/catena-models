//! Gradium Phonon speech generation and Mimi decoding through Catena.
pub mod preparation;
use crate::{
    catena, hfhub, model,
    weights::{Weights, preparation::Binding as Operand},
};
use anyhow::{Context, Result, bail, ensure};
use catena_lang::runtime::{Artifact, MemOwn, Runtime, Value};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use tokenizers::{Encoding, Tokenizer};

pub use crate::tts::Audio;

pub const DEFAULT_MODEL: &str = "gradium/phonon-31m-en";
pub const DEFAULT_VOICE: &str = "Easton";
pub const CODEC: &str = "gradium/phonon-mimi";
pub const CODEC_REVISION: &str = "33519be1a8e61c6fa00c6e154e2d38e76a367919";
pub const VOICE_REPO: &str = "gradium/phonon-voices";
pub const VOICE_REPO_REVISION: &str = "0ee17205f254b5abc43b266c8a0f215565c352f7";
pub const VOICE_CATALOG: &str = "voices.jsonl";
pub const TOKENIZER: &str = "tokenizer.5.0.0.json";
/// Checkpoint files read from the model and codec repositories.
pub const MODEL_FILES: [&str; 5] = [
    "config.json",
    "generation_config.json",
    "processor_config.json",
    TOKENIZER,
    "model.safetensors",
];
pub const CODEC_FILES: [&str; 2] = ["config.json", "model.safetensors"];
/// Model tensors passed to `phonon.synthesize` as BF16, besides its bindings.
pub const LATENT_STD: &str = "flow_lm.emb_std";
pub const LATENT_MEAN: &str = "flow_lm.emb_mean";
const HIDDEN: usize = 512;
/// Elements of the `LATENT_STD` and `LATENT_MEAN` tensors.
pub const LATENT: usize = 32;
pub const CONTEXT: usize = 8192;
pub const SAMPLE_RATE: u32 = 24_000;
/// Values per frame of a voice's `speaker_wavs`.
pub const VOICE_CHANNELS: usize = HIDDEN;
/// F32 samples per generated frame.
pub const FRAME_SAMPLES: usize = 1920;
/// Elements of the zeroed F32 scratch `phonon.synthesize` takes.
pub const SCRATCH: usize = 2048;
/// Longer text is split into utterances of at most this many tokens.
pub const UTTERANCE_TOKENS: usize = 50;
/// Silence between consecutive utterances.
pub const PAUSE_SAMPLES: usize = SAMPLE_RATE as usize * 120 / 1000;

/// Validated model and codec configuration: the supported backbone depth.
pub struct Config {
    layers: u64,
    source: &'static str,
}

struct Assets {
    model: PathBuf,
    codec: PathBuf,
    voice: PathBuf,
}

#[derive(Deserialize)]
struct VoiceEntry {
    name: String,
    filename: String,
}

/// The requested name is absent from the Gradium voice catalog.
#[derive(Debug)]
pub struct UnknownVoice(String);

impl std::fmt::Display for UnknownVoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unknown Phonon voice {:?}. See {VOICE_REPO}/{VOICE_CATALOG} for names",
            self.0
        )
    }
}

impl std::error::Error for UnknownVoice {}

/// The checkpoint filename of a voice in the `VOICE_CATALOG` text.
pub fn voice_filename(catalog: &str, name: &str) -> Result<String> {
    let mut filename = None;
    for (index, line) in catalog.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let voice: VoiceEntry = serde_json::from_str(line)
            .with_context(|| format!("parsing {VOICE_CATALOG} line {}", index + 1))?;
        if voice.name == name {
            ensure!(filename.is_none(), "ambiguous Phonon voice {name:?}");
            ensure!(
                voice.filename.ends_with(".safetensors"),
                "invalid checkpoint filename for Phonon voice {name:?}"
            );
            filename = Some(voice.filename);
        }
    }
    filename.ok_or_else(|| UnknownVoice(name.to_owned()).into())
}

fn resolve_voice(name: &str) -> Result<PathBuf> {
    let name = name.trim();
    ensure!(!name.is_empty(), "Phonon voice name must not be blank");
    let root = hfhub::resolve_revision(VOICE_REPO, VOICE_REPO_REVISION, &[VOICE_CATALOG])?;
    let catalog = root.join(VOICE_CATALOG);
    let data = std::fs::read_to_string(&catalog)
        .with_context(|| format!("reading {}", catalog.display()))?;
    let filename = voice_filename(&data, name)?;
    let snapshot = hfhub::resolve_revision(VOICE_REPO, VOICE_REPO_REVISION, &[&filename])
        .with_context(|| format!("resolving Phonon voice {name:?}"))?;
    ensure!(
        snapshot == root,
        "Phonon voice catalog changed snapshots during resolution. Retry loading the voice"
    );
    Ok(snapshot.join(filename))
}

impl Assets {
    fn resolve(model: &str, voice: &str) -> Result<Self> {
        let voice = resolve_voice(voice)?;
        let model = hfhub::resolve_model_files(model, &MODEL_FILES)?;
        let codec = hfhub::resolve_revision(CODEC, CODEC_REVISION, &CODEC_FILES)?;
        Ok(Self {
            model,
            codec,
            voice,
        })
    }
}

impl Config {
    /// Read and validate the model and codec snapshot directories.
    pub fn read(model: &Path, codec: &Path) -> Result<Self> {
        use serde_json::{Value, json};
        let path = model.join("config.json");
        let config: Value = hfhub::read_json(&path)?;
        let (layers, source) = match config["num_hidden_layers"].as_u64() {
            Some(6) => (6, include_str!("../catena/phonon-6.hex")),
            Some(9) => (9, include_str!("../catena/phonon-9.hex")),
            _ => bail!(
                "{}: unsupported Phonon backbone depth {}",
                path.display(),
                config["num_hidden_layers"]
            ),
        };
        check_config_fields(
            &config,
            &json!({
                "model_type": "gradium_phonon_tts",
                "hidden_size": HIDDEN, "audio_prompt_dim": HIDDEN, "latent_dim": LATENT, "vocab_size": 4001,
                "intermediate_size": 2048, "head_dim": 64, "hidden_act": "gelu",
                "num_attention_heads": 8, "num_key_value_heads": 8,
                "layer_norm_eps": 1e-5, "frame_rate": 12.5, "max_position_embeddings": CONTEXT,
                "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
                "num_speakers": 1, "num_speakers_embedding_dim": 16, "duration_delta": 0.0,
                "duration_delta_embedding_dim": 128, "audio_bos_token_id": 4,
                "audio_prompt_token_id": 5, "audio_latent_token_id": 6,
                "flow_head_config": {
                    "flow_type": "lsd", "hidden_size": HIDDEN, "condition_dim": HIDDEN, "latent_dim": LATENT,
                    "frequency_embedding_size": 256, "num_hidden_layers": 6, "hidden_act": "silu",
                    "layer_norm_eps": 1e-6, "rms_norm_eps": 1e-5,
                }
            }),
            "",
        ).with_context(|| path.display().to_string())?;
        check_config(
            model.join("generation_config.json"),
            json!({
                "do_sample": true, "eos_threshold": 4.0, "min_frames_before_eos": 0,
                "num_flow_steps": 1, "temperature": 0.3,
            }),
        )?;
        check_config(
            model.join("processor_config.json"),
            json!({
                "audio_tokenizer": {"audio_tokenizer_name_or_path": CODEC},
                "frames_after_eos": 1, "frames_after_eos_short_text": 3, "short_text_max_words": 4,
                "fade_in_duration": 0.0,
            }),
        )?;
        check_config(
            codec.join("config.json"),
            json!({
                "model_type": "pocket_tts_audio_tokenizer", "audio_channels": 1, "latent_dim": LATENT,
                "sampling_rate": SAMPLE_RATE, "latent_stride": 16, "upsample_groups": HIDDEN,
                "hidden_size": HIDDEN, "head_dim": 64, "intermediate_size": 2048, "num_attention_heads": 8,
                "num_key_value_heads": 8, "num_hidden_layers": 2, "hidden_act": "gelu",
                "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
                "sliding_window": 251, "norm_eps": 1e-5, "use_causal_conv": true,
                "pad_mode": "constant", "trim_right_ratio": 1.0, "upsampling_ratios": [6, 5, 4],
                "num_filters": 64, "num_residual_layers": 1, "residual_kernel_size": 3,
                "compress": 2, "use_conv_shortcut": false, "kernel_size": 7, "last_kernel_size": 3,
            }),
        )?;
        Ok(Self { layers, source })
    }
}

/// The text budget `phonon.setup` returns for a voice of `frames` frames,
/// which must be positive.
pub fn text_limit(frames: usize) -> Result<usize> {
    ensure!((1..CONTEXT).contains(&frames), "invalid voice frame count");
    let limit = (((CONTEXT - 1 - frames) * 6).max(150) - 150) / 31;
    ensure!(limit > 0, "Phonon voice leaves no room for text");
    Ok(limit)
}

/// The most F32 samples `phonon.synthesize` returns for `tokens` text tokens.
pub fn waveform_samples(tokens: usize) -> usize {
    ((tokens + 6) * 25).div_ceil(6) * FRAME_SAMPLES
}

fn check_config(path: PathBuf, expected: serde_json::Value) -> Result<()> {
    check_config_fields(&hfhub::read_json(&path)?, &expected, "")
        .with_context(|| path.display().to_string())
}

fn check_config_fields(
    actual: &serde_json::Value,
    expected: &serde_json::Value,
    key: &str,
) -> Result<()> {
    if let Some(fields) = expected.as_object() {
        for (name, expected) in fields {
            let key = if key.is_empty() {
                name.clone()
            } else {
                format!("{key}.{name}")
            };
            check_config_fields(&actual[name], expected, &key)?;
        }
    } else {
        ensure!(
            actual == expected,
            "unsupported {key}: expected {expected}, got {actual}"
        );
    }
    Ok(())
}

/// Complete Hex bundle exporting `phonon.setup` and `phonon.synthesize`; the
/// receiver supplies only Catena's core stdlib (`catena::runtime_sources`).
///
/// `phonon.setup(voice: bf16[frames * VOICE_CHANNELS], frames: u64,
/// ...bindings)` returns `{voice, cond, bos, time, text_limit: u64}`. The
/// first four are BF16 buffers to pass back unchanged, and can be reused for
/// every utterance of the voice. `text_limit` is `text_limit(frames)`.
///
/// `phonon.synthesize(ids: u64[], voice, bos, cond, time, zero: f32[SCRATCH],
/// LATENT_STD: bf16[LATENT], LATENT_MEAN: bf16[LATENT], frames: u64,
/// tokens: u64, words: u64, seed: u64, utterance: u64, ...bindings)` returns
/// `{waveform, descriptor}`: see `decode_utterance`. The waveform holds up to
/// `waveform_samples(tokens)` F32 samples.
pub fn bundle(config: &Config) -> String {
    let mut source = include_str!("../catena/memory.hex").to_owned();
    for fragment in [
        include_str!("../catena/numeric.hex"),
        include_str!("../catena/resources.hex"),
        include_str!("../catena/cooperative.hex"),
        include_str!("../catena/bf16_simt.hex"),
        &crate::weights::cast_to_bf16_source(),
        include_str!("../catena/whisper.hex"),
        include_str!("../catena/phonon.hex"),
        config.source,
    ] {
        crate::qwen::fused::append_hex(&mut source, fragment);
    }
    source.replace(" bf16x2.dot-f32 ", " stdlib.models.math.dot-pair-f32 ")
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Packing {
    Conv,
    Qkv,
    Transposed,
}

/// One weight argument: the named tensors of `shape`, concatenated and packed.
#[derive(Debug, Deserialize)]
pub struct Binding {
    pub names: Vec<String>,
    pub shape: Vec<usize>,
    pub packing: Option<Packing>,
    /// Read from the codec checkpoint and stored as F32. Model tensors are BF16.
    #[serde(default)]
    pub codec: bool,
}

/// The weight arguments that follow the inputs of `setup` or `synthesize`, in order.
pub fn bindings(config: &Config, entry: &str) -> Result<Vec<Binding>> {
    let key = match entry {
        "setup" => "setup".to_owned(),
        "synthesize" => format!("synthesize-{}", config.layers),
        _ => bail!("unknown Phonon entry {entry}"),
    };
    serde_json::from_str::<Bindings>(include_str!("weights.json"))?
        .remove(&key)
        .with_context(|| format!("missing generated Phonon bindings for {key}"))
}

type Bindings = BTreeMap<String, Vec<Binding>>;

/// Validate the catalog voice tensor and return the number of conditioning frames.
pub fn voice_frames(shape: &[usize]) -> Result<usize> {
    let [1, VOICE_CHANNELS, frames] = *shape else {
        bail!("expected speaker_wavs with shape [1, {VOICE_CHANNELS}, frames]");
    };
    text_limit(frames)?;
    Ok(frames)
}

/// One utterance: its token IDs and spoken word count.
pub struct Prompt {
    pub ids: Vec<u64>,
    pub words: usize,
}

impl Prompt {
    fn tokenize(tokenizer: &Tokenizer, text: &str) -> Result<Encoding> {
        let text = text.trim();
        ensure!(!text.is_empty(), "text must contain a spoken prompt");
        let encoded = tokenizer.encode(text, false).map_err(anyhow::Error::msg)?;
        let ids = encoded.get_ids();
        ensure!(!ids.is_empty(), "text produces no tokens");
        ensure!(
            ids.iter().all(|id| !(1..=6).contains(id) && *id < 4001),
            "text contains a reserved Phonon token"
        );
        Ok(encoded)
    }

    fn from_encoding(tokenizer: &Tokenizer, encoded: &Encoding) -> Result<Self> {
        let ids = encoded.get_ids();
        ensure!(ids.len() < CONTEXT, "text exceeds Phonon's context window");
        let normalized = tokenizer.decode(ids, true).map_err(anyhow::Error::msg)?;
        let words = normalized.split_whitespace().count();
        ensure!(words > 0, "text produces no spoken words");
        Ok(Self {
            ids: ids.iter().map(|&x| u64::from(x)).collect(),
            words,
        })
    }

    /// Split text into utterances of at most `UTTERANCE_TOKENS` tokens.
    pub fn split(tokenizer: &Tokenizer, text: &str) -> Result<Vec<Self>> {
        let text = text.trim();
        let encoded = Self::tokenize(tokenizer, text)?;
        if encoded.len() <= UTTERANCE_TOKENS {
            return Ok(vec![Self::from_encoding(tokenizer, &encoded)?]);
        }

        let offsets = encoded.get_offsets();
        let mut prompts = Vec::new();
        let (mut first, mut start) = (0, 0);
        while offsets.len() - first > UTTERANCE_TOKENS {
            // Prefer the latest sentence end, then a clause or word boundary.
            // Use source offsets so model normalization runs again for each chunk.
            let (next, end) = (first + 1..=first + UTTERANCE_TOKENS)
                .filter_map(|next| {
                    let end = offsets[next - 1].1;
                    (end > start
                        && end < text.len()
                        && end <= offsets[next].0
                        && text.is_char_boundary(end))
                    .then_some((next, end))
                })
                .max_by_key(|&(next, end)| (boundary_rank(text, end), next))
                .context("could not split Phonon text at a token boundary")?;
            let chunk = Self::tokenize(tokenizer, &text[start..end])?;
            prompts.push(Self::from_encoding(tokenizer, &chunk)?);
            (first, start) = (next, end);
        }
        let chunk = Self::tokenize(tokenizer, &text[start..])?;
        prompts.push(Self::from_encoding(tokenizer, &chunk)?);
        Ok(prompts)
    }
}

fn boundary_rank(text: &str, end: usize) -> u8 {
    let (left, right) = text.split_at(end);
    let left = left.trim_end();
    if right.starts_with(['.', '!', '?', '…', ',', ';', ':', '"', '”', '»', ')', ']']) {
        return 0;
    }
    let punctuation = left.trim_end_matches(['"', '\'', '”', '’', '»', ')', ']']);
    let mut chars = punctuation.chars().rev();
    let mark = chars.next();
    let decimal = matches!(mark, Some('.' | ','))
        && chars.next().is_some_and(|c| c.is_ascii_digit())
        && right.starts_with(|c: char| c.is_ascii_digit());
    match mark {
        Some('.' | '!' | '?' | '…') if !decimal => 3,
        Some(',' | ';' | ':') if !decimal => 2,
        _ if left.len() < end || right.starts_with(char::is_whitespace) => 1,
        _ => 0,
    }
}

pub fn read_tokenizer(dir: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(dir.join(TOKENIZER))
        .map_err(anyhow::Error::msg)
        .context("loading Phonon tokenizer")
}

fn outputs<const N: usize>(values: Vec<Value<'static>>) -> Result<[MemOwn; N]> {
    values
        .into_iter()
        .map(|v| match v {
            Value::MemOwn(mem) => Ok(mem),
            _ => bail!("invalid Phonon entrypoint result"),
        })
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| anyhow::anyhow!("invalid Phonon entrypoint result count"))
}

type Key = (bool, String);

struct Graphs {
    runtime: Runtime,
    artifact: Artifact,
    direct: BTreeMap<Key, MemOwn>,
    prepared: Vec<MemOwn>,
    setup: Vec<Operand<Key>>,
    synth: Vec<Operand<Key>>,
    latents: [Operand<Key>; 2],
}

impl Graphs {
    fn load(model: &Weights, codec: &Weights, voice: &Weights, config: &Config) -> Result<Self> {
        let shard = |is_codec: bool| if is_codec { codec } else { model };
        let voice_dtype = voice
            .tensors
            .get("speaker_wavs")
            .context("voice has no speaker_wavs")?
            .dtype;
        let program = preparation::build(config, voice_dtype, |is_codec, name, shape| {
            let weights = shard(is_codec);
            weights.validate(name, shape)?;
            let info = &weights.tensors[name];
            Ok(((is_codec, name.to_owned()), info.dtype, info.byte_offset()))
        })?;
        let (runtime, artifact) = catena::load_runtime(
            catena::runtime_sources(&program.source),
            model::configured_gpu_dialect()?,
        )?;
        let upload = |key: &Key| shard(key.0).upload_raw(&runtime, &key.1);
        let raw = program.raw.iter().map(upload).collect::<Result<Vec<_>>>()?;
        let prepared = artifact
            .exec_values(
                preparation::WEIGHT_ENTRY,
                raw.iter().map(|m| m.as_ref().into()).collect(),
            )?
            .into_iter()
            .map(|v| match v {
                Value::MemOwn(m) => Ok(m),
                _ => bail!("expected prepared weight"),
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            prepared.len() == program.outputs,
            "wrong number of prepared weights"
        );
        drop(raw);
        let mut direct = BTreeMap::new();
        for operand in program
            .setup
            .iter()
            .chain(&program.synth)
            .chain(&program.latents)
        {
            if let Operand::Asset(key) = operand
                && !direct.contains_key(key)
            {
                direct.insert(key.clone(), upload(key)?);
            }
        }
        Ok(Self {
            runtime,
            artifact,
            direct,
            prepared,
            setup: program.setup,
            synth: program.synth,
            latents: program.latents,
        })
    }

    fn weight(&self, operand: &Operand<Key>) -> &MemOwn {
        match operand {
            Operand::Asset(key) => &self.direct[key],
            Operand::Prepared(index) => &self.prepared[*index],
        }
    }
    fn execute<'a>(
        &'a self,
        entry: &str,
        mut inputs: Vec<Value<'a>>,
    ) -> Result<Vec<Value<'static>>> {
        let weights = match entry {
            "setup" => &self.setup,
            "synthesize" => &self.synth,
            _ => bail!("unknown Phonon entry"),
        };
        inputs.extend(
            weights
                .iter()
                .map(|operand| self.weight(operand).as_ref().into()),
        );
        Ok(self
            .artifact
            .exec_values(&format!("phonon.{entry}"), inputs)?)
    }
    fn run<'a, const N: usize>(
        &'a self,
        entry: &str,
        inputs: Vec<Value<'a>>,
    ) -> Result<[MemOwn; N]> {
        outputs(self.execute(entry, inputs)?)
            .with_context(|| format!("decoding Phonon {entry} outputs"))
    }
}

/// Phonon speech synthesis with a catalog voice and Mimi decoder.
pub struct Phonon {
    graphs: Graphs,
    tokenizer: Tokenizer,
    voice_frames: usize,
    voice: MemOwn,
    cond: MemOwn,
    bos: MemOwn,
    time: MemOwn,
    zero: MemOwn,
}

impl Phonon {
    pub fn load() -> Result<Self> {
        Self::load_model(DEFAULT_MODEL)
    }

    /// Load a compatible Phonon checkpoint from the Hugging Face Hub.
    pub fn load_model(model: &str) -> Result<Self> {
        Self::load_model_with_voice(model, DEFAULT_VOICE)
    }

    /// Load a compatible Phonon checkpoint and a voice by its catalog name.
    pub fn load_model_with_voice(model: &str, voice: &str) -> Result<Self> {
        let assets = Assets::resolve(model, voice)?;
        let config = Config::read(&assets.model, &assets.codec)?;
        let tokenizer = read_tokenizer(&assets.model)?;
        let model = Weights::open(&assets.model.join("model.safetensors"))?;
        let codec = Weights::open(&assets.codec.join("model.safetensors"))?;
        let voice_weights = Weights::open(&assets.voice)?;
        let frames = voice_frames(
            &voice_weights
                .tensors
                .get("speaker_wavs")
                .context("missing speaker_wavs")?
                .shape,
        )?;
        let graphs = Graphs::load(&model, &codec, &voice_weights, &config)?;
        let runtime = &graphs.runtime;
        let raw_voice = voice_weights.upload_raw(runtime, "speaker_wavs")?;
        let [conditioning] = outputs(
            graphs
                .artifact
                .exec_values(preparation::VOICE_ENTRY, vec![raw_voice.as_ref().into()])?,
        )?;
        let mut setup = graphs.execute(
            "setup",
            vec![conditioning.as_ref().into(), (frames as u64).into()],
        )?;
        let Some(Value::U64(text_limit)) = setup.pop() else {
            bail!("invalid Phonon setup result");
        };
        ensure!(text_limit > 0, "Phonon voice leaves no room for text");
        let [voice, cond, bos, time] = outputs(setup)?;
        let zero = runtime.mem_f32(&[0.0; SCRATCH])?;
        Ok(Self {
            graphs,
            tokenizer,
            voice_frames: frames,
            voice,
            cond,
            bos,
            time,
            zero,
        })
    }

    /// Synthesize text as one or more utterances with a reproducible sampling seed.
    pub fn synthesize(&self, text: &str, seed: u64, timings: bool) -> Result<Audio> {
        let prompts = Prompt::split(&self.tokenizer, text)?;
        let mut samples = Vec::new();
        let mut synthesis = Duration::ZERO;
        for (index, prompt) in prompts.iter().enumerate() {
            let started = Instant::now();
            let chunk = self.synthesize_utterance(prompt, seed, index as u64)?;
            synthesis += started.elapsed();
            if index == 0 {
                samples = chunk;
            } else {
                samples.resize(samples.len() + PAUSE_SAMPLES, 0.0);
                samples.extend(chunk);
            }
        }
        if timings {
            let synthesis = synthesis.as_secs_f64();
            let seconds = samples.len() as f64 / f64::from(SAMPLE_RATE);
            eprintln!(
                "Phonon synthesis {synthesis:.3}s, audio {seconds:.3}s, real-time factor {:.3} ({} utterances)",
                synthesis / seconds,
                prompts.len()
            );
        }
        Ok(Audio {
            samples,
            sample_rate: SAMPLE_RATE,
        })
    }

    fn synthesize_utterance(&self, prompt: &Prompt, seed: u64, index: u64) -> Result<Vec<f32>> {
        let ids = self.graphs.runtime.mem_u64(&prompt.ids)?;
        let [waveform, descriptor] = self.graphs.run(
            "synthesize",
            vec![
                ids.as_ref().into(),
                self.voice.as_ref().into(),
                self.bos.as_ref().into(),
                self.cond.as_ref().into(),
                self.time.as_ref().into(),
                self.zero.as_ref().into(),
                self.graphs.weight(&self.graphs.latents[0]).as_ref().into(),
                self.graphs.weight(&self.graphs.latents[1]).as_ref().into(),
                (self.voice_frames as u64).into(),
                (prompt.ids.len() as u64).into(),
                (prompt.words as u64).into(),
                seed.into(),
                index.into(),
            ],
        )?;
        let samples = waveform.try_to_f32_vec()?;
        ensure!(
            samples.len() <= waveform_samples(prompt.ids.len()),
            "Phonon returned more audio than its frame budget"
        );
        decode_utterance(&descriptor.try_to_u64_vec()?, samples)
    }
}

/// The samples of one `phonon.synthesize` result. The descriptor is
/// `[sample_count, status]`, and only status 0 carries audio.
pub fn decode_utterance(descriptor: &[u64], mut samples: Vec<f32>) -> Result<Vec<f32>> {
    let [sample_count, error] = descriptor else {
        bail!("invalid Phonon result descriptor");
    };
    match error {
        0 => {}
        1 => bail!("text and audio frame budget exceed Phonon's context window"),
        2 => bail!("Phonon produced an invalid end-of-speech score"),
        _ => bail!("invalid Phonon generation status"),
    }
    let valid = usize::try_from(*sample_count).context("invalid Phonon sample count")?;
    ensure!(
        valid > 0 && valid <= samples.len(),
        "invalid Phonon sample count"
    );
    samples.truncate(valid);
    ensure!(
        samples.iter().all(|x| x.is_finite()),
        "Mimi produced non-finite audio"
    );
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIGS: [Config; 2] = [
        Config {
            layers: 6,
            source: include_str!("../catena/phonon-6.hex"),
        },
        Config {
            layers: 9,
            source: include_str!("../catena/phonon-9.hex"),
        },
    ];

    #[test]
    fn budgets_and_binding_sizes() -> Result<()> {
        assert_eq!(text_limit(125)?, 1556);
        assert_eq!(text_limit(8135)?, 6);
        for frames in [0, 8161, CONTEXT] {
            assert!(text_limit(frames).is_err());
        }
        assert_eq!(waveform_samples(50), 234 * FRAME_SAMPLES);
        Ok(())
    }

    #[test]
    fn bindings_match_entry_signatures() -> Result<()> {
        for (config, synthesize) in CONFIGS.iter().zip([154, 178]) {
            let source = bundle(config);
            for (entry, inputs, count) in [("setup", 1, 17), ("synthesize", 8, synthesize)] {
                let bindings = bindings(config, entry)?;
                assert_eq!(bindings.len(), count);
                // Model arguments are single unpacked tensors.
                assert!(
                    bindings
                        .iter()
                        .all(|b| b.codec || (b.names.len() == 1 && b.packing.is_none()))
                );
                let signature = source
                    .lines()
                    .find(|line| line.starts_with(&format!("(def program phonon.{entry} ")))
                    .and_then(|line| line.split(" -> ").next())
                    .context("missing entry")?;
                assert_eq!(
                    signature.matches("((cap.ref mem) val)").count(),
                    inputs + count
                );
            }
            assert!(bindings(config, "decode").is_err());
        }
        Ok(())
    }

    #[test]
    fn catalog_and_result_validation() -> Result<()> {
        let catalog = "{\"name\":\"A\",\"filename\":\"a.safetensors\"}\n\n\
            {\"name\":\"B\",\"filename\":\"b.bin\"}\n\
            {\"name\":\"C\",\"filename\":\"c.safetensors\"}\n\
            {\"name\":\"C\",\"filename\":\"d.safetensors\"}";
        assert_eq!(voice_filename(catalog, "A")?, "a.safetensors");
        assert!(voice_filename(catalog, "B").is_err());
        assert!(voice_filename(catalog, "C").is_err());
        assert!(
            voice_filename(catalog, "D")
                .unwrap_err()
                .is::<UnknownVoice>()
        );

        assert_eq!(decode_utterance(&[2, 0], vec![0.5, -0.5, 9.])?, [0.5, -0.5]);
        for descriptor in [&[2, 1][..], &[2, 2], &[2, 3], &[0, 0], &[4, 0], &[2]] {
            assert!(decode_utterance(descriptor, vec![0.; 3]).is_err());
        }
        assert!(decode_utterance(&[1, 0], vec![f32::NAN]).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires the three cached Gradium repositories"]
    fn cached_assets_and_prompt() -> Result<()> {
        let assets = Assets::resolve(DEFAULT_MODEL, DEFAULT_VOICE)?;
        let config = Config::read(&assets.model, &assets.codec)?;
        let model = Weights::open(&assets.model.join("model.safetensors"))?;
        let codec = Weights::open(&assets.codec.join("model.safetensors"))?;
        let voice = Weights::open(&assets.voice)?;
        let info = &voice.tensors["speaker_wavs"];
        assert_eq!(voice_frames(&info.shape)?, 125);
        preparation::build(&config, info.dtype, |is_codec, name, shape| {
            let checkpoint = if is_codec { &codec } else { &model };
            checkpoint.validate(name, shape)?;
            let t = &checkpoint.tensors[name];
            Ok(((), t.dtype, t.byte_offset()))
        })?;
        let tokenizer = read_tokenizer(&assets.model)?;
        let prompts = Prompt::split(&tokenizer, " Hello from Phonon. ")?;
        let [prompt] = prompts.as_slice() else {
            bail!("short text should produce one utterance");
        };
        assert_eq!(prompt.ids, [3248, 744, 949, 1286, 697, 860, 663]);
        assert_eq!(prompt.words, 3);
        assert_eq!(
            Prompt::split(&tokenizer, "This is a longer sentence")?[0].words,
            5
        );
        assert!(Prompt::split(&tokenizer, "   ").is_err());
        assert!(Prompt::split(&tokenizer, "Hello <|audio_bos|>").is_err());
        let long = "Hello from Phonon. ".repeat(1_500);
        let utterances = Prompt::split(&tokenizer, &long)?;
        assert_eq!(
            utterances.iter().map(|prompt| prompt.words).sum::<usize>(),
            4_500
        );
        assert!(
            utterances
                .iter()
                .all(|prompt| prompt.ids.len() <= UTTERANCE_TOKENS)
        );
        let long_clause = "bonjour ".repeat(100);
        let utterances = Prompt::split(&tokenizer, &long_clause)?;
        assert!(utterances.len() > 1);
        assert_eq!(
            utterances.iter().map(|prompt| prompt.words).sum::<usize>(),
            100
        );
        Ok(())
    }

    #[test]
    #[ignore = "compiles both Phonon backbones for HIP and CUDA"]
    fn synthesis_compiles_without_host_reads() -> Result<()> {
        use catena_lang::runtime::GpuDialect;
        for config in CONFIGS {
            let source = bundle(&config);
            for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
                let module = catena::compile_sources(catena::runtime_sources(&source), dialect)?;
                assert!(!module.source.contains("= catena_host_read("));
            }
        }
        Ok(())
    }
}

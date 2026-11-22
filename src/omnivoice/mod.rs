//! Text-only k2-fsa/OmniVoice synthesis with the bundled Higgs audio decoder.
pub mod graph;
pub mod prompt;
pub mod setup;

use crate::{
    catena, hfhub, model,
    spec::ModelConfig,
    tts::Audio,
    weights::{Weights, preparation::Binding},
};
use anyhow::{Context, Result, anyhow, ensure};
use catena_lang::runtime::{Artifact, MemOwn, Runtime, Value};
use serde_json::{Value as Json, json};
use std::path::Path;
use tokenizers::Tokenizer;

pub const DEFAULT_MODEL: &str = "k2-fsa/OmniVoice";
pub const SAMPLE_RATE: u32 = 24_000;
pub const FRAME_SAMPLES: usize = 960;
pub const MAX_FRAMES: usize = 30 * SAMPLE_RATE as usize / FRAME_SAMPLES;
pub const FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "model.safetensors",
    "audio_tokenizer/config.json",
    "audio_tokenizer/model.safetensors",
];

pub struct OmniVoice {
    runtime: Runtime,
    artifact: Artifact,
    weights: Vec<MemOwn>,
    #[cfg(test)]
    model_weights: usize,
    tokenizer: Tokenizer,
    config: ModelConfig,
}

impl OmniVoice {
    /// Resolve the OmniVoice checkpoint through Hugging Face Hub and load the HEX graph.
    pub fn load() -> Result<Self> {
        let directory = hfhub::resolve_model_files(DEFAULT_MODEL, FILES)?;
        Self::load_snapshot(&directory, false)
    }

    fn load_snapshot(directory: &Path, diagnostics: bool) -> Result<Self> {
        let config = config(directory)?;
        let tokenizer = Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|err| anyhow!("loading OmniVoice tokenizer: {err}"))?;
        let dialect = model::configured_gpu_dialect()?;
        let model = Weights::open(&directory.join(setup::MODEL))?;
        let codec = Weights::open(&directory.join(setup::CODEC))?;
        let files = [&model, &codec];
        let setup = setup::program(&config, diagnostics, dialect, |file, name, shape| {
            let index = usize::from(file == setup::CODEC);
            files[index].validate(name, shape)?;
            Ok(((index, name.to_owned()), files[index].tensors[name].dtype))
        })?;
        let (runtime, artifact) =
            catena::load_runtime(catena::runtime_sources(&setup.source), dialect)
                .context("compiling and loading OmniVoice HEX")?;
        let upload = |(file, name): &(usize, String)| files[*file].upload_raw(&runtime, name);
        let raw = setup
            .inputs
            .iter()
            .map(upload)
            .collect::<Result<Vec<_>>>()?;
        let outputs = artifact.exec_values(
            "omnivoice.setup",
            raw.iter().map(|m| m.as_ref().into()).collect(),
        )?;
        drop(raw);
        ensure!(
            outputs.len() == setup.outputs,
            "invalid OmniVoice setup output count"
        );
        let mut outputs: Vec<_> = outputs.into_iter().map(Some).collect();
        let weights = setup
            .weights
            .into_iter()
            .map(|binding| match binding {
                Binding::Asset(key) => upload(&key),
                Binding::Prepared(i) => match outputs[i].take() {
                    Some(Value::MemOwn(weight)) => Ok(weight),
                    _ => Err(anyhow!("invalid OmniVoice setup output {i}")),
                },
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            runtime,
            artifact,
            weights,
            #[cfg(test)]
            model_weights: setup.model_weights,
            tokenizer,
            config,
        })
    }

    /// Generate a single text-only utterance with reproducible sampling.
    pub fn synthesize(&self, text: &str, seed: u64) -> Result<Audio> {
        self.synthesize_with_instruct(text, seed, None)
    }

    /// Generate an utterance with optional English voice directions.
    /// Free-form emotion directions are experimental.
    pub fn synthesize_with_instruct(
        &self,
        text: &str,
        seed: u64,
        instruct: Option<&str>,
    ) -> Result<Audio> {
        let (tokens, frames) = prepare(&self.tokenizer, &self.config, text, instruct)?;
        let text = self.runtime.mem_u64(&tokens)?;
        let mut args = vec![
            text.as_ref().into(),
            (tokens.len() as u64).into(),
            (frames as u64).into(),
            seed.into(),
        ];
        args.extend(self.weights.iter().map(|w| w.as_ref().into()));
        let wave = self.execute("omnivoice.synthesize", args)?;
        let samples = wave.try_to_f32_vec()?;
        ensure!(
            samples.len() == waveform_samples(frames) && samples.iter().all(|x| x.is_finite()),
            "OmniVoice returned an invalid waveform"
        );
        Ok(Audio {
            samples,
            sample_rate: SAMPLE_RATE,
        })
    }

    fn execute(&self, entry: &str, args: Vec<Value<'_>>) -> Result<MemOwn> {
        let mut results = self
            .artifact
            .exec_values(entry, args)
            .context("executing OmniVoice HEX")?;
        ensure!(results.len() == 1, "invalid OmniVoice output count");
        match results.pop() {
            Some(Value::MemOwn(samples)) => Ok(samples),
            _ => Err(anyhow!("invalid OmniVoice output type")),
        }
    }
}

/// Prompt tokens and the frame count of one utterance, checked against the checkpoint.
pub fn prepare(
    tokenizer: &Tokenizer,
    config: &ModelConfig,
    text: &str,
    instruct: Option<&str>,
) -> Result<(Vec<u64>, usize)> {
    let tokens = prompt::tokens(tokenizer, text, instruct)?;
    let frames = prompt::duration_frames(text);
    ensure!(
        frames <= MAX_FRAMES,
        "OmniVoice supports up to 30 seconds per utterance"
    );
    ensure!(
        tokens.len() + frames <= config.max_position_embeddings
            && tokens.iter().all(|&id| id < config.vocabulary_size as u64),
        "OmniVoice prompt exceeds the checkpoint's context or vocabulary"
    );
    Ok((tokens, frames))
}

/// The F32 sample count `omnivoice.synthesize` returns for a frame count.
pub fn waveform_samples(frames: usize) -> usize {
    frames * FRAME_SAMPLES + SAMPLE_RATE as usize / 5
}

/// Validate the model and codec configuration files of a snapshot directory.
pub fn config(directory: &Path) -> Result<ModelConfig> {
    let model: Json = hfhub::read_json(directory.join("config.json"))?;
    let codec: Json = hfhub::read_json(directory.join("audio_tokenizer/config.json"))?;
    let mut qwen = ModelConfig::qwen3_0_6b();
    qwen.vocabulary_size = 151_676;
    qwen.eos_token_id = Some(151_645);
    let fields = [
        (&model, "/model_type", json!("omnivoice")),
        (&model, "/num_audio_codebook", json!(8)),
        (&model, "/audio_vocab_size", json!(1025)),
        (&model, "/audio_mask_id", json!(1024)),
        (&model, "/llm_config/model_type", json!("qwen3")),
        (
            &model,
            "/llm_config/vocab_size",
            json!(qwen.vocabulary_size),
        ),
        (&model, "/llm_config/hidden_size", json!(qwen.hidden_size)),
        (
            &model,
            "/llm_config/intermediate_size",
            json!(qwen.intermediate_size),
        ),
        (
            &model,
            "/llm_config/num_hidden_layers",
            json!(qwen.num_layers),
        ),
        (
            &model,
            "/llm_config/num_attention_heads",
            json!(qwen.num_attention_heads),
        ),
        (
            &model,
            "/llm_config/num_key_value_heads",
            json!(qwen.num_key_value_heads),
        ),
        (&model, "/llm_config/head_dim", json!(qwen.head_dim)),
        (
            &model,
            "/llm_config/rope_parameters/rope_theta",
            json!(qwen.rope_theta as u64),
        ),
        (
            &model,
            "/llm_config/rope_parameters/rope_type",
            json!("default"),
        ),
        (&model, "/llm_config/rms_norm_eps", json!(1.0e-6)),
        (&model, "/llm_config/attention_bias", json!(false)),
        (&model, "/llm_config/hidden_act", json!("silu")),
        (
            &model,
            "/llm_config/max_position_embeddings",
            json!(qwen.max_position_embeddings),
        ),
        (&codec, "/model_type", json!("higgs_audio_v2_tokenizer")),
        (&codec, "/sample_rate", json!(SAMPLE_RATE)),
        (&codec, "/codebook_size", json!(1024)),
        (&codec, "/codebook_dim", json!(64)),
        (&codec, "/acoustic_model_config/hidden_size", json!(256)),
        (
            &codec,
            "/acoustic_model_config/decoder_hidden_size",
            json!(1024),
        ),
        (
            &codec,
            "/acoustic_model_config/hop_length",
            json!(FRAME_SAMPLES),
        ),
        (
            &codec,
            "/acoustic_model_config/upsampling_ratios",
            json!(graph::STRIDES),
        ),
    ];
    for (config, path, expected) in fields {
        let actual = config.pointer(path);
        ensure!(
            actual == Some(&expected),
            "unsupported OmniVoice config field {path}: expected {expected}, got {actual:?}"
        );
    }
    Ok(qwen)
}

#[cfg(test)]
mod tests {
    use super::*;
    use catena_lang::runtime::GpuDialect;

    #[test]
    fn synthesize_signature_matches_bindings() -> Result<()> {
        for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
            let p = setup::program(&ModelConfig::qwen3_0_6b(), false, dialect, |_, _, _| {
                Ok(((), safetensors::Dtype::F32))
            })?;
            let signature = |name: &str| {
                p.source
                    .lines()
                    .find(|line| line.starts_with(&format!("(def program {name} :")))
                    .unwrap()
                    .split_once(" -> ")
                    .unwrap()
            };
            let (inputs, outputs) = signature("omnivoice.setup");
            assert_eq!(inputs.matches(graph::REF).count(), p.inputs.len());
            assert_eq!(outputs.matches(graph::OWN).count(), p.outputs);
            let (inputs, _) = signature("omnivoice.synthesize");
            assert_eq!(inputs.matches(graph::REF).count(), 1 + p.weights.len());
            assert_eq!(inputs.matches("(u64 val)").count(), 3);
        }
        assert_eq!(waveform_samples(67), 69_120);
        Ok(())
    }

    #[test]
    #[ignore = "compiles the full OmniVoice graph for HIP and CUDA"]
    fn graph_compiles_for_both_gpu_dialects() -> Result<()> {
        for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
            let p = setup::program(&ModelConfig::qwen3_0_6b(), false, dialect, |_, _, _| {
                Ok(((), safetensors::Dtype::F32))
            })?;
            let module = catena::compile_sources(catena::runtime_sources(&p.source), dialect)?;
            ensure!(
                !module.source.contains("catena_host_read"),
                "unexpected model readback"
            );
        }
        Ok(())
    }

    fn compare_reference(output: &[u16], variable: &str, minimum: f32) -> Result<()> {
        let Some(path) = std::env::var_os(variable) else {
            return Ok(());
        };
        let bytes = std::fs::read(path)?;
        ensure!(
            bytes.len() == 4 * output.len(),
            "unexpected reference length"
        );
        let reference = bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        let wave: Vec<_> = output
            .iter()
            .map(|&x| crate::weights::bf16_to_f32(x))
            .collect();
        let dot = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
        let similarity =
            dot(&wave, &reference) / (dot(&wave, &wave) * dot(&reference, &reference)).sqrt();
        eprintln!("{variable} cosine similarity {similarity:.4}");
        ensure!(
            similarity > minimum,
            "GPU output diverged from the reference"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a GPU and the OmniVoice checkpoint"]
    fn gpu_reference() -> Result<()> {
        let directory = hfhub::resolve_model_files(DEFAULT_MODEL, FILES)?;
        let model = OmniVoice::load_snapshot(&directory, true)?;
        let text = "This is a sentence without any voice prompt.";
        let ids = prompt::tokens(&model.tokenizer, text, None)?;
        assert_eq!(
            ids,
            [
                151670, 4064, 151671, 151672, 4064, 151673, 151674, 1986, 374, 264, 11652, 2041,
                894, 7743, 9934, 13, 151675
            ]
        );
        let conditioned = prompt::tokens(&model.tokenizer, text, Some(" excited and happy "))?;
        assert_eq!(
            &conditioned[..9],
            &[151670, 4064, 151671, 151672, 39441, 1608, 323, 6247, 151673]
        );
        assert_eq!(&conditioned[9..], &ids[6..]);
        assert_eq!(prompt::duration_frames(text), 67);
        let multilingual = "你好 （世界） [sigh]！";
        assert_eq!(
            prompt::tokens(&model.tokenizer, multilingual, None)?,
            [
                151670, 4064, 151671, 151672, 4064, 151673, 151674, 108386, 7, 99489, 8, 220,
                11880, 1090, 60, 6313, 151675
            ]
        );
        assert_eq!(prompt::duration_frames("你好，世界！"), 38);

        let frames = 3u64;
        let codes: Vec<u64> = (0..frames * 8).map(|i| (i * 37 + 13) % 1024).collect();
        let mut args = vec![model.runtime.mem_u64(&codes)?.into(), frames.into()];
        args.extend(
            model.weights[model.model_weights..]
                .iter()
                .map(|w| w.as_ref().into()),
        );
        let wave = model.execute("omnivoice.decode", args)?.try_to_u16_vec()?;
        ensure!(
            wave.len() == frames as usize * FRAME_SAMPLES,
            "unexpected decoder length"
        );
        compare_reference(&wave, "OMNIVOICE_REFERENCE_DECODER", 0.95)?;

        let tokens = model.runtime.mem_u64(&ids)?;
        for (frames, reference) in [
            (3u64, "OMNIVOICE_REFERENCE_LOGITS"),
            (127, "OMNIVOICE_REFERENCE_LOGITS_127"),
            (128, "OMNIVOICE_REFERENCE_LOGITS_128"),
            (129, "OMNIVOICE_REFERENCE_WIDE_LOGITS"),
        ] {
            let mut masked = vec![1024; frames as usize * 8];
            masked[8..16].copy_from_slice(&codes[..8]);
            let mut args = vec![
                tokens.as_ref().into(),
                (ids.len() as u64).into(),
                frames.into(),
                model.runtime.mem_u64(&masked)?.into(),
            ];
            args.extend(
                model.weights[..model.model_weights]
                    .iter()
                    .map(|w| w.as_ref().into()),
            );
            let logits = model.execute("omnivoice.logits", args)?.try_to_u16_vec()?;
            ensure!(
                logits.len() == frames as usize * 8200,
                "unexpected logit count"
            );
            compare_reference(&logits, reference, 0.98)?;
        }

        let codes = model.runtime.mem_u64(&[1024, 1024, 7, 1024, 1024])?;
        let scores = model
            .runtime
            .mem_f32(&[0.5, 3., 1., 5., f32::MIN, 6., 1., 9., -2., 4.])?;
        let selected = model
            .execute(
                "omnivoice.select",
                vec![codes.into(), scores.into(), 5u64.into(), 2u64.into()],
            )?
            .try_to_u64_vec()?;
        assert_eq!(selected, [1024, 5, 7, 9, 1024]);
        Ok(())
    }
}

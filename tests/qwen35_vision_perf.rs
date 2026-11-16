//! Opt-in warmed Qwen3.5 encoder benchmark; no production instrumentation.
use anyhow::Result;
use catena_models::{
    catena::InferenceOptions,
    model::{Model, configured_gpu_dialect},
    tokenizer::QwenTokenizer,
    vision::{PreparedMultimodalPrompt, VisionModel, image::ImagePatches},
};
use serde_json::json;
use std::{path::PathBuf, time::Instant};
// Optional external-profiler ranges; never used by production inference.
struct Trace(Option<libloading::Library>);
impl Trace {
    fn new() -> Result<Self> {
        Ok(Self(if std::env::var_os("QWEN35_PERF_MARKERS").is_some() {
            Some(unsafe { libloading::Library::new("/opt/rocm/lib/librocprofiler-sdk-roctx.so")? })
        } else {
            None
        }))
    }
    fn begin(&self, label: &str) -> Result<()> {
        if let Some(lib) = &self.0 {
            let label = std::ffi::CString::new(label)?;
            unsafe {
                lib.get::<unsafe extern "C" fn(*const std::ffi::c_char) -> i32>(
                    b"roctxRangePushA\0",
                )?(label.as_ptr());
            }
        }
        Ok(())
    }
    fn end(&self) -> Result<()> {
        if let Some(lib) = &self.0 {
            unsafe {
                lib.get::<unsafe extern "C" fn() -> i32>(b"roctxRangePop\0")?();
            }
        }
        Ok(())
    }
}
fn cached(name: &str) -> Result<PathBuf> {
    if name == "mmproj-BF16.gguf" {
        if let Some(path) = std::env::var_os("QWEN35_PERF_PROJECTOR") {
            return Ok(path.into());
        }
    }
    let size = std::env::var("QWEN35_PERF_SIZE").unwrap_or_else(|_| "0.8B".into());
    let root = PathBuf::from(std::env::var_os("HOME").unwrap()).join(format!(
        ".cache/huggingface/hub/models--unsloth--Qwen3.5-{size}-GGUF/snapshots"
    ));
    for entry in std::fs::read_dir(root)? {
        let path = entry?.path().join(name);
        if path.exists() {
            return Ok(path);
        }
    }
    anyhow::bail!("missing {name}")
}
#[test]
#[ignore = "requires HIP/CUDA and cached Qwen3.5 BF16 projector"]
fn warmed_encoder() -> Result<()> {
    let dir = PathBuf::from(
        std::env::var("QWEN35_PERF_OUTPUT").unwrap_or_else(|_| "target/qwen35-vision-perf".into()),
    );
    std::fs::create_dir_all(&dir)?;
    let runs: usize = std::env::var("QWEN35_PERF_RUNS")
        .unwrap_or_else(|_| "3".into())
        .parse()?;
    anyhow::ensure!(runs > 0);
    let vision = VisionModel::load_gguf(
        &cached("mmproj-BF16.gguf")?,
        configured_gpu_dialect()?,
        None,
    )?;
    let mut results = Vec::new();
    let trace = Trace::new()?;
    for (w, h) in [(256, 256), (320, 256), (257, 193), (1024, 1024)] {
        if std::env::var_os("QWEN35_PERF_QUICK").is_some() && w != 256 {
            continue;
        }
        let rgb = (0..w * h)
            .flat_map(|i| {
                [
                    (i % w * 3 + i / w * 5) as u8,
                    (i % w * 7 + i / w * 2) as u8,
                    (i % w + i / w * 11) as u8,
                ]
            })
            .collect::<Vec<_>>();
        let t = Instant::now();
        let patches = ImagePatches::from_rgb(&rgb, w, h, vision.config())?;
        let preprocessing_ms = t.elapsed().as_secs_f64() * 1000.;
        if let Ok(prefix) = std::env::var("QWEN35_VISION_REFERENCE") {
            let path = PathBuf::from(format!("{prefix}-{w}x{h}-pixels.f32"));
            if path.exists() {
                let bytes = std::fs::read(path)?;
                let reference = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect::<Vec<_>>();
                anyhow::ensure!(
                    patches.pixels == reference,
                    "preprocessed pixels differ from reference"
                );
            }
        }
        let image = vision.encode_patches(&patches)?;
        let values = image.to_f32_vec()?;
        anyhow::ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite vision embedding"
        );
        std::fs::write(
            dir.join(format!("vision-{w}x{h}.f32")),
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        if let Ok(prefix) = std::env::var("QWEN35_VISION_REFERENCE") {
            let reference = PathBuf::from(format!("{prefix}-{w}x{h}.f32"));
            if reference.exists() {
                let bytes = std::fs::read(reference)?;
                let expected = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect::<Vec<_>>();
                let actual = image.to_f32_vec()?;
                anyhow::ensure!(
                    actual.len() == expected.len() && !expected.is_empty(),
                    "reference shape mismatch"
                );
                let error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| ((*a as f64) - (*b as f64)).powi(2))
                    .sum::<f64>();
                let signal = expected.iter().map(|b| (*b as f64).powi(2)).sum::<f64>();
                let relative = (error / signal).sqrt();
                eprintln!("vision {w}x{h} reference relative RMS: {relative}");
                anyhow::ensure!(
                    relative.is_finite() && relative < 0.10,
                    "vision reference mismatch: {relative}"
                );
            }
        }
        let mut samples = Vec::new();
        for _ in 0..runs {
            trace.begin(&format!("vision-{w}x{h}"))?;
            let t = Instant::now();
            std::hint::black_box(vision.encode_patches(&patches)?);
            samples.push(t.elapsed().as_secs_f64() * 1000.);
            trace.end()?;
        }
        eprintln!("vision {w}x{h}: {samples:?} ms");
        results.push(json!({"kind":"vision","width":w,"height":h,"tokens":image.token_count(),"preprocessing_ms":preprocessing_ms,"ms":samples}));
    }
    if std::env::var_os("QWEN35_PERF_END_TO_END").is_some() {
        let size = std::env::var("QWEN35_PERF_SIZE").unwrap_or_else(|_| "0.8B".into());
        let path = cached(&format!("Qwen3.5-{size}-BF16.gguf"))?;
        let model = Model::load_gguf_with_multimodal_options_and_dialect(
            &path,
            InferenceOptions::default().with_wmma_prefill(true),
            configured_gpu_dialect()?,
        )?;
        let tokenizer = QwenTokenizer::from_gguf(&path)?.unwrap();
        for count in [1, 2] {
            let mut samples = Vec::new();
            for run in 0..=runs {
                let t = Instant::now();
                let mut images = Vec::new();
                for color in [[255u8, 0, 0], [0, 0, 255]].into_iter().take(count) {
                    let patches = ImagePatches::from_rgb(
                        &color.repeat(256 * 256),
                        256,
                        256,
                        vision.config(),
                    )?;
                    images.push(vision.encode_patches(&patches)?);
                }
                let vision_ms = t.elapsed().as_secs_f64() * 1000.;
                let prepared =
                    PreparedMultimodalPrompt::new(&tokenizer, "Describe the colors.", images)?;
                let mut session = model.session(prepared.token_count() + 4)?;
                let prefill = Instant::now();
                let mut token = session.prefill_multimodal_next_token(&prepared)?;
                let prefill_ms = prefill.elapsed().as_secs_f64() * 1000.;
                let decode = Instant::now();
                let mut tokens = vec![token];
                for _ in 0..2 {
                    token = session.decode_next_token(token)?;
                    tokens.push(token);
                }
                let decode_ms = decode.elapsed().as_secs_f64() * 1000.;
                let sample = json!({"vision_ms":vision_ms,"prefill_ms":prefill_ms,"decode_ms":decode_ms,"total_ms":t.elapsed().as_secs_f64()*1000.,"prompt_tokens":prepared.token_count(),"tokens":tokens});
                eprintln!("{count} image(s): {sample}");
                if run > 0 {
                    samples.push(sample);
                }
            }
            results.push(json!({"kind":"end_to_end","images":count,"samples":samples}));
        }
    }
    std::fs::write(
        dir.join("timings.json"),
        serde_json::to_vec_pretty(&results)?,
    )?;
    Ok(())
}

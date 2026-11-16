//! Opt-in, warmed first-use inference benchmarks. No production runtime hooks.
use anyhow::Result;
use base64::{Engine, engine::general_purpose::STANDARD};
use catena_models::{
    catena::InferenceOptions,
    model::configured_gpu_dialect,
    systemone::{Request, Runner},
    vision::{VisionModel, image::ImagePatches},
};
use serde_json::{Value, json};
use std::{path::PathBuf, time::Instant};
// Optional external-profiler ranges; never used by production inference.
struct Trace(Option<libloading::Library>);
impl Trace {
    fn new() -> Result<Self> {
        Ok(Self(if std::env::var_os("LFM3_PERF_MARKERS").is_some() {
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
    let root = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/huggingface")
        });
    for entry in std::fs::read_dir(root.join("hub/models--LiquidAI--d1-2B-GGUF/snapshots"))? {
        let p = entry?.path().join(name);
        if p.exists() {
            return Ok(p);
        }
    }
    anyhow::bail!("missing cached {name}")
}
fn questions() -> Value {
    json!({"refund":{"type":"noul","instructions":"Is the customer requesting a refund?"},"team":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Charges and refunds","technical":"Technical problems","sales":"New purchases"}},"urgency":{"type":"score","instructions":"How urgent is this?","criteria":["Can wait","Today","Immediate"]}})
}
#[test]
#[ignore = "requires a HIP/CUDA GPU and cached d1 BF16 model/projector"]
fn warmed_inference() -> Result<()> {
    let dir = PathBuf::from(
        std::env::var("LFM3_PERF_OUTPUT").unwrap_or_else(|_| "target/lfm3-perf".into()),
    );
    std::fs::create_dir_all(&dir)?;
    let runs: usize = std::env::var("LFM3_PERF_RUNS")
        .ok()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(3);
    anyhow::ensure!(runs > 0, "runs must be positive");
    let mut results = Vec::new();
    let trace = Trace::new()?;
    let quick = std::env::var_os("LFM3_PERF_QUICK").is_some();
    let projector = cached("mmproj-d1-2B-BF16.gguf")?;
    let vision = VisionModel::load_gguf(&projector, configured_gpu_dialect()?, None)?;
    for (w, h) in [(256, 256), (320, 256), (1024, 1024)] {
        if quick && w != 256 {
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
        let patches = ImagePatches::from_rgb(&rgb, w, h, vision.config())?;
        let image = vision.encode_patches(&patches)?;
        let values = image.to_f32_vec()?;
        std::fs::write(
            dir.join(format!("vision-{w}x{h}.f32")),
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let mut samples = Vec::new();
        for _ in 0..runs {
            trace.begin(&format!("vision-{w}x{h}"))?;
            let t = Instant::now();
            let image = vision.encode_patches(&patches)?;
            trace.end()?;
            samples.push(t.elapsed().as_secs_f64() * 1000.);
            std::hint::black_box(image);
        }
        eprintln!("vision {w}x{h}: {samples:?} ms");
        results.push(
            json!({"kind":"vision","width":w,"height":h,"tokens":image.token_count(),"ms":samples}),
        );
    }
    drop(vision);
    let model = cached("d1-2B-BF16.gguf")?;
    let runner = Runner::load(
        &model,
        Some(&projector),
        InferenceOptions::default().with_wmma_prefill(true),
        8192,
        "d1".into(),
    )?;
    let mut requests = Vec::new();
    for repeats in [1, 125, 500] {
        requests.push(json!({"state":format!("{} Please refund the duplicate charge.","Normal account activity. ".repeat(repeats)),"questions":questions()}));
    }
    let mut png = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut png, 256, 256);
        enc.set_color(png::ColorType::Rgb);
        enc.write_header()?
            .write_image_data(&[255, 0, 0].repeat(256 * 256))?;
    }
    let image = format!("data:image/png;base64,{}", STANDARD.encode(png));
    let q = json!({"color":{"type":"choice","instructions":"What color is the picture?","criteria":{"red":"Red","blue":"Blue","green":"Green"}}});
    requests.push(json!({"state":null,"images":[image.clone()],"questions":q}));
    requests.push(json!({"state":null,"images":[image],"questions":{"color":q["color"],"red":{"type":"noul","instructions":"Is the picture red?"},"brightness":{"type":"score","instructions":"How bright is the picture?","criteria":["Dark","Medium","Bright"]}}}));
    let mut responses = Vec::new();
    for (i, body) in requests.iter().enumerate() {
        if quick && i != 1 {
            continue;
        }
        let request = Request::parse(body.clone())?;
        let response = runner.run(&request)?;
        responses.push(response.clone());
        let mut samples = Vec::new();
        for _ in 0..runs {
            trace.begin(&format!("decision-{i}"))?;
            let t = Instant::now();
            std::hint::black_box(runner.run(&request)?);
            trace.end()?;
            samples.push(t.elapsed().as_secs_f64() * 1000.);
        }
        eprintln!("decision {i}: {samples:?} ms, {}", response["usage"]);
        results.push(json!({"kind":"decision","case":i,"usage":response["usage"],"ms":samples}));
    }
    for (name, value) in [
        ("timings.json", json!(results)),
        ("requests.json", json!(requests)),
        ("responses.json", json!(responses)),
    ] {
        std::fs::write(dir.join(name), serde_json::to_vec_pretty(&value)?)?;
    }
    Ok(())
}

use anyhow::{Result, ensure};
use catena_lang::runtime::{GpuDialect, Runtime, Value};
use catena_models::{
    catena::{compile_sources, core_sources, runtime_sources},
    tts,
    weights::{Weights, bf16},
};
use std::{collections::BTreeMap, path::Path};

fn length_source() -> Vec<String> {
    let model = tts::graph::SOURCE
        .replace("tts.", "stdlib.tts.")
        .replace("stdlib.stdlib.tts.", "stdlib.tts.");
    [include_str!("../src/catena/memory.hex"), include_str!("../src/catena/numeric.hex"), include_str!("../src/catena/resources.hex"), include_str!("../src/catena/cooperative.hex"), &model, r#"
(def program check-length : {((cap.ref mem) val)} -> (u64 val) =
 ([header.] {[.header] (const.u64.0x0000000000000000)} stdlib.qwen3.memory.ref-read-host-f32 [frames.] [.frames] stdlib.tts.checked-length))
(def program read-u64 : {((cap.ref mem) val)} -> (u64 val) =
 ([header.] {[.header] (const.u64.0x0000000000000000)} stdlib.qwen3.memory.ref-read-host-u64))
(def program dynamic-allocation : {((cap.ref mem) val)} -> ((cap.own mem) val) =
 ([header.] {[.header] (const.u64.0x0000000000000000)} stdlib.qwen3.memory.ref-read-host-u64 [count.] [.count] stdlib.qwen3.memory.alloc-f32))
"#].into_iter().map(str::to_owned).collect()
}

#[test]
fn length_and_readback_compile() -> Result<()> {
    let source = length_source();
    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let module = compile_sources(
            core_sources()
                .map(|s| -> &str { s })
                .chain(source.iter().map(String::as_str)),
            dialect,
        )?;
        ensure!(
            module.source.contains("catena_host_read"),
            "missing host read"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires HIP GPU"]
fn length_and_dynamic_allocation() -> Result<()> {
    let source = length_source();
    let mut rt = Runtime::new(GpuDialect::Hip)?;
    let artifact = rt.load(compile_sources(
        core_sources()
            .map(|s| -> &str { s })
            .chain(source.iter().map(String::as_str)),
        GpuDialect::Hip,
    )?)?;
    for (value, expected) in [
        (1., 1),
        (1875., 1875),
        (0., 0),
        (-1., 0),
        (1.5, 0),
        (1876., 0),
        (f32::NAN, 0),
        (f32::INFINITY, 0),
        (f32::NEG_INFINITY, 0),
    ] {
        let header = rt.mem_f32(&[value])?;
        let [Value::U64(actual)] = artifact.exec("check-length", [header.as_ref().into()])? else {
            anyhow::bail!("wrong length ABI")
        };
        ensure!(
            actual == expected,
            "{value}: got {actual}, expected {expected}"
        );
    }
    // Keep all 64 bits, including values that cannot round-trip through F32.
    let header = rt.mem_u64(&[0xfedc_ba98_7654_3210])?;
    let [Value::U64(actual)] = artifact.exec("read-u64", [header.as_ref().into()])? else {
        anyhow::bail!("wrong readback ABI")
    };
    ensure!(actual == 0xfedc_ba98_7654_3210, "truncated U64 readback");
    let header = rt.mem_u64(&[17])?;
    let [Value::MemOwn(output)] = artifact.exec("dynamic-allocation", [header.as_ref().into()])?
    else {
        anyhow::bail!("wrong allocation ABI")
    };
    ensure!(
        output.try_to_f32_vec()?.len() == 17,
        "wrong dynamic allocation size"
    );
    Ok(())
}

fn assert_dump(dir: &Path, name: &str, values: &[f32]) -> Result<()> {
    let actual: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    ensure!(
        actual == std::fs::read(dir.join(format!("{name}.f32")))?,
        "baseline differs: {name}"
    );
    Ok(())
}

#[test]
#[ignore = "requires HIP GPU, cached MMS weights, and /tmp/mms-single-baseline"]
fn exported_single_call_matches_baseline() -> Result<()> {
    let dir = tts::snapshot("facebook/mms-tts-eng")?;
    let source = tts::graph::bundle(&tts::Config::read(&dir)?)?;
    let module = compile_sources(runtime_sources(&source), GpuDialect::Hip)?;
    let mut rt = Runtime::new(GpuDialect::Hip)?;
    let artifact = rt.load(module)?;
    let mut checkpoint = Weights::open(&dir.join("model.safetensors"))?;
    let bindings = tts::bindings();
    let ordinary = bindings
        .iter()
        .filter(|(_, b)| !b.normalized && !b.transpose)
        .map(|(n, _)| n.clone())
        .collect::<Vec<_>>();
    checkpoint.prepare(&rt, &ordinary, false)?;
    let mut prepared = BTreeMap::new();
    for (name, binding) in &bindings {
        if binding.normalized || binding.transpose {
            prepared.insert(
                name.clone(),
                rt.mem_u16(&bf16(&tts::materialize(&checkpoint, name, binding)?))?,
            );
        }
    }
    let ids = tts::tokenizer::Tokenizer::read(&dir)?.encode("Hi.")?;
    let tokens = rt.mem_u64(&ids)?;
    let args = |rate: f32| -> Result<Vec<Value<'_>>> {
        let mut values = vec![
            tokens.as_ref().into(),
            (ids.len() as u64).into(),
            0u32.into(),
            0u32.into(),
            0.8f32.into(),
            0.667f32.into(),
            rate.into(),
        ];
        for (name, _) in &bindings {
            values.push(if let Some(w) = prepared.get(name) {
                w.as_ref().into()
            } else {
                checkpoint.gpu_ref(name)?.into()
            });
        }
        Ok(values)
    };
    let baseline = Path::new("/tmp/mms-single-baseline");
    let mut values = artifact
        .exec_values("tts.synthesize-debug", args(1.)?)?
        .into_iter();
    let Some(Value::MemOwn(waveform)) = values.next() else {
        anyhow::bail!("missing waveform")
    };
    ensure!(
        matches!(values.next(), Some(Value::U64(0))),
        "synthesis failed"
    );
    let wave = waveform.try_to_f32_vec()?;
    assert_dump(baseline, "waveform", &wave)?;
    for (name, is_bf16) in [
        ("durations", false),
        ("stats", true),
        ("duration_noise", false),
        ("latent_noise", false),
        ("flow_output", true),
    ] {
        let Some(Value::MemOwn(mem)) = values.next() else {
            anyhow::bail!("missing capture {name}")
        };
        let data = if is_bf16 {
            mem.try_to_u16_vec()?
                .into_iter()
                .map(|v| f32::from_bits((v as u32) << 16))
                .collect()
        } else {
            mem.try_to_f32_vec()?
        };
        assert_dump(baseline, name, &data)?;
    }
    ensure!(values.next().is_none(), "extra debug outputs");
    let mut values = artifact
        .exec_values("tts.synthesize", args(1.)?)?
        .into_iter();
    let Some(Value::MemOwn(mem)) = values.next() else {
        anyhow::bail!("missing waveform")
    };
    ensure!(
        matches!(values.next(), Some(Value::U64(0))),
        "normal synthesis failed"
    );
    ensure!(
        mem.try_to_f32_vec()? == wave,
        "normal/debug waveforms differ"
    );
    // Direct artifact caller supplies no intermediate buffers or frame count.
    // An extremely slow rate produces the existing invalid-duration sentinel.
    let mut invalid = artifact
        .exec_values("tts.synthesize-debug", args(1e-20)?)?
        .into_iter();
    let Some(Value::MemOwn(mem)) = invalid.next() else {
        anyhow::bail!("missing failure waveform")
    };
    ensure!(
        matches!(invalid.next(), Some(Value::U64(1))),
        "invalid duration accepted"
    );
    ensure!(mem.try_to_f32_vec()?.is_empty(), "failure produced audio");
    let captures = invalid.collect::<Vec<_>>();
    for value in &captures[3..] {
        let Value::MemOwn(mem) = value else {
            anyhow::bail!("bad failure capture")
        };
        // Both absent decoder captures have zero bytes, regardless of dtype.
        ensure!(
            mem.try_to_u16_vec()?.is_empty(),
            "decoder capture allocated on failure"
        );
    }
    Ok(())
}

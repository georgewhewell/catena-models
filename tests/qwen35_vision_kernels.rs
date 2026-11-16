//! Qwen-specific fusions retain F32 LayerNorm and axial RoPE arithmetic.
use anyhow::Result;
use catena_lang::runtime::{Runtime, Value};
use catena_models::catena::{compile_sources, core_sources};
#[allow(dead_code)]
#[path = "support/qwen_tiles.rs"]
mod tiles;
fn rounded(x: f32) -> f32 {
    catena_models::weights::bf16_to_f32(catena_models::weights::f32_to_bf16(x))
}
fn module() -> Result<catena_lang::runtime::RuntimeModule> {
    let entries = r#"
(def program norm : {((cap.own mem) val) ((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) (u64 val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val)} =
 ([x residual weight bias rows cols.]
 ({[.x] [.residual] [.weight] [.bias] [.rows] [.cols] (const.u32.0x358637BD u32.bitcast-f32)} stdlib.vision.qwen.add-ln [pair.])
 ([.pair] *.elim [sum norm.])
 ([.norm] meta.master.complete-own [done phase.])
 ({[.x] [.phase]} meta.master.after-own [input.])
 ({[.residual] [.phase]} meta.master.after-own [res.])
 {[.input] [.res] [.sum] [.done]} ))
(def program qkv : {((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) (u64 val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val)} =
 ([x positions frequencies count cols.]
 ({[.x] [.positions] [.frequencies] [.count] [.cols]} stdlib.vision.qwen.qkv [pair.])
 ([.pair] *.elim [q tail.]) ([.tail] *.elim [k v.])
 ([.v] meta.master.complete-own [done phase.])
 ({[.x] [.phase]} meta.master.after-own [input.])
 {[.input] [.q] [.k] [.done]} ))
"#;
    let start = entries.find("(def program qkv :").unwrap();
    let qkv72 = entries[start..]
        .replace("program qkv :", "program qkv72 :")
        .replace(
            "stdlib.vision.qwen.qkv [pair.]",
            "stdlib.vision.qwen.qkv-d72 [pair.]",
        );
    let entries = format!("{entries}\n{qkv72}");
    let module = compile_sources(
        core_sources().map(|s| -> &str { s }).chain([
            include_str!("../src/catena/memory.hex"),
            include_str!("../src/catena/numeric.hex"),
            include_str!("../src/catena/resources.hex"),
            include_str!("../src/catena/cooperative.hex"),
            include_str!("../src/catena/qwen35_vision_fused.hex"),
            &entries,
        ]),
        tiles::dialect(),
    )?;
    assert!(!module.source.contains("fdot2"));
    Ok(module)
}
#[test]
fn graph_compiles() -> Result<()> {
    let m = module()?;
    if tiles::dialect() == catena_lang::runtime::GpuDialect::Cuda {
        let dir = std::env::temp_dir().join(format!("qwen35-vision-fused-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("kernel.cu");
        std::fs::write(&path, m.source)?;
        let result = std::process::Command::new("nvcc")
            .args(["-std=c++17", "-arch=sm_80", "--fmad=false", "--ptx"])
            .arg(path)
            .arg("-o")
            .arg(dir.join("kernel.ptx"))
            .output()?;
        anyhow::ensure!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    Ok(())
}
#[test]
#[ignore = "requires HIP/CUDA GPU"]
fn layernorm_and_qkv_match_cpu() -> Result<()> {
    let mut rt = Runtime::new(tiles::dialect())?;
    let artifact = rt.load(module()?)?;
    for cols in [768usize, 1024, 1152] {
        let rows = 3;
        let x = (0..rows * cols)
            .map(|i| ((i * 7 % 47) as f32 - 21.) / 17.)
            .collect::<Vec<_>>();
        let residual = x.iter().map(|x| 0.1 + x * 0.3).collect::<Vec<_>>();
        let weight = (0..cols)
            .map(|i| 0.7 + (i % 5) as f32 / 10.)
            .collect::<Vec<_>>();
        let bias = (0..cols).map(|i| (i % 7) as f32 / 100.).collect::<Vec<_>>();
        let wm = rt.mem_f32(&weight)?;
        let bm = rt.mem_f32(&bias)?;
        let [_, _, Value::MemOwn(sum), Value::MemOwn(norm)] = artifact.exec(
            "norm",
            [
                rt.mem_f32(&x)?.into(),
                rt.mem_f32(&residual)?.into(),
                wm.as_ref().into(),
                bm.as_ref().into(),
                (rows as u64).into(),
                (cols as u64).into(),
            ],
        )?
        else {
            anyhow::bail!("norm ABI")
        };
        let sum = sum.try_to_f32_vec()?;
        let norm = norm.try_to_f32_vec()?;
        for r in 0..rows {
            let values = (0..cols)
                .map(|j| rounded(x[r * cols + j] + residual[r * cols + j]))
                .collect::<Vec<_>>();
            let mean = values.iter().sum::<f32>() / cols as f32;
            let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / cols as f32;
            for j in 0..cols {
                assert_eq!(sum[r * cols + j], values[j]);
                let expected =
                    rounded((values[j] - mean) / (variance + 1e-6).sqrt() * weight[j] + bias[j]);
                // Parallel F32 reductions can straddle a BF16 rounding boundary.
                let bits = catena_models::weights::f32_to_bf16(expected.abs());
                let ulp =
                    (catena_models::weights::bf16_to_f32(bits + 1) - expected.abs()).max(1e-6);
                assert!(
                    (norm[r * cols + j] - expected).abs() <= ulp,
                    "{cols}/{r}/{j}: {} vs {expected}",
                    norm[r * cols + j]
                );
            }
        }
        let x = (0..rows * cols * 3)
            .map(|i| ((i * 13 % 73) as f32 - 36.) / 43.)
            .collect::<Vec<_>>();
        let positions = [0u64, 0, 2, 3, 7, 1];
        let pm = rt.mem_u64(&positions)?;
        let dim = if cols == 1152 { 72 } else { 64 };
        let quarter = dim / 4;
        let frequencies = (0..quarter)
            .map(|i| 10000f32.powf(-(i as f32) / quarter as f32))
            .collect::<Vec<_>>();
        let fm = rt.mem_f32(&frequencies)?;
        let [_, Value::MemOwn(q), Value::MemOwn(k), Value::MemOwn(v)] = artifact.exec(
            if dim == 72 { "qkv72" } else { "qkv" },
            [
                rt.mem_f32(&x)?.into(),
                pm.as_ref().into(),
                fm.as_ref().into(),
                ((rows * cols) as u64).into(),
                (cols as u64).into(),
            ],
        )?
        else {
            anyhow::bail!("qkv ABI")
        };
        for (part, out) in [q, k, v].iter().enumerate() {
            let out = out.try_to_f32_vec()?;
            for r in 0..rows {
                for j in 0..cols {
                    let value = x[r * cols * 3 + part * cols + j];
                    let expected = if part == 2 {
                        value
                    } else {
                        let channel = j % dim;
                        let pair = channel % (dim / 2);
                        let angle =
                            positions[r * 2 + pair / quarter] as f32 * frequencies[pair % quarter];
                        let other = if channel < dim / 2 {
                            j + dim / 2
                        } else {
                            j - dim / 2
                        };
                        let rotated = x[r * cols * 3 + part * cols + other]
                            * if channel < dim / 2 { -1. } else { 1. };
                        value * angle.cos() + rotated * angle.sin()
                    };
                    assert!(
                        (out[r * cols + j] - rounded(expected)).abs() < 2e-6,
                        "{cols}/{part}/{r}/{j}"
                    );
                }
            }
        }
    }
    Ok(())
}

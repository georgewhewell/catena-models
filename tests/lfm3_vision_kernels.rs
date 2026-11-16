//! Native BF16 operands with F32 vision interfaces and accumulators.
use anyhow::Result;
use catena_lang::runtime::{Runtime, Value};
use catena_models::{
    catena::{compile_sources, core_sources},
    weights::{bf16_to_f32, f32_to_bf16},
};
#[path = "support/qwen_tiles.rs"]
mod tiles;
fn module() -> Result<catena_lang::runtime::RuntimeModule> {
    let mut entries = String::new();
    for (name, kind, size) in [
        ("linear64", "linear", 64),
        ("linear128", "linear", 128),
        ("gate", "gate", 64),
        ("patch", "patch", 64),
        ("gelu", "gelu", 64),
        ("gelu-tanh", "gelu-tanh", 64),
        ("linear-bf16", "linear-bf16", 64),
        ("gelu-bf16", "gelu-bf16", 64),
        ("gelu-tanh-bf16", "gelu-tanh-bf16", 64),
        ("patch-bf16", "patch-bf16", 64),
    ] {
        let extra = match kind {
            "gate" => " [.up] [.bias]",
            "patch" | "patch-bf16" => " [.up]",
            _ => "",
        };
        entries += &format!(
            "(def program {name} : {{((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) (u64 val) (u64 val) (u64 val)}} -> {{((cap.own mem) val) ((cap.own mem) val)}} =\n ([x w bias up rows k n.]\n ({{[.x] [.w] [.bias] [.rows] [.k] [.n]{extra}}} stdlib.vision.wmma.m{size}n64k32.{kind}.linear-or.native [out.])\n ([.out] meta.master.complete-own [complete phase.])\n ({{[.x] [.phase]}} meta.master.after-own [input.])\n {{[.input] [.complete]}}\n ))\n"
        );
    }
    let sources = [
        include_str!("../src/catena/memory.hex"),
        include_str!("../src/catena/numeric.hex"),
        include_str!("../src/catena/resources.hex"),
        include_str!("../src/catena/cooperative.hex"),
        include_str!("../src/catena/vision_wmma.hex"),
        &entries,
    ];
    let module = compile_sources(
        core_sources().map(|s| -> &str { s }).chain(sources),
        tiles::dialect(),
    )?;
    tiles::check_native(&module, tiles::dialect());
    assert!(!module.source.contains("fdot2"));
    assert!(!module.source.contains("catena_bf16_dot"));
    Ok(module)
}
#[test]
fn matrix_core_graph_compiles() -> Result<()> {
    let m = module()?;
    if tiles::dialect() == catena_lang::runtime::GpuDialect::Cuda {
        tiles::check_toolkit(m, "lfm3_vision")?;
    }
    Ok(())
}
fn rounded(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(x))
}
#[test]
#[ignore = "requires HIP/CUDA GPU"]
fn projections_and_fused_epilogues() -> Result<()> {
    let m = module()?;
    let mut rt = Runtime::new(tiles::dialect())?;
    let artifact = rt.load(m)?;
    for (rows, n, k) in [
        (1, 67, 96),
        (17, 67, 96),
        (65, 128, 768),
        (17, 67, 1024),
        (3, 67, 4304),
        (1, 4304, 80),
        (3, 1152, 80),
    ] {
        let x = (0..rows * k)
            .map(|i| ((i * 7 % 31) as f32 - 15.) / 35.)
            .collect::<Vec<_>>();
        let w = (0..n * k)
            .map(|i| ((i * 11 % 37) as f32 - 18.) / 75.)
            .collect::<Vec<_>>();
        let up = w.iter().map(|x| x * 0.3 + 0.013).collect::<Vec<_>>();
        let bias = (0..n).map(|i| (i % 7) as f32 / 100.).collect::<Vec<_>>();
        let wm = rt.mem_f32(&w)?;
        let um = rt.mem_f32(&up)?;
        let bm = rt.mem_f32(&bias)?;
        for name in [
            "linear64",
            "linear128",
            "gate",
            "patch",
            "gelu",
            "gelu-tanh",
            "linear-bf16",
            "patch-bf16",
            "gelu-bf16",
            "gelu-tanh-bf16",
        ] {
            let [Value::MemOwn(_input), Value::MemOwn(out)] = artifact.exec(
                name,
                [
                    rt.mem_f32(&x)?.into(),
                    wm.as_ref().into(),
                    bm.as_ref().into(),
                    um.as_ref().into(),
                    (rows as u64).into(),
                    (k as u64).into(),
                    (n as u64).into(),
                ],
            )?
            else {
                anyhow::bail!("projection ABI")
            };
            let actual = out.try_to_f32_vec()?;
            for r in 0..rows {
                for c in 0..n {
                    let g = (0..k)
                        .map(|j| rounded(x[r * k + j]) * rounded(w[c * k + j]))
                        .sum::<f32>()
                        + bias[c];
                    let u = (0..k)
                        .map(|j| rounded(x[r * k + j]) * rounded(up[c * k + j]))
                        .sum::<f32>();
                    let bf16 = name.ends_with("-bf16");
                    let g = if bf16 && !name.starts_with("patch") {
                        rounded(g)
                    } else {
                        g
                    };
                    let mut expected = match name.trim_end_matches("-bf16") {
                        "gate" => g / (1. + (-g).exp()) * (u + bias[c]),
                        "patch" => g + u,
                        "gelu-tanh" => g / (1. + (-1.5957692 * (g + 0.044715 * g * g * g)).exp()),
                        "gelu" => {
                            let z = g / std::f32::consts::SQRT_2;
                            let a = z.abs();
                            let t = 1. / (1. + 0.3275911 * a);
                            let p = (((((1.0614054 * t - 1.4531521) * t) + 1.4214138) * t
                                - 0.28449672)
                                * t
                                + 0.2548296)
                                * t;
                            let e = (1. - p * (-a * a).exp()).copysign(z);
                            0.5 * g * (1. + e)
                        }
                        _ => g,
                    };
                    if bf16 {
                        expected = rounded(expected);
                    }
                    let limit = if bf16 {
                        (bf16_to_f32(f32_to_bf16(expected.abs()) + 1) - expected.abs()).max(1e-5)
                    } else {
                        0.001 + expected.abs() * 0.001
                    };
                    assert!(
                        (actual[r * n + c] - expected).abs() <= limit,
                        "{name} {rows}/{n}/{k} [{r},{c}]: {} vs {expected}",
                        actual[r * n + c]
                    );
                }
            }
        }
    }
    if std::env::var_os("LFM3_MATRIX_BENCH").is_some() {
        for rows in [64, 256, 4096] {
            let (k, n) = (1024usize, 1024usize);
            let wm = rt.mem_f32(&vec![0.125; k * n])?;
            let up = rt.mem_f32(&vec![0.25; k * n])?;
            let bias = rt.mem_f32(&vec![0.; n])?;
            for name in ["linear64", "linear128"] {
                let mut samples = Vec::new();
                for i in 0..4 {
                    let input = rt.mem_f32(&vec![0.25; rows * k])?;
                    let start = std::time::Instant::now();
                    std::hint::black_box(artifact.exec::<7, 2>(
                        name,
                        [
                            input.into(),
                            wm.as_ref().into(),
                            bias.as_ref().into(),
                            up.as_ref().into(),
                            (rows as u64).into(),
                            (k as u64).into(),
                            (n as u64).into(),
                        ],
                    )?);
                    if i > 0 {
                        samples.push(start.elapsed().as_secs_f64() * 1000.);
                    }
                }
                eprintln!("{name} rows={rows} k={k} n={n}: {samples:?} ms");
            }
        }
    }
    Ok(())
}

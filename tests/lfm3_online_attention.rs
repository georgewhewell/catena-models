//! Mask, tail, GQA and precision checks for bounded online-softmax kernels.
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
    for (name, vision, window) in [
        ("vision", true, 0),
        ("vision72", true, 0),
        ("full", false, 0),
        ("slide", false, 256),
    ] {
        let call = if vision {
            format!(
                "{{[.q] [.k] [.v] [.rows] [.rows] [.heads] [.kvheads] [.scale] (const.u64.0x0000000000000000) (const.u64.0x0000000000000000)}} stdlib.vision.online.d{}.run",
                if name == "vision72" { 72 } else { 64 }
            )
        } else {
            format!(
                "{{[.q] [.k] [.k] [.rows] [.rows] [.heads] [.kvheads] [.scale] (const.u64.0x{:016X}) (const.u64.0x{window:016X}) [.gate] (const.u64.0x0000000000000000) (const.u64.0x0000000000000000) [.valuebase]}} stdlib.lfm3.online.text.d128.run",
                usize::from(window > 0)
            )
        };
        entries += &format!(
            "(def program {name} : {{((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) (u64 val) (u64 val) (u64 val) (f32 val)}} -> {{((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val)}} =\n ([q k v gate rows heads kvheads scale.]\n ({{[.rows] [.kvheads]}} stdlib.models.memory.u64.mul [kv.])\n ({{[.kv] (const.u64.0x0000000000000080)}} stdlib.models.memory.u64.mul [valuebase.])\n ({call} [out.])\n ([.out] meta.master.complete-own [complete phase.])\n ({{[.q] [.phase]}} meta.master.after-own [q2.])\n ({{[.k] [.phase]}} meta.master.after-own [k2.])\n ({{[.v] [.phase]}} meta.master.after-own [v2.])\n ({{[.gate] [.phase]}} meta.master.after-own [g2.])\n {{[.q2] [.k2] [.v2] [.g2] [.complete]}}\n ))\n"
        );
    }
    let sources = [
        include_str!("../src/catena/memory.hex"),
        include_str!("../src/catena/numeric.hex"),
        include_str!("../src/catena/resources.hex"),
        include_str!("../src/catena/cooperative.hex"),
        include_str!("../src/catena/lfm3_online_attention.hex"),
        include_str!("../src/catena/vision_online_attention.hex"),
        &entries,
    ];
    let m = compile_sources(
        core_sources().map(|s| -> &str { s }).chain(sources),
        tiles::dialect(),
    )?;
    tiles::check_native(&m, tiles::dialect());
    assert!(!m.source.contains("fdot2"));
    assert!(!m.source.contains("catena_bf16_dot"));
    Ok(m)
}
#[test]
fn online_graph_compiles() -> Result<()> {
    let m = module()?;
    if tiles::dialect() == catena_lang::runtime::GpuDialect::Cuda {
        tiles::check_toolkit(m, "lfm3_online")?;
    }
    Ok(())
}
#[test]
#[ignore = "requires HIP/CUDA GPU"]
fn online_attention_matches_cpu() -> Result<()> {
    let m = module()?;
    let mut rt = Runtime::new(tiles::dialect())?;
    let artifact = rt.load(m)?;
    for rows in [17, 65, 273] {
        for (name, dim, vision, window) in [
            ("vision", 64, true, 0),
            ("vision72", 72, true, 0),
            ("full", 128, false, 0),
            ("slide", 128, false, 256),
        ] {
            let heads = 2;
            let kvheads = 1;
            let scale = 1. / (dim as f32).sqrt();
            let rounded = |v| bf16_to_f32(f32_to_bf16(v));
            let q = (0..rows * heads * dim)
                .map(|i| rounded(((i * 7 % 23) as f32 - 11.) / 17.))
                .collect::<Vec<_>>();
            let k = (0..rows * kvheads * dim)
                .map(|i| rounded(((i * 11 % 31) as f32 - 15.) / 21.))
                .collect::<Vec<_>>();
            let v = (0..rows * kvheads * dim)
                .map(|i| rounded(((i * 13 % 37) as f32 - 18.) / 25.))
                .collect::<Vec<_>>();
            let qm = if vision {
                rt.mem_f32(&q)?
            } else {
                rt.mem_u16(&q.iter().map(|x| f32_to_bf16(*x)).collect::<Vec<_>>())?
            };
            let km = if vision {
                rt.mem_f32(&k)?
            } else {
                rt.mem_u16(
                    &k.iter()
                        .chain(&v)
                        .map(|x| f32_to_bf16(*x))
                        .collect::<Vec<_>>(),
                )?
            };
            let vm = if vision {
                rt.mem_f32(&v)?
            } else {
                rt.mem_u16(&[0])?
            };
            let gm = rt.mem_u16(&vec![f32_to_bf16(0.); rows * heads])?;
            let [
                Value::MemOwn(_q),
                Value::MemOwn(_k),
                Value::MemOwn(_v),
                Value::MemOwn(_g),
                Value::MemOwn(out),
            ] = artifact.exec(
                name,
                [
                    qm.into(),
                    km.into(),
                    vm.into(),
                    gm.into(),
                    (rows as u64).into(),
                    (heads as u64).into(),
                    (kvheads as u64).into(),
                    scale.into(),
                ],
            )?
            else {
                anyhow::bail!("attention ABI")
            };
            let actual = if vision {
                out.try_to_f32_vec()?
            } else {
                out.try_to_u16_vec()?.into_iter().map(bf16_to_f32).collect()
            };
            let mut max_error = 0f32;
            for row in 0..rows {
                for head in 0..heads {
                    let end = if window > 0 { row + 1 } else { rows };
                    let begin = if window > 0 {
                        (row + 1).saturating_sub(window)
                    } else {
                        0
                    };
                    let scores = (begin..end)
                        .map(|key| {
                            (0..dim)
                                .map(|d| {
                                    q[(row * heads + head) * dim + d] as f64
                                        * k[key * dim + d] as f64
                                })
                                .sum::<f64>()
                                * scale as f64
                        })
                        .collect::<Vec<_>>();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let probs = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
                    let sum: f64 = probs.iter().sum();
                    for d in 0..dim {
                        let expected = probs
                            .iter()
                            .enumerate()
                            .map(|(i, p)| p / sum * v[(begin + i) * dim + d] as f64)
                            .sum::<f64>() as f32
                            * if vision { 1. } else { 0.5 };
                        let got = actual[(row * heads + head) * dim + d];
                        assert!(got.is_finite());
                        max_error = max_error.max((got - expected).abs());
                    }
                }
            }
            eprintln!("{name} rows={rows}: maximum error {max_error}");
            assert!(max_error < 0.007);
        }
    }
    Ok(())
}

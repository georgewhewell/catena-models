use catena_lang::runtime::{GpuDialect, RuntimeModule};
use catena_models::catena::{compile_sources, core_sources};
use std::process::Command;

fn module(dialect: GpuDialect) -> anyhow::Result<RuntimeModule> {
    compile_sources(
        core_sources().chain([
            include_str!("../src/catena/memory.hex"),
            include_str!("../src/catena/numeric.hex"),
            "(def program test-exp : (f32 val) -> (f32 val) = (stdlib.models.math.exp))",
            "(def program test-bf16-exp : (f32 val) -> (f32 val) = (f32.to-bf16 stdlib.models.memory.bf16.exp bf16.to-f32))",
        ]),
        dialect,
    )
}

#[test]
fn exponential_uses_hex_on_both_backends() -> anyhow::Result<()> {
    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let module = module(dialect)?;
        assert!(
            module
                .functions
                .iter()
                .all(|function| !function.source_name.starts_with("stdlib.models."))
        );
        assert!(!module.source.contains("expf("));
        // The compiler emits unused SDK helper definitions alongside BF16 support.
        assert_eq!(module.source.matches("catena_bf16_exp(").count(), 1);
    }
    for source in [
        include_str!("../src/catena/qwen35.hex"),
        include_str!("../src/catena/qwen35_fused_delta.hex"),
        include_str!("../src/catena/qwen35_batch_scan.hex"),
        include_str!("../src/catena/qwen35_online.hex"),
        include_str!("../src/catena/memory.hex"),
    ] {
        assert!(!source.contains("stdlib.numeric.exp"));
        assert!(!source.contains("native-exp"));
    }
    Ok(())
}

#[test]
#[ignore = "requires hipcc and a HIP GPU; checks emitted host/device arithmetic"]
fn exponential_accuracy_and_device_bits() -> anyhow::Result<()> {
    let module = module(GpuDialect::Hip)?;
    let symbol = &module
        .functions
        .iter()
        .find(|f| f.source_name == "test-exp")
        .unwrap()
        .symbol;
    let directory = std::env::var_os("CATENA_EXP_TEST_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join(format!("catena-exp-{}", std::process::id())));
    std::fs::create_dir_all(&directory)?;
    // The scalar Hex program can execute unchanged on either host or device.
    let source = module.source.replace(
        "extern \"C\" __host__ void",
        "__host__ __device__ inline void",
    ) + &include_str!("support/portable_exp.cpp").replace("EXP_SYMBOL", symbol);
    let cpp = directory.join("exp.cpp");
    let executable = directory.join("exp");
    std::fs::write(&cpp, source)?;
    let mut build = Command::new(std::env::var_os("HIPCC").unwrap_or_else(|| "hipcc".into()));
    build.args([
        "-O2",
        "-std=c++17",
        "--offload-arch=gfx1151",
        "-ffp-contract=off",
        "-fno-fast-math",
    ]);
    // Optional MPFR oracle; otherwise use the host's long-double exponential.
    if let Some(include) = std::env::var_os("CATENA_EXP_MPFR_INCLUDE") {
        build.args(["-DCATENA_EXP_MPFR", "-I"]).arg(include);
        build.arg("-l:libmpfr.so.6");
    }
    let build = build.arg(&cpp).arg("-o").arg(&executable).output()?;
    anyhow::ensure!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let result = Command::new(&executable).output()?;
    eprint!("{}", String::from_utf8_lossy(&result.stdout));
    anyhow::ensure!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(())
}

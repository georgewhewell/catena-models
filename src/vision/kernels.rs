//! Select Hex helpers for the shared vision composition.
//! This renders a graph; allocation, kernels, scheduling and releases execute in Hex.
use super::{VisionConfig, graph};

pub(super) fn source(config: &VisionConfig, names: &[String]) -> String {
    let graph = graph::source(config, names);
    let mut source = [
        include_str!("../catena/memory.hex"),
        include_str!("../catena/numeric.hex"),
        include_str!("../catena/resources.hex"),
        include_str!("../catena/qwen35_vision.hex"),
        &graph,
    ]
    .join("\n");
    crate::qwen::fused::append_hex(&mut source, include_str!("../catena/cooperative.hex"));
    source.push_str(include_str!("../catena/vision_wmma.hex"));
    source.push_str(include_str!("../catena/vision_online_attention.hex"));
    source.push_str(include_str!("../catena/qwen35_vision_fused.hex"));
    if config.kind == super::VisionKind::Lfm3 {
        source.push_str(include_str!("../catena/lfm3_vision.hex"));
        source.push_str(include_str!("../catena/lfm3_online_attention.hex"));
    }
    source
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires QWEN35_PROJECTOR; compiles actual metadata and optional 4B output width"]
    fn qwen35_graph_compiles() -> anyhow::Result<()> {
        let path = std::path::PathBuf::from(std::env::var("QWEN35_PROJECTOR")?);
        let mut config = VisionConfig::from_gguf(&path)?;
        let mut names = config
            .weight_specs(&path)?
            .into_iter()
            .map(|s| s.gguf_name)
            .collect::<Vec<_>>();
        names.push("__zero".into());
        if std::env::var_os("QWEN35_COMPILE_4B").is_some() {
            anyhow::ensure!(config.hidden_size == 1024 && config.layers == 24);
            config.output_size = 2560;
        }
        let source = source(&config, &names);
        let module = crate::catena::compile_sources(
            crate::catena::runtime_sources(&source),
            crate::model::configured_gpu_dialect()?,
        )?;
        assert!(!module.source.contains("fdot2"));
        assert!(!module.source.contains("catena_bf16_dot"));
        if let Ok(path) = std::env::var("QWEN35_VISION_CODEGEN_OUTPUT") {
            std::fs::write(path, module.source)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires LFM3_VL_PROJECTOR pointing to the cached BF16 projector"]
    fn lfm3_graph_compiles() -> anyhow::Result<()> {
        let path = std::path::PathBuf::from(std::env::var("LFM3_VL_PROJECTOR")?);
        let config = VisionConfig::from_gguf(&path)?;
        anyhow::ensure!(config.kind == super::super::VisionKind::Lfm3);
        let mut names = config
            .weight_specs(&path)?
            .into_iter()
            .map(|s| s.gguf_name)
            .collect::<Vec<_>>();
        names.push("__zero".into());
        let source = source(&config, &names);
        let module = crate::catena::compile_sources(
            crate::catena::runtime_sources(&source),
            crate::model::configured_gpu_dialect()?,
        )?;
        assert!(!module.source.contains("__builtin_amdgcn_fdot2"));
        Ok(())
    }
}

use super::{
    config::{TransformerConfig, read},
    ops::Ops,
};
use crate::weights::Weights;
use anyhow::Result;
use catena_lang::runtime::{MemOwn, MemRef};
use std::path::Path;
/// The checkpoint tensors of `flux.denoise` and their shapes. All are stored as
/// BF16; `graph::transformer` gives the argument order.
pub fn weight_shapes(config: &TransformerConfig) -> Vec<(String, Vec<usize>)> {
    let h = config.hidden_size();
    let ff = config.intermediate_size();
    let mut shapes = Vec::new();
    for (name, shape) in [
        ("x_embedder.weight", vec![h, config.in_channels]),
        (
            "context_embedder.weight",
            vec![h, config.joint_attention_dim],
        ),
        (
            "time_guidance_embed.timestep_embedder.linear_1.weight",
            vec![h, config.timestep_guidance_channels],
        ),
        (
            "time_guidance_embed.timestep_embedder.linear_2.weight",
            vec![h, h],
        ),
        ("double_stream_modulation_img.linear.weight", vec![6 * h, h]),
        ("double_stream_modulation_txt.linear.weight", vec![6 * h, h]),
        ("single_stream_modulation.linear.weight", vec![3 * h, h]),
        ("norm_out.linear.weight", vec![2 * h, h]),
        ("proj_out.weight", vec![config.in_channels, h]),
    ] {
        shapes.push((name.to_owned(), shape));
    }
    for layer in 0..config.num_layers {
        for (i, suffix) in super::graph::DUAL_NAMES.iter().enumerate() {
            let shape = match i {
                4 | 5 | 10 | 11 => vec![config.attention_head_dim],
                12 | 14 => vec![2 * ff, h],
                13 | 15 => vec![h, ff],
                _ => vec![h, h],
            };
            shapes.push((format!("transformer_blocks.{layer}.{suffix}"), shape));
        }
    }
    for layer in 0..config.num_single_layers {
        for (suffix, shape) in [
            ("attn.to_qkv_mlp_proj.weight", vec![3 * h + 2 * ff, h]),
            ("attn.to_out.weight", vec![h, h + ff]),
            ("attn.norm_q.weight", vec![config.attention_head_dim]),
            ("attn.norm_k.weight", vec![config.attention_head_dim]),
        ] {
            shapes.push((format!("single_transformer_blocks.{layer}.{suffix}"), shape));
        }
    }
    shapes
}
pub struct Transformer {
    pub ops: Ops,
    weights: Weights,
    program: super::graph::Program,
}
impl Transformer {
    pub fn load(snapshot: &Path, height: usize, width: usize, steps: usize) -> Result<Self> {
        Self::load_with_debug(snapshot, height, width, steps, false)
    }
    pub fn load_with_debug(
        snapshot: &Path,
        height: usize,
        width: usize,
        steps: usize,
        debug: bool,
    ) -> Result<Self> {
        let config: TransformerConfig = read(snapshot.join("transformer/config.json"))?;
        let mut weights = Weights::open(&super::config::transformer_checkpoint(snapshot))?;
        let program = super::graph::transformer(&config, height, width, steps, debug)?;
        let shapes = super::preparation::ordered(&program.weights, weight_shapes(&config))?;
        let ops = super::preparation::Stage::Denoise.load(&mut weights, shapes, &program.source)?;
        Ok(Self {
            ops,
            weights,
            program,
        })
    }
    pub fn denoise(
        &self,
        noise: MemRef<'_>,
        conditioning: MemRef<'_>,
        dump: Option<&Path>,
    ) -> Result<MemOwn> {
        let mut args = vec![noise.into(), conditioning.into()];
        for name in &self.program.weights {
            args.push(self.weights.gpu_ref(name)?.into());
        }
        super::graph::execute(
            &self.ops.artifact,
            "flux.denoise",
            args,
            &self.program,
            dump,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_shapes_cover_program_arguments() -> Result<()> {
        // Klein 4B and 9B.
        for (num_attention_heads, joint_attention_dim, num_layers, num_single_layers) in
            [(24, 7680, 5, 20), (32, 12288, 8, 24)]
        {
            let config = TransformerConfig {
                attention_head_dim: 128,
                axes_dims_rope: vec![32; 4],
                eps: 1e-6,
                guidance_embeds: false,
                in_channels: 128,
                joint_attention_dim,
                mlp_ratio: 3.,
                num_attention_heads,
                num_layers,
                num_single_layers,
                rope_theta: 2000.,
                timestep_guidance_channels: 256,
            };
            let shapes = weight_shapes(&config);
            let mut names: Vec<_> = shapes.iter().map(|(name, _)| name.clone()).collect();
            let mut arguments = crate::flux::graph::transformer(&config, 16, 16, 1, false)?.weights;
            names.sort();
            arguments.sort();
            assert_eq!(names, arguments);
            let h = config.hidden_size();
            assert!(shapes.contains(&("x_embedder.weight".into(), vec![h, 128])));
            assert!(shapes.contains(&(
                "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight".into(),
                vec![9 * h, h]
            )));
        }
        Ok(())
    }
}

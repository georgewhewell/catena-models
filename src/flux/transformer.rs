use super::{
    config::{TransformerConfig, read},
    ops::Ops,
};
use crate::weights::Weights;
use anyhow::Result;
use catena_lang::runtime::{MemOwn, MemRef};
use std::path::Path;
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
        let h = config.hidden_size();
        let ff = config.intermediate_size();
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
            weights.validate(name, &shape)?;
        }
        for layer in 0..config.num_layers {
            for (i, suffix) in super::graph::DUAL_NAMES.iter().enumerate() {
                let shape = match i {
                    4 | 5 | 10 | 11 => vec![config.attention_head_dim],
                    12 | 14 => vec![2 * ff, h],
                    13 | 15 => vec![h, ff],
                    _ => vec![h, h],
                };
                weights.validate(&format!("transformer_blocks.{layer}.{suffix}"), &shape)?;
            }
        }
        for layer in 0..config.num_single_layers {
            for (suffix, shape) in [
                ("attn.to_qkv_mlp_proj.weight", vec![3 * h + 2 * ff, h]),
                ("attn.to_out.weight", vec![h, h + ff]),
                ("attn.norm_q.weight", vec![config.attention_head_dim]),
                ("attn.norm_k.weight", vec![config.attention_head_dim]),
            ] {
                weights.validate(
                    &format!("single_transformer_blocks.{layer}.{suffix}"),
                    &shape,
                )?;
            }
        }
        let program = super::graph::transformer(&config, height, width, steps, debug)?;
        let ops = Ops::new(Some(&program.source))?;
        weights.prepare(&ops.runtime, &program.weights, false)?;
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

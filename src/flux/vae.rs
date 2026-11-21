use super::{
    config::{VaeConfig, read},
    ops::Ops,
};
use crate::weights::Weights;
use anyhow::Result;
use catena_lang::runtime::MemOwn;
use std::path::Path;
pub struct VaeDecoder {
    pub ops: Ops,
    weights: Weights,
    program: super::graph::Program,
}
impl VaeDecoder {
    pub fn load(snapshot: &Path, height: usize, width: usize) -> Result<Self> {
        Self::load_with_debug(snapshot, height, width, false)
    }
    pub fn load_with_debug(
        snapshot: &Path,
        height: usize,
        width: usize,
        debug: bool,
    ) -> Result<Self> {
        let config: VaeConfig = read(snapshot.join("vae/config.json"))?;
        let mut weights = Weights::open(&snapshot.join("vae/diffusion_pytorch_model.safetensors"))?;
        for (name, shape) in weight_shapes() {
            weights.validate(&name, &shape)?;
        }
        let program = super::graph::vae(&config, height, width, debug)?;
        let ops = Ops::new(Some(&program.source))?;
        weights.prepare(&ops.runtime, &program.weights, true)?;
        Ok(Self {
            ops,
            weights,
            program,
        })
    }
    pub fn decode(&self, latents: &MemOwn, dump: Option<&Path>) -> Result<MemOwn> {
        let mut args = vec![latents.as_ref().into()];
        for name in &self.program.weights {
            args.push(self.weights.gpu_ref(name)?.into());
        }
        super::graph::execute(
            &self.ops.artifact,
            "flux.decode-image",
            args,
            &self.program,
            dump,
        )
    }
}

/// The checkpoint tensors of `flux.decode-image` and their shapes. All are
/// stored as F32; `graph::vae` gives the argument order.
pub fn weight_shapes() -> Vec<(String, Vec<usize>)> {
    let mut w = Vec::new();
    w.push(("bn.running_mean".to_owned(), vec![128]));
    w.push(("bn.running_var".to_owned(), vec![128]));
    for (name, ci, co, ks) in [
        ("post_quant_conv", 32, 32, 1),
        ("decoder.conv_in", 32, 512, 3),
        ("decoder.conv_out", 128, 3, 3),
    ] {
        conv(&mut w, name, ci, co, ks);
    }
    for layer in 0..2 {
        res(
            &mut w,
            &format!("decoder.mid_block.resnets.{layer}"),
            512,
            512,
        );
    }
    let mut ci = 512;
    for (block, co) in [512, 512, 256, 128].into_iter().enumerate() {
        for layer in 0..3 {
            res(
                &mut w,
                &format!("decoder.up_blocks.{block}.resnets.{layer}"),
                ci,
                co,
            );
            ci = co;
        }
        if block < 3 {
            conv(
                &mut w,
                &format!("decoder.up_blocks.{block}.upsamplers.0.conv"),
                co,
                co,
                3,
            );
        }
    }
    w.push(("decoder.conv_norm_out.weight".to_owned(), vec![128]));
    w.push(("decoder.conv_norm_out.bias".to_owned(), vec![128]));
    for suffix in ["group_norm.weight", "group_norm.bias"] {
        w.push((
            format!("decoder.mid_block.attentions.0.{suffix}"),
            vec![512],
        ));
    }
    for suffix in ["to_q", "to_k", "to_v", "to_out.0"] {
        w.push((
            format!("decoder.mid_block.attentions.0.{suffix}.weight"),
            vec![512, 512],
        ));
        w.push((
            format!("decoder.mid_block.attentions.0.{suffix}.bias"),
            vec![512],
        ));
    }
    w
}

type Shapes = Vec<(String, Vec<usize>)>;
fn conv(w: &mut Shapes, name: &str, ci: usize, co: usize, ks: usize) {
    w.push((format!("{name}.weight"), vec![co, ci, ks, ks]));
    w.push((format!("{name}.bias"), vec![co]));
}
fn res(w: &mut Shapes, name: &str, ci: usize, co: usize) {
    for (n, c) in [("norm1", ci), ("norm2", co)] {
        for suffix in ["weight", "bias"] {
            w.push((format!("{name}.{n}.{suffix}"), vec![c]));
        }
    }
    conv(w, &format!("{name}.conv1"), ci, co, 3);
    conv(w, &format!("{name}.conv2"), co, co, 3);
    if ci != co {
        conv(w, &format!("{name}.conv_shortcut"), ci, co, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_shapes_cover_program_arguments() -> Result<()> {
        let config = VaeConfig {
            block_out_channels: vec![128, 256, 512, 512],
            latent_channels: 32,
            norm_num_groups: 32,
            batch_norm_eps: 1e-4,
            patch_size: vec![2, 2],
            force_upcast: true,
            layers_per_block: 2,
        };
        let mut names: Vec<_> = weight_shapes().into_iter().map(|(name, _)| name).collect();
        let mut arguments = crate::flux::graph::vae(&config, 16, 16, false)?.weights;
        names.sort();
        arguments.sort();
        assert_eq!(names, arguments);
        Ok(())
    }
}

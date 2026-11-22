//! The checkpoint-to-inference ABI shared by the standalone loader and job renderers.
use super::{
    graph,
    graph::{OWN, Packing, REF},
};
use crate::{
    qwen::{c, call, header},
    spec::{ModelConfig, WeightDtype, weight_specs},
    weights::{
        preparation::{self, Binding, Storage},
        widen_bf16_source,
    },
};
use anyhow::{Result, ensure};
use catena_lang::runtime::GpuDialect;
use safetensors::Dtype;
use std::fmt::Write;

pub const MODEL: &str = "model.safetensors";
pub const CODEC: &str = "audio_tokenizer/model.safetensors";

pub struct Program<T> {
    pub source: String,
    pub inputs: Vec<T>,
    pub outputs: usize,
    pub weights: Vec<Binding<T>>,
    #[cfg(test)]
    pub(super) model_weights: usize,
}

struct Setup<T> {
    inputs: Vec<T>,
    outputs: Vec<String>,
    body: String,
}

impl<T> Setup<T> {
    fn input(&mut self, asset: T) -> String {
        let name = format!("raw{}", self.inputs.len());
        self.inputs.push(asset);
        name
    }

    fn output(&mut self, name: String) -> Binding<T> {
        let index = self.outputs.len();
        self.outputs.push(name);
        Binding::Prepared(index)
    }

    fn tensor(
        &mut self,
        asset: T,
        dtype: Dtype,
        shape: &[usize],
        packing: Packing,
    ) -> Result<Binding<T>> {
        if matches!(
            (packing, dtype),
            (Packing::F32, Dtype::F32) | (Packing::Bf16, Dtype::BF16)
        ) {
            return Ok(Binding::Asset(asset));
        }
        let input = self.input(asset);
        let count = c(shape.iter().product());
        let out = format!("ready{}", self.outputs.len());
        match packing {
            Packing::TransposedConv => {
                let [ci, co, taps] = *shape else {
                    anyhow::bail!("expected [in, out, taps] convolution")
                };
                let op = if dtype == Dtype::BF16 {
                    "stdlib.weights.permute-bf16-bf16"
                } else {
                    "stdlib.weights.permute-f32-bf16"
                };
                call(
                    &mut self.body,
                    &[
                        &input,
                        &c(ci),
                        &c(taps),
                        &c(taps),
                        &c(co * taps),
                        &c(1),
                        &count,
                    ],
                    op,
                    &out,
                );
            }
            Packing::Bf16 => call(
                &mut self.body,
                &[&input, &count],
                "inference.f32-to-bf16",
                &out,
            ),
            Packing::F32 => call(&mut self.body, &[&input, &count], "vision.widen-bf16", &out),
        }
        Ok(self.output(out))
    }

    fn quantizer(&mut self, parts: Vec<(T, Dtype)>, width: usize) -> Binding<T> {
        let mut borrowed = parts.iter().all(|(_, dtype)| *dtype == Dtype::F32);
        let mut group = Vec::new();
        for (asset, dtype) in parts {
            let input = self.input(asset);
            if borrowed {
                group.push(input);
            } else {
                let out = format!("{input}-f32");
                let (op, count) = if dtype == Dtype::BF16 {
                    ("vision.widen-bf16", width)
                } else {
                    // Preserve all F32 bits without requiring 8-byte alignment.
                    ("qwen3.copy-buffer-16", width * 2)
                };
                call(&mut self.body, &[&input, &c(count)], op, &out);
                group.push(out);
            }
        }
        // Each codebook is one contiguous row. Pair adjacent rows until all eight
        // occupy a single allocation; release the intermediate pairs after each merge.
        let mut width = width;
        while group.len() > 1 {
            let mut next = Vec::new();
            for pair in group.chunks_exact(2) {
                let label = format!("q{}-{}-{}", self.outputs.len(), width, next.len());
                let size = c(2 * width);
                let buffer = format!("{label}-alloc");
                call(
                    &mut self.body,
                    &[&size],
                    "stdlib.models.memory.alloc-f32",
                    &buffer,
                );
                let op = if borrowed {
                    "stdlib.weights.concat-f32-ref"
                } else {
                    "stdlib.weights.concat-f32-own"
                };
                let mut out = format!("{label}-cat");
                call(
                    &mut self.body,
                    &[
                        &pair[0],
                        &pair[1],
                        &c(width),
                        &c(width),
                        &size,
                        &size,
                        &buffer,
                    ],
                    op,
                    &out,
                );
                if !borrowed {
                    for (i, input) in pair.iter().enumerate() {
                        let released = format!("{label}-free{i}");
                        call(
                            &mut self.body,
                            &[input, &out],
                            "stdlib.models.resources.free-f32-before",
                            &released,
                        );
                        out = released;
                    }
                }
                next.push(out);
            }
            group = next;
            width *= 2;
            borrowed = false;
        }
        self.output(group.pop().unwrap())
    }
}

/// Resolve original tensors by filename, name and required shape. The returned
/// setup inputs and inference bindings use exactly the caller's asset type.
pub fn program<T>(
    config: &ModelConfig,
    diagnostics: bool,
    dialect: GpuDialect,
    mut resolve: impl FnMut(&str, &str, &[usize]) -> Result<(T, Dtype)>,
) -> Result<Program<T>> {
    let (mut source, codec) = graph::build(config, diagnostics, dialect);
    let mut setup = Setup {
        inputs: Vec::new(),
        outputs: Vec::new(),
        body: String::new(),
    };
    let mut tensor = |file, name: &str, shape: &[usize]| {
        let (asset, dtype) = resolve(file, name, shape)?;
        ensure!(
            matches!(dtype, Dtype::BF16 | Dtype::F32),
            "{file}/{name}: expected F32 or BF16"
        );
        Ok::<_, anyhow::Error>((asset, dtype))
    };
    let mut bind = |name: &str, shape: &[usize], packing| {
        let (asset, dtype) = tensor(MODEL, name, shape)?;
        setup.tensor(asset, dtype, shape, packing)
    };
    let h = config.hidden_size;
    let mut weights = vec![
        bind(
            "llm.embed_tokens.weight",
            &[config.vocabulary_size, h],
            Packing::Bf16,
        )?,
        bind("audio_embeddings.weight", &[8200, h], Packing::Bf16)?,
        bind("audio_heads.weight", &[8200, h], Packing::Bf16)?,
        bind("llm.norm.weight", &[h], Packing::F32)?,
    ];
    for spec in weight_specs(config)?
        .into_iter()
        .filter(|s| s.name.starts_with("model.layers."))
    {
        let name = spec.name.replacen("model.", "llm.", 1);
        weights.push(bind(
            &name,
            &spec.shape,
            if spec.dtype == WeightDtype::F32 {
                Packing::F32
            } else {
                Packing::Bf16
            },
        )?);
    }
    let model_weights = weights.len();
    ensure!(
        model_weights == 4 + 11 * config.num_layers,
        "invalid Qwen3 layer bindings"
    );
    for (suffix, shape) in [
        ("codebook.embed", &[1024, 64][..]),
        ("project_out.weight", &[1024, 64][..]),
        ("project_out.bias", &[1024][..]),
    ] {
        let parts = (0..8)
            .map(|i| tensor(CODEC, &format!("quantizer.quantizers.{i}.{suffix}"), shape))
            .collect::<Result<_>>()?;
        weights.push(setup.quantizer(parts, shape.iter().product()));
    }
    for t in codec {
        let (asset, dtype) = tensor(CODEC, &t.name, &t.shape)?;
        weights.push(setup.tensor(asset, dtype, &t.shape, t.packing)?);
    }
    for fragment in [
        widen_bf16_source(),
        preparation::permute(Storage::F32, Storage::Bf16),
        preparation::permute(Storage::Bf16, Storage::Bf16),
        preparation::concat_f32(true),
        preparation::concat_f32(false),
    ] {
        crate::qwen::fused::append_hex(&mut source, &fragment);
    }
    {
        crate::qwen::fused::append_hex(&mut source, include_str!("../catena/session_copy.hex"));
    }
    let names: Vec<_> = (0..setup.inputs.len()).map(|i| format!("raw{i}")).collect();
    let inputs: Vec<_> = names.iter().map(|name| (name.as_str(), REF)).collect();
    header(
        &mut source,
        "omnivoice.setup",
        &inputs,
        &format!("{{{}}}", vec![OWN; setup.outputs.len()].join(" ")),
    );
    source.push_str(&setup.body);
    writeln!(source, "  {{[.{}]}}\n ))", setup.outputs.join("] [.")).unwrap();
    Ok(Program {
        source: graph::exports(source),
        inputs: setup.inputs,
        outputs: setup.outputs.len(),
        weights,
        #[cfg(test)]
        model_weights,
    })
}

//! Shared checkpoint binding and GPU preparation for MMS synthesis.
use super::{Config, bindings, graph};
use crate::{
    qwen::{c, call, header},
    weights::{
        Weights,
        preparation::{Binding, Storage, permute, row_norm},
    },
};
use anyhow::{Result, bail, ensure};
use catena_lang::runtime::{Artifact, MemOwn, Runtime, Value};
use safetensors::Dtype;
use std::{collections::BTreeMap, fmt::Write};

pub const SETUP_ENTRY: &str = "tts.prepare-weights";

pub struct Program<T> {
    pub source: String,
    pub inputs: Vec<T>,
    pub outputs: usize,
    pub weights: Vec<Binding<T>>,
}

fn storage(dtype: Dtype) -> Result<(Storage, &'static str)> {
    match dtype {
        Dtype::F32 => Ok((Storage::F32, "f32")),
        Dtype::BF16 => Ok((Storage::Bf16, "bf16")),
        _ => bail!("unsupported MMS checkpoint dtype {dtype:?}"),
    }
}

/// The callback validates the source tensor's shape and returns its original byte offset.
pub fn build<T>(
    config: &Config,
    mut tensor: impl FnMut(&str, &[usize]) -> Result<(T, Dtype, usize)>,
) -> Result<Program<T>> {
    let mut inputs = Vec::new();
    let mut raw = |asset| {
        let name = format!("raw{}", inputs.len());
        inputs.push(asset);
        name
    };
    let mut weights = Vec::new();
    let mut results = Vec::new();
    let mut body = String::new();
    let mut kernels = BTreeMap::new();
    for (name, spec) in bindings() {
        let output = format!("weight{}", results.len());
        if spec.normalized {
            let base = name.strip_suffix(".weight").unwrap();
            let [rows, channels, taps] = spec.shape[..] else {
                bail!("invalid normalized MMS shape for {name}");
            };
            let (v, v_type, _) = tensor(&format!("{base}.weight_v"), &spec.shape)?;
            let (g, g_type, _) = tensor(&format!("{base}.weight_g"), &[rows, 1, 1])?;
            let (v_storage, v_name) = storage(v_type)?;
            let (g_storage, g_name) = storage(g_type)?;
            let entry = format!("stdlib.weights.row-norm-{v_name}-{g_name}");
            kernels
                .entry(entry.clone())
                .or_insert_with(|| row_norm(&entry, v_storage, g_storage));
            call(
                &mut body,
                &[&raw(v), &raw(g), &c(channels * taps), &c(rows)],
                &entry,
                &output,
            );
        } else {
            let (asset, dtype, offset) = tensor(&name, &spec.shape)?;
            let (input, ty) = storage(dtype)?;
            if !spec.transpose && input == Storage::Bf16 && offset.is_multiple_of(4) {
                weights.push(Binding::Asset(asset));
                continue;
            }
            let count = c(spec.shape.iter().product());
            let source = raw(asset);
            if spec.transpose {
                let [ci, co, taps] = spec.shape[..] else {
                    bail!("invalid transposed MMS shape for {name}");
                };
                // Result [co,ci,k] addresses source [ci,co,k].
                let entry = format!("stdlib.weights.permute-{ty}-bf16");
                kernels
                    .entry(entry.clone())
                    .or_insert_with(|| permute(input, Storage::Bf16));
                call(
                    &mut body,
                    &[
                        &source,
                        &c(ci),
                        &c(taps),
                        &c(taps),
                        &c(co * taps),
                        &c(1),
                        &count,
                    ],
                    &entry,
                    &output,
                );
            } else if input == Storage::Bf16 {
                // Re-align the odd-halfword checkpoint ranges for packed inference reads.
                let entry = "stdlib.weights.permute-bf16-bf16";
                kernels
                    .entry(entry.to_owned())
                    .or_insert_with(|| permute(input, input));
                call(
                    &mut body,
                    &[&source, &c(1), &c(1), &c(1), &c(1), &c(1), &count],
                    entry,
                    &output,
                );
            } else {
                call(
                    &mut body,
                    &[&source, &count],
                    "stdlib.inference.f32-to-bf16",
                    &output,
                );
            }
        }
        weights.push(Binding::Prepared(results.len()));
        results.push(output);
    }
    let mut source = graph::bundle(config)?;
    for fragment in kernels.into_values() {
        crate::qwen::fused::append_hex(&mut source, &fragment);
    }
    let names: Vec<_> = (0..inputs.len()).map(|i| format!("raw{i}")).collect();
    let parameters: Vec<_> = names
        .iter()
        .map(|n| (n.as_str(), "((cap.ref mem) val)"))
        .collect();
    let returns = vec!["((cap.own mem) val)"; results.len()].join(" ");
    header(
        &mut source,
        SETUP_ENTRY,
        &parameters,
        &format!("{{{returns}}}"),
    );
    source.push_str(&body);
    writeln!(source, "  {{[.{}]}}\n ))", results.join("] [.")).unwrap();
    Ok(Program {
        source,
        inputs,
        outputs: results.len(),
        weights,
    })
}

impl Program<String> {
    pub fn checkpoint(config: &Config, checkpoint: &Weights) -> Result<Self> {
        build(config, |name, shape| {
            checkpoint.validate(name, shape)?;
            let info = &checkpoint.tensors[name];
            Ok((name.to_owned(), info.dtype, info.byte_offset()))
        })
    }

    pub fn prepare(
        &self,
        checkpoint: &Weights,
        runtime: &Runtime,
        artifact: &Artifact,
    ) -> Result<Vec<MemOwn>> {
        let inputs = self
            .inputs
            .iter()
            .map(|name| checkpoint.upload_raw(runtime, name))
            .collect::<Result<Vec<_>>>()?;
        let mut prepared = artifact
            .exec_values(
                SETUP_ENTRY,
                inputs.iter().map(|m| m.as_ref().into()).collect(),
            )?
            .into_iter()
            .map(|v| match v {
                Value::MemOwn(m) => Ok(Some(m)),
                _ => bail!("invalid MMS setup result"),
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            prepared.len() == self.outputs,
            "wrong MMS setup result count"
        );
        drop(inputs);
        self.weights
            .iter()
            .map(|weight| match weight {
                Binding::Asset(name) => checkpoint.upload_raw(runtime, name),
                Binding::Prepared(i) => Ok(prepared[*i].take().unwrap()),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catena::{compile_sources, runtime_sources};
    use catena_lang::runtime::GpuDialect;

    #[test]
    fn original_checkpoint_setup_compiles_for_both_dialects() -> Result<()> {
        let config = Config(Config::expected());
        // Exercise mixed storage in normalized gains, all four transposes, and
        // both packed-aligned and odd-halfword direct paths.
        let program = build(&config, |name, _| {
            let dtype = if name.ends_with(".weight_g") || name.ends_with(".bias") {
                Dtype::BF16
            } else {
                Dtype::F32
            };
            let offset = if name.starts_with("decoder.") { 2 } else { 4 };
            Ok((name.to_owned(), dtype, offset))
        })?;
        assert_eq!(program.weights.len(), 460);
        assert!(
            program
                .weights
                .iter()
                .any(|b| matches!(b, Binding::Asset(_)))
        );
        assert!(program.inputs.iter().any(|n| n.ends_with(".weight_g")));
        for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
            let module = compile_sources(runtime_sources(&program.source), dialect)?;
            assert!(!module.source.contains("catena_host_read"));
        }
        Ok(())
    }
}

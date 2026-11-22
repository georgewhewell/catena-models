//! Original Whisper weights for the encoder, cross attention, and decoder.
use super::{Config, graph, weight_shapes};
use crate::{
    qwen::{c, call, header},
    weights::preparation::Binding,
};
use anyhow::{Result, ensure};
use safetensors::Dtype;
use std::{collections::BTreeMap, fmt::Write};

pub const ENTRY: &str = "whisper.setup";

pub struct Program<T> {
    pub source: String,
    pub inputs: Vec<T>,
    pub weights: Vec<Binding<T>>,
}

/// Bind the model's graph-order arguments. Only tensors needing a cast are
/// passed to setup. An already-BF16 checkpoint needs no setup call.
pub fn program<T>(
    config: &Config,
    debug: bool,
    mut tensor: impl FnMut(&str, &[usize]) -> Result<(T, Dtype)>,
) -> Result<Program<T>> {
    let mut source = graph::bundle(config, debug)?;
    let shapes = weight_shapes(config)
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let mut inputs = Vec::new();
    let mut weights = Vec::new();
    let mut body = String::new();
    for name in graph::encoder_names(config)
        .into_iter()
        .chain(graph::cross_names(config))
        .chain(graph::decoder_names(config))
    {
        let shape = &shapes[&name];
        let (asset, dtype) = tensor(&name, shape)?;
        if dtype == Dtype::BF16 {
            weights.push(Binding::Asset(asset));
        } else {
            ensure!(
                dtype == Dtype::F32,
                "{name}: expected BF16 or F32, got {dtype:?}"
            );
            let i = inputs.len();
            call(
                &mut body,
                &[&format!("raw{i}"), &c(shape.iter().product())],
                "stdlib.inference.f32-to-bf16",
                &format!("weight{i}"),
            );
            inputs.push(asset);
            weights.push(Binding::Prepared(i));
        }
    }
    if !inputs.is_empty() {
        let names = (0..inputs.len())
            .map(|i| format!("raw{i}"))
            .collect::<Vec<_>>();
        let args = names
            .iter()
            .map(|n| (n.as_str(), "((cap.ref mem) val)"))
            .collect::<Vec<_>>();
        let outputs = (0..inputs.len())
            .map(|i| format!("[.weight{i}]"))
            .collect::<Vec<_>>()
            .join(" ");
        header(
            &mut source,
            ENTRY,
            &args,
            &format!(
                "{{{}}}",
                vec!["((cap.own mem) val)"; inputs.len()].join(" ")
            ),
        );
        source.push_str(&body);
        writeln!(source, "  {{{outputs}}}\n ))")?;
    }
    Ok(Program {
        source,
        inputs,
        weights,
    })
}

pub fn from_checkpoint(
    config: &Config,
    debug: bool,
    weights: &crate::weights::Weights,
) -> Result<Program<String>> {
    program(config, debug, |name, shape| {
        weights.validate(name, shape)?;
        Ok((name.to_owned(), weights.tensors[name].dtype))
    })
}

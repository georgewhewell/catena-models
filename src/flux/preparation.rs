//! Original checkpoint ranges and their inference arguments for a Klein stage.
use super::ops::Ops;
use crate::{
    qwen::{c, call, header},
    weights::{Weights, preparation::Binding},
};
use anyhow::{Context, Result, bail, ensure};
use catena_lang::runtime::Value;
use safetensors::Dtype;
use std::{collections::BTreeMap, fmt::Write};

#[derive(Clone, Copy)]
pub enum Stage {
    Text,
    Denoise,
    Vae,
}

impl Stage {
    pub fn entry(self) -> &'static str {
        match self {
            Self::Text => "flux.setup-text",
            Self::Denoise => "flux.setup-denoise",
            Self::Vae => "flux.setup-vae",
        }
    }

    pub fn setup<T>(
        self,
        shapes: Vec<(String, Vec<usize>)>,
        mut tensor: impl FnMut(&str, &[usize]) -> Result<(Dtype, T)>,
    ) -> Result<Setup<T>> {
        let mut setup = Setup {
            entry: self.entry(),
            source: String::new(),
            inputs: Vec::new(),
            bindings: Vec::with_capacity(shapes.len()),
        };
        let mut body = String::new();
        for (name, shape) in shapes {
            let (dtype, view) = tensor(&name, &shape)?;
            ensure!(
                matches!(dtype, Dtype::F32 | Dtype::BF16),
                "{name}: unsupported dtype {dtype:?}"
            );
            let f32 = match self {
                Self::Text => name.ends_with("norm.weight"),
                Self::Denoise => false,
                Self::Vae => true,
            };
            if (dtype == Dtype::F32) == f32 {
                setup.bindings.push(Binding::Asset(view));
                continue;
            }
            let n = setup.inputs.len();
            let op = if f32 {
                "stdlib.flux.widen-bf16"
            } else {
                "stdlib.flux.to-bf16"
            };
            call(
                &mut body,
                &[&format!("w{n}"), &c(shape.iter().product())],
                op,
                &format!("out{n}"),
            );
            setup.inputs.push(view);
            setup.bindings.push(Binding::Prepared(n));
        }
        let n = setup.inputs.len();
        if n > 0 {
            const OWN: &str = "((cap.own mem) val)";
            let names: Vec<_> = (0..n).map(|i| format!("w{i}")).collect();
            let inputs = names
                .iter()
                .map(|s| (s.as_str(), "((cap.ref mem) val)"))
                .collect::<Vec<_>>();
            let results = (0..n).map(|i| format!("[.out{i}]")).collect::<Vec<_>>();
            let ty = vec![OWN; n].join(" ");
            header(
                &mut setup.source,
                setup.entry,
                &inputs,
                &format!("{{{ty}}}"),
            );
            writeln!(setup.source, "{body}  {{{}}}\n ))", results.join(" ")).unwrap();
        }
        Ok(setup)
    }

    pub(super) fn load(
        self,
        weights: &mut Weights,
        shapes: Vec<(String, Vec<usize>)>,
        source: &str,
    ) -> Result<Ops> {
        let setup = self.setup(shapes, |name, shape| {
            weights.validate(name, shape)?;
            Ok((weights.tensors[name].dtype, name.to_owned()))
        })?;
        let ops = Ops::new(Some(&format!("{source}\n{}", setup.source)))?;
        for name in &setup.inputs {
            weights.prepare_raw(&ops.runtime, name)?;
        }
        if !setup.inputs.is_empty() {
            let inputs = setup
                .inputs
                .iter()
                .map(|n| Ok(weights.gpu_ref(n)?.into()))
                .collect::<Result<Vec<_>>>()?;
            let outputs = ops.artifact.exec_values(setup.entry, inputs)?;
            ensure!(
                outputs.len() == setup.inputs.len(),
                "wrong Flux setup output count"
            );
            let owned = outputs
                .into_iter()
                .map(|v| match v {
                    Value::MemOwn(m) => Ok(m),
                    _ => bail!("expected owned Flux setup result"),
                })
                .collect::<Result<Vec<_>>>()?;
            for (name, value) in setup.inputs.iter().zip(owned) {
                weights.insert_prepared(name, value)?;
            }
        }
        for binding in setup.bindings {
            if let Binding::Asset(name) = binding {
                weights.prepare_raw(&ops.runtime, &name)?;
            }
        }
        Ok(ops)
    }
}

pub struct Setup<T> {
    pub entry: &'static str,
    pub source: String,
    pub inputs: Vec<T>,
    pub bindings: Vec<Binding<T>>,
}

/// Reorder the published checkpoint shapes to match inference argument order.
pub fn ordered(
    names: &[String],
    shapes: Vec<(String, Vec<usize>)>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let mut shapes: BTreeMap<_, _> = shapes.into_iter().collect();
    let ordered = names
        .iter()
        .map(|name| {
            Ok((
                name.clone(),
                shapes
                    .remove(name)
                    .with_context(|| format!("unknown Flux weight {name}"))?,
            ))
        })
        .collect::<Result<_>>()?;
    ensure!(shapes.is_empty(), "Flux graph omitted checkpoint weights");
    Ok(ordered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use catena_lang::codegen::GpuDialect;

    #[test]
    fn klein_setup_uses_only_ranges_that_need_casting() -> Result<()> {
        let shapes = || {
            vec![
                ("layer.norm.weight".into(), vec![4]),
                ("proj.weight".into(), vec![2, 2]),
            ]
        };
        let setup = Stage::Text.setup(shapes(), |name, _| {
            Ok((
                if name.starts_with("proj") {
                    Dtype::F32
                } else {
                    Dtype::BF16
                },
                name.to_owned(),
            ))
        })?;
        assert_eq!(setup.inputs, ["layer.norm.weight", "proj.weight"]);
        assert_eq!(setup.bindings, [Binding::Prepared(0), Binding::Prepared(1)]);
        let single =
            Stage::Vae.setup(vec![("bias".into(), vec![2])], |_, _| Ok((Dtype::BF16, ())))?;
        let passthrough = Stage::Denoise.setup(shapes(), |_, _| Ok((Dtype::BF16, ())))?;
        assert!(passthrough.source.is_empty() && passthrough.inputs.is_empty());
        assert_eq!(
            passthrough.bindings,
            [Binding::Asset(()), Binding::Asset(())]
        );
        assert!(
            Stage::Text
                .setup(shapes(), |_, _| Ok((Dtype::F16, ())))
                .is_err()
        );
        let source = super::super::graph::bundle(&format!("{}\n{}", setup.source, single.source));
        for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
            crate::catena::compile_sources(crate::catena::runtime_sources(&source), dialect)?;
        }
        Ok(())
    }
}

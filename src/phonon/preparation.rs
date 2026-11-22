//! Checkpoint bindings for both the standalone runtime and remote jobs.
use super::{Binding as Spec, Config, LATENT, LATENT_MEAN, LATENT_STD, Packing, bindings};
use crate::{
    qwen::{c, call, header},
    weights::preparation::{self, Binding, Storage},
};
use anyhow::{Result, bail, ensure};
use safetensors::Dtype;
use std::{collections::BTreeMap, fmt::Write};

const REF: &str = "((cap.ref mem) val)";
const OWN: &str = "((cap.own mem) val)";
pub const WEIGHT_ENTRY: &str = "phonon.prepare-weights";
pub const VOICE_ENTRY: &str = "phonon.prepare-voice";

pub struct Program<T> {
    pub source: String,
    pub raw: Vec<T>,
    pub outputs: usize,
    pub setup: Vec<Binding<T>>,
    pub synth: Vec<Binding<T>>,
    /// Standard deviation and mean, in synthesis argument order.
    pub latents: [Binding<T>; 2],
}

struct Builder<T> {
    raw: Vec<T>,
    body: String,
    outputs: Vec<String>,
}

impl<T> Builder<T> {
    fn raw_name(&mut self, asset: T) -> String {
        let name = format!("raw{}", self.raw.len());
        self.raw.push(asset);
        name
    }

    fn weight(
        &mut self,
        spec: &Spec,
        tensor: &mut impl FnMut(bool, &str, &[usize]) -> Result<(T, Dtype, usize)>,
    ) -> Result<Binding<T>> {
        let elements = spec.shape.iter().product::<usize>();
        let mut parts = Vec::new();
        for name in &spec.names {
            let (asset, dtype, offset) = tensor(spec.codec, name, &spec.shape)?;
            ensure!(
                matches!(dtype, Dtype::F32 | Dtype::BF16),
                "unsupported Phonon dtype {dtype:?}"
            );
            if spec.packing.is_none()
                && ((spec.codec && dtype == Dtype::F32)
                    || (!spec.codec && dtype == Dtype::BF16 && offset.is_multiple_of(4)))
            {
                ensure!(
                    spec.names.len() == 1,
                    "unpacked binding must name one tensor"
                );
                return Ok(Binding::Asset(asset));
            }
            let input = self.raw_name(asset);
            let out = format!("{input}-ready");
            if let Some(packing) = &spec.packing {
                let a = spec.shape[0];
                let b = spec.shape[1];
                let taps = spec.shape.get(2).copied().unwrap_or(1);
                let (ci, co, si, so) = match packing {
                    Packing::Transposed => (a, b, b * taps, taps),
                    Packing::Conv | Packing::Qkv => (b, a, taps, b * taps),
                };
                let kernel = match dtype {
                    Dtype::F32 => "stdlib.weights.permute-f32-f32",
                    _ => "stdlib.weights.permute-bf16-f32",
                };
                call(
                    &mut self.body,
                    &[&input, &c(ci), &c(co), &c(1), &c(si), &c(so), &c(elements)],
                    kernel,
                    &out,
                );
            } else {
                let kernel = match (spec.codec, dtype) {
                    (false, Dtype::F32) => "inference.f32-to-bf16",
                    (false, Dtype::BF16) => "qwen3.copy-buffer-16",
                    (true, Dtype::BF16) => "vision.widen-bf16",
                    _ => bail!("unexpected Phonon storage"),
                };
                call(&mut self.body, &[&input, &c(elements)], kernel, &out);
            }
            parts.push(out);
        }
        // Q, K, V are stored as three [out, in] matrices. Join the transposed
        // rows to form one [in, 3*out] projection.
        let mut group = parts.into_iter();
        let mut ready = group
            .next()
            .ok_or_else(|| anyhow::anyhow!("empty Phonon binding"))?;
        for (i, right) in group.enumerate() {
            let label = format!("qkv{}-{i}", self.outputs.len());
            let total = c(elements * (i + 2));
            let buffer = format!("{label}-buffer");
            call(
                &mut self.body,
                &[&total],
                "stdlib.models.memory.alloc-f32",
                &buffer,
            );
            call(
                &mut self.body,
                &[
                    &ready,
                    &right,
                    &c(spec.shape[0] * (i + 1)),
                    &c(spec.shape[0]),
                    &total,
                    &total,
                    &buffer,
                ],
                "stdlib.weights.concat-f32-own",
                &label,
            );
            let left_free = format!("{label}-left");
            call(
                &mut self.body,
                &[&ready, &label],
                "stdlib.models.resources.free-f32-before",
                &left_free,
            );
            ready = format!("{label}-ready");
            call(
                &mut self.body,
                &[&right, &left_free],
                "stdlib.models.resources.free-f32-before",
                &ready,
            );
        }
        self.outputs.push(ready);
        Ok(Binding::Prepared(self.outputs.len() - 1))
    }
}

/// The callback validates the shape in the original shard and supplies its byte offset.
pub fn build<T: Clone>(
    config: &Config,
    voice_dtype: Dtype,
    mut tensor: impl FnMut(bool, &str, &[usize]) -> Result<(T, Dtype, usize)>,
) -> Result<Program<T>> {
    let mut builder = Builder {
        raw: Vec::new(),
        body: String::new(),
        outputs: Vec::new(),
    };
    let mut seen = BTreeMap::new();
    let mut args = |entry| {
        bindings(config, entry)?
            .into_iter()
            .map(|spec| {
                if let Some(binding) = seen.get(&spec.names) {
                    return Ok(Binding::clone(binding));
                }
                let operand = builder.weight(&spec, &mut tensor)?;
                seen.insert(spec.names, operand.clone());
                Ok(operand)
            })
            .collect::<Result<Vec<_>>>()
    };
    let setup = args("setup")?;
    let synth = args("synthesize")?;
    let mut latent = |name: &str| {
        builder.weight(
            &Spec {
                names: vec![name.into()],
                shape: vec![LATENT],
                packing: None,
                codec: false,
            },
            &mut tensor,
        )
    };
    let latents = [latent(LATENT_STD)?, latent(LATENT_MEAN)?];

    let mut source = super::bundle(config);
    for fragment in [
        preparation::permute(Storage::F32, Storage::F32),
        preparation::permute(Storage::Bf16, Storage::F32),
        preparation::concat_f32(false),
        crate::weights::widen_bf16_source(),
        include_str!("../catena/session_copy.hex").to_owned(),
    ] {
        crate::qwen::fused::append_hex(&mut source, &fragment);
    }
    let names: Vec<_> = (0..builder.raw.len()).map(|i| format!("raw{i}")).collect();
    let parameters: Vec<_> = names.iter().map(|n| (n.as_str(), REF)).collect();
    let outputs = builder.outputs.len();
    header(
        &mut source,
        WEIGHT_ENTRY,
        &parameters,
        &format!("{{{}}}", vec![OWN; outputs].join(" ")),
    );
    source.push_str(&builder.body);
    writeln!(source, "  {{[.{}]}}\n ))", builder.outputs.join("] [.")).unwrap();

    // The frame count lives in the argument's byte length, so the same voice
    // prep entry works for every utterance without a scalar request argument.
    let (count, cast) = match voice_dtype {
        Dtype::F32 => (
            "stdlib.models.memory.ref-count-f32",
            "inference.f32-to-bf16",
        ),
        Dtype::BF16 => (
            "stdlib.models.memory.ref-count-bf16",
            "qwen3.copy-buffer-16",
        ),
        _ => bail!("unsupported voice storage {voice_dtype:?}"),
    };
    header(&mut source, VOICE_ENTRY, &[("voice", REF)], OWN);
    call(&mut source, &["voice"], count, "elements");
    call(&mut source, &["voice", "elements"], cast, "ready");
    source.push_str("  [.ready]\n ))\n");
    // Internal GPU helpers must never become externally callable kernels.
    let renames = source
        .split("(def program ")
        .skip(1)
        .filter_map(|def| {
            let name = def.split_whitespace().next()?;
            (!name.starts_with("stdlib.")
                && !matches!(
                    name,
                    WEIGHT_ENTRY | VOICE_ENTRY | "phonon.setup" | "phonon.synthesize"
                ))
            .then(|| (name.to_owned(), format!("stdlib.phonon.shared.{name}")))
        })
        .collect();
    Ok(Program {
        source: crate::catena::rename_programs(&source, &renames),
        raw: builder.raw,
        outputs,
        setup,
        synth,
        latents,
    })
}

//! Hex kernels for materializing checkpoint ranges on the device.
use crate::qwen::{call, header, nested_product, pack, unpack};
use std::fmt::Write;

const REF: &str = "((cap.ref mem) val)";
const OWN: &str = "((cap.own mem) val)";
const U64: &str = "(u64 val)";

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Storage {
    Bf16,
    F32,
}
impl Storage {
    fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
        }
    }
}

fn kernel(family: &str, arg: &str, index: &str, input: Storage, output: Storage) -> String {
    let value = if input == output {
        "[.value]".to_owned()
    } else {
        format!("[.value] {}.to-{}", input.name(), output.name())
    };
    include_str!("../catena/weight_cast.hex")
        .replace("{{FAMILY}}", family)
        .replace("{{ARG}}", arg)
        .replace("{{INDEX}}", index)
        .replace("{{VALUE}}", &value)
        .replace("{{INPUT}}", input.name())
        .replace("{{OUTPUT}}", output.name())
}

fn cast(family: &str, entry: &str, input: Storage, output: Storage) -> String {
    let mut s = kernel(
        family,
        REF,
        " ([.input] [memory.])\n ([.element] [index.])",
        input,
        output,
    );
    header(&mut s, entry, &[("input", REF), ("count", U64)], OWN);
    call(
        &mut s,
        &["input", "count", "count"],
        &format!("{family}.run"),
        "out",
    );
    s.push_str("  [.out]\n ))\n");
    s
}
pub(crate) fn cast_to_bf16_source() -> String {
    cast(
        "stdlib.whisper.convert",
        "inference.f32-to-bf16",
        Storage::F32,
        Storage::Bf16,
    )
}
pub(crate) fn widen_bf16_source() -> String {
    cast(
        "stdlib.vision.widen",
        "vision.widen-bf16",
        Storage::Bf16,
        Storage::F32,
    )
}

/// A contiguous three-dimensional result. Coordinates in the result address
/// the source by its three element strides, supplied by the model setup.
pub fn permute(input: Storage, output: Storage) -> String {
    let entry = format!("stdlib.weights.permute-{}-{}", input.name(), output.name());
    let args = [
        ("memory", REF),
        ("middle", U64),
        ("inner", U64),
        ("s0", U64),
        ("s1", U64),
        ("s2", U64),
    ];
    let names: Vec<_> = args.iter().map(|(name, _)| name.to_string()).collect();
    let types: Vec<_> = args.iter().map(|(_, ty)| ty.to_string()).collect();
    let mut index = String::new();
    unpack(&mut index, "input", &names);
    index.push_str(include_str!("../catena/weight_permute_index.hex"));
    let mut s = kernel(&entry, &nested_product(&types), &index, input, output);
    let mut inputs = args.to_vec();
    inputs.push(("count", U64));
    header(&mut s, &entry, &inputs, OWN);
    pack(&mut s, &names, "view");
    call(
        &mut s,
        &["view", "count", "count"],
        &format!("{entry}.run"),
        "out",
    );
    writeln!(s, "  [.out]\n ))").unwrap();
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catena::{compile_sources, runtime_sources};
    use anyhow::Result;
    use catena_lang::runtime::{GpuDialect, Runtime, RuntimeModule, Value};

    fn permutation_module(dialect: GpuDialect) -> Result<RuntimeModule> {
        let source = [
            include_str!("../catena/memory.hex"),
            &permute(Storage::F32, Storage::Bf16),
            "(def program permute : {((cap.ref mem) val)} -> ((cap.own mem) val) =
             ([input.] ({[.input] (const.u64.0x0000000000000003) (const.u64.0x0000000000000002)
             (const.u64.0x0000000000000002) (const.u64.0x0000000000000004)
             (const.u64.0x0000000000000001) (const.u64.0x000000000000000C)}
             stdlib.weights.permute-f32-bf16)))",
        ]
        .join("\n");
        compile_sources(runtime_sources(&source), dialect)
    }

    #[test]
    fn permutation_compiles_for_both_dialects() -> Result<()> {
        for dialect in [GpuDialect::Cuda, GpuDialect::Hip] {
            permutation_module(dialect)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires GPU"]
    fn permuted_weights_match_reference() -> Result<()> {
        let dialect = crate::model::configured_gpu_dialect()?;
        let mut runtime = Runtime::new(dialect)?;
        let artifact = runtime.load(permutation_module(dialect)?)?;
        let input = runtime.mem_f32(&(0..12).map(|n| n as f32).collect::<Vec<_>>())?;
        let [Value::MemOwn(out)] = artifact.exec("permute", [input.as_ref().into()])? else {
            anyhow::bail!("missing permutation result")
        };
        assert_eq!(
            out.try_to_u16_vec()?,
            crate::weights::bf16(&[0., 1., 4., 5., 8., 9., 2., 3., 6., 7., 10., 11.])
        );
        Ok(())
    }
}

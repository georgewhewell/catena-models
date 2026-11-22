//! Hex kernels for materializing checkpoint ranges on the device.
use crate::qwen::{call, header, nested_product, pack, unpack};
use std::fmt::Write;

/// `stdlib.weights.row-norm(v, gain, columns, rows)` prepares row-major
/// F32 checkpoint vectors as a contiguous row-major BF16 output.
pub const ROW_NORM: &str = include_str!("../catena/weight_norm.hex");

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

/// One inference argument, drawn from the checkpoint or an owned setup result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Binding<T> {
    Asset(T),
    Prepared(usize),
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

fn concat_kernel(
    name: &str,
    left: &str,
    right: &str,
    extent: &str,
    active: &str,
    select: &str,
) -> String {
    include_str!("../catena/weight_concat.hex")
        .replace("{{FAMILY}}", name)
        .replace("{{LEFT}}", left)
        .replace("{{RIGHT}}", right)
        .replace("{{EXTENT}}", extent)
        .replace("{{ACTIVE}}", active)
        .replace("{{SELECT}}", select)
}

pub(crate) fn speech_concat() -> String {
    concat_kernel(
        "stdlib.tts.prealloc.concat",
        OWN,
        OWN,
        OWN,
        "  ({[.extent] (const.u64.0x0000000000000000)} stdlib.models.memory.own-read-u64 [v2.])
  ({[.v2] (const.u64.0x0000000000000100)} stdlib.models.memory.u64.div [v3.])
  ({[.count] (const.u64.0x0000000000000753)} stdlib.models.memory.u64.div [v4.])
  ({[.v3] [.v4]} stdlib.models.memory.u64.mul [v5.])",
        " ({[.input-arg-0] [.t7]} tts.read-bf16 [t12.])
 ([.t12] bf16.to-f32 [t13.])
 ({[.input-arg-1] [.t11]} tts.read-bf16 [t14.])
 ([.t14] bf16.to-f32 [t15.])
 ({[.t4] [.t13] [.t15]} stdlib.models.memory.select-f32 [t16.])
 ([.t16] f32.to-bf16 [t17.])
 ({[.out] [.element] [.t17]} stdlib.models.memory.write-bf16 [t18.])",
    )
}

/// Specialize the same row-concatenation kernel for full-precision checkpoint
/// data. The setup graph supplies equally typed inputs and a fully sized output.
pub fn concat_f32(borrowed: bool) -> String {
    let (kind, read) = if borrowed { (REF, "ref") } else { (OWN, "own") };
    let select = format!(
        " ({{[.input-arg-0] [.t7]}} stdlib.models.memory.{read}-read-f32 [left.])
 ({{[.input-arg-1] [.t11]}} stdlib.models.memory.{read}-read-f32 [right.])
 ({{[.t4] [.left] [.right]}} stdlib.models.memory.select-f32 [value.])
 ({{[.out] [.element] [.value]}} stdlib.models.memory.write-f32 [t18.])"
    );
    concat_kernel(
        &format!("stdlib.weights.concat-f32-{read}"),
        kind,
        kind,
        U64,
        "  ([.extent] [v5.])",
        &select,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catena::{compile_sources, runtime_sources};
    use anyhow::Result;
    use catena_lang::runtime::{GpuDialect, Runtime, RuntimeModule, Value};

    #[test]
    #[ignore = "requires GPU"]
    fn checkpoint_cast_rounds_finite_values_and_preserves_nan_class() -> Result<()> {
        let dialect = crate::model::configured_gpu_dialect()?;
        let mut runtime = Runtime::new(dialect)?;
        let source = format!(
            "{}\n{}",
            include_str!("../catena/memory.hex"),
            cast_to_bf16_source()
        );
        let artifact = runtime.load(compile_sources(runtime_sources(&source), dialect)?)?;
        let bits = [
            0,
            0x8000_0000,
            1,
            0x0000_8000,
            0x0000_8001,
            0x007f_ffff,
            0x0080_0000,
            0x3f80_7fff,
            0x3f80_8000,
            0x3f81_8000,
            0xbf81_8000,
            0x7f7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0xff80_0001,
        ];
        let values = bits.map(f32::from_bits);
        let input = runtime.mem_f32(&values)?;
        let [Value::MemOwn(out)] = artifact.exec(
            "inference.f32-to-bf16",
            [input.as_ref().into(), (values.len() as u64).into()],
        )?
        else {
            anyhow::bail!("missing BF16 result")
        };
        let rounded = out.try_to_u16_vec()?;
        assert_eq!(rounded.len(), bits.len());
        for (i, actual) in rounded.into_iter().enumerate() {
            let host = crate::weights::f32_to_bf16(values[i]);
            if values[i].is_nan() {
                assert!(crate::weights::bf16_to_f32(actual).is_nan());
                eprintln!("NaN {:08x} host {host:04x} device {actual:04x}", bits[i]);
            } else {
                assert_eq!(actual, host, "input {:08x}", bits[i]);
            }
        }
        Ok(())
    }

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
    #[ignore = "requires GPU"]
    fn speech_concat_respects_active_frames() -> Result<()> {
        let dialect = crate::model::configured_gpu_dialect()?;
        let mut source = include_str!("../catena/memory.hex").to_owned();
        for fragment in [
            include_str!("../catena/cooperative.hex"),
            &crate::catena::definitions(include_str!("../catena/tts_model.hex"), &["tts.read-bf16:"]),
            &speech_concat(),
            "(def program concat : {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val)
              ((cap.own mem) val)} -> ((cap.own mem) val) =
              ([x y meta out.] ({[.x] [.y] (const.u64.0x0000000000000001) (const.u64.0x0000000000000001)
              (const.u64.0x0000000000000EA6) [.meta] [.out]} stdlib.tts.prealloc.concat)))",
        ] {
            crate::qwen::fused::append_hex(&mut source, fragment);
        }
        let mut runtime = Runtime::new(dialect)?;
        let artifact = runtime.load(compile_sources(runtime_sources(&source), dialect)?)?;
        let x = runtime.mem_u16(&vec![0x3f80; 1875])?;
        let y = runtime.mem_u16(&vec![0x4000; 1875])?;
        let header = runtime.mem_u64(&[256])?;
        let out = runtime.mem_u16(&vec![0x1234; 3750])?;
        let [Value::MemOwn(out)] =
            artifact.exec("concat", [x.into(), y.into(), header.into(), out.into()])?
        else {
            anyhow::bail!("missing speech concat result")
        };
        let values = out.try_to_u16_vec()?;
        assert_eq!(values[..2], [0x3f80, 0x4000]);
        assert!(values[2..].iter().all(|&n| n == 0x1234));
        Ok(())
    }

    fn concat_module(dialect: GpuDialect) -> Result<RuntimeModule> {
        let mut source = include_str!("../catena/memory.hex").to_owned();
        for fragment in [
            include_str!("../catena/cooperative.hex"),
            &concat_f32(true),
            &concat_f32(false),
            "(def program concat : {((cap.ref mem) val) ((cap.ref mem) val)} -> ((cap.own mem) val) =
             ([x y.]
              ((const.u64.0x0000000000000006) stdlib.models.memory.alloc-f32 [out.])
              ({[.x] [.y] (const.u64.0x0000000000000002) (const.u64.0x0000000000000001)
                (const.u64.0x0000000000000006) (const.u64.0x0000000000000006) [.out]}
               stdlib.weights.concat-f32-ref)))",
        ] {
            crate::qwen::fused::append_hex(&mut source, fragment);
        }
        compile_sources(runtime_sources(&source), dialect)
    }

    #[test]
    fn checkpoint_concat_compiles_for_both_dialects() -> Result<()> {
        for dialect in [GpuDialect::Cuda, GpuDialect::Hip] {
            concat_module(dialect)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires GPU"]
    fn concatenated_weights_keep_all_f32_bits() -> Result<()> {
        let dialect = crate::model::configured_gpu_dialect()?;
        let mut runtime = Runtime::new(dialect)?;
        let artifact = runtime.load(concat_module(dialect)?)?;
        let left = [0x3f80_0001, 0x8000_0000, 0x7fc1_2345, 1];
        let right = [0xbf80_0001, 0x7f80_0000];
        let x = runtime.mem_f32(&left.map(f32::from_bits))?;
        let y = runtime.mem_f32(&right.map(f32::from_bits))?;
        let [Value::MemOwn(out)] =
            artifact.exec("concat", [x.as_ref().into(), y.as_ref().into()])?
        else {
            anyhow::bail!("missing concat result")
        };
        let bits: Vec<_> = out
            .try_to_f32_vec()?
            .into_iter()
            .map(f32::to_bits)
            .collect();
        assert_eq!(
            bits,
            [left[0], left[1], right[0], left[2], left[3], right[1]]
        );
        Ok(())
    }

    fn norm_module(dialect: GpuDialect) -> Result<RuntimeModule> {
        let mut source = include_str!("../catena/memory.hex").to_owned();
        for fragment in [
            include_str!("../catena/numeric.hex"),
            include_str!("../catena/cooperative.hex"),
            ROW_NORM,
            "(def program norm : {((cap.ref mem) val) ((cap.ref mem) val)} -> ((cap.own mem) val) =
             ([v g.] ({[.v] [.g] (const.u64.0x0000000000000002) (const.u64.0x0000000000000002)} stdlib.weights.row-norm)))",
        ] {
            crate::qwen::fused::append_hex(&mut source, fragment);
        }
        compile_sources(runtime_sources(&source), dialect)
    }

    #[test]
    fn weight_normalization_compiles_for_both_dialects() -> Result<()> {
        for dialect in [GpuDialect::Cuda, GpuDialect::Hip] {
            norm_module(dialect)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires GPU"]
    fn normalized_weights_match_reference() -> Result<()> {
        let dialect = crate::model::configured_gpu_dialect()?;
        let mut runtime = Runtime::new(dialect)?;
        let artifact = runtime.load(norm_module(dialect)?)?;
        let input = runtime.mem_f32(&[3., 4., 5., 12.])?;
        let gain = runtime.mem_f32(&[10., 6.5])?;
        let [Value::MemOwn(out)] =
            artifact.exec("norm", [input.as_ref().into(), gain.as_ref().into()])?
        else {
            anyhow::bail!("missing weight norm result")
        };
        assert_eq!(
            out.try_to_u16_vec()?,
            crate::weights::bf16(&[6., 8., 2.5, 6.])
        );
        Ok(())
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

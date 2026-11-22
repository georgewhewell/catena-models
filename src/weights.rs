//! Validated, mmap-backed safetensors with GPU borrows tied to their storage.
use crate::checkpoint::GpuRegistration;
use anyhow::{Context, Result, bail, ensure};
use catena_lang::{
    codegen::GpuDialect,
    runtime::{MemOwn, MemRef, Runtime},
};
use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, tensor::Metadata};
use std::{
    collections::BTreeMap,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

/// Both directions use the same bounds-checked GPU element loop.
pub(crate) fn cast_to_bf16_source() -> String {
    cast_source(
        "stdlib.whisper.convert",
        "inference.f32-to-bf16",
        "f32",
        "bf16",
    )
}

pub(crate) fn widen_bf16_source() -> String {
    cast_source("stdlib.vision.widen", "vision.widen-bf16", "bf16", "f32")
}

fn cast_source(family: &str, entry: &str, input: &str, output: &str) -> String {
    include_str!("catena/weight_cast.hex")
        .replace("{{FAMILY}}", family)
        .replace("{{ENTRY}}", entry)
        .replace("{{INPUT}}", input)
        .replace("{{OUTPUT}}", output)
}

/// Read only the length and JSON header of a shard. Tensor offsets in the
/// returned metadata are relative to the returned start of tensor data.
pub fn read_safetensors_header(
    mut reader: impl Read,
    file_bytes: u64,
) -> Result<(usize, Metadata)> {
    let mut length = [0; 8];
    reader.read_exact(&mut length)?;
    let length = u64::from_le_bytes(length);
    // This is the safetensors format's maximum JSON header size.
    ensure!(length <= 100_000_000, "safetensors header too large");
    let data_start = 8 + length;
    ensure!(data_start <= file_bytes, "truncated safetensors header");
    let mut json = vec![0; length as usize];
    reader.read_exact(&mut json)?;
    // Metadata's deserializer checks shapes, dtypes, contiguity and sizes.
    let metadata: Metadata = serde_json::from_slice(&json)?;
    ensure!(
        u64::try_from(metadata.data_len())? == file_bytes - data_start,
        "safetensors tensor data length mismatch"
    );
    Ok((usize::try_from(data_start)?, metadata))
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    file: usize,
    offset: usize,
    len: usize,
}
struct Shard {
    registration: Option<GpuRegistration>,
    bytes: Mmap,
}
pub struct Weights {
    shards: Vec<Shard>,
    pub tensors: BTreeMap<String, TensorInfo>,
    converted: BTreeMap<String, MemOwn>,
}
impl Weights {
    pub fn open(checkpoint: &Path) -> Result<Self> {
        let start = std::time::Instant::now();
        let indexed: Option<crate::hfhub::SafetensorsIndex> =
            if checkpoint.to_string_lossy().ends_with(".index.json") {
                Some(crate::hfhub::read_json(checkpoint)?)
            } else {
                None
            };
        let paths: Vec<PathBuf> = if let Some(i) = &indexed {
            i.shards()?
                .into_iter()
                .map(|n| checkpoint.parent().unwrap().join(n))
                .collect()
        } else {
            vec![checkpoint.to_owned()]
        };
        let mut shards = Vec::new();
        let mut tensors = BTreeMap::new();
        for path in &paths {
            let file = File::open(path)
                .with_context(|| format!("missing checkpoint {}", path.display()))?;
            // SAFETY: model files are immutable for the lifetime of this loader.
            let bytes = unsafe { MmapOptions::new().map(&file)? };
            let (data_start, metadata) = read_safetensors_header(&bytes[..], bytes.len() as u64)
                .with_context(|| format!("invalid safetensors {}", path.display()))?;
            for (name, t) in metadata.tensors() {
                ensure!(!tensors.contains_key(&name), "duplicate tensor {name}");
                if let Some(i) = &indexed {
                    let declared = i
                        .weight_map
                        .get(&name)
                        .with_context(|| format!("unindexed tensor {name}"))?;
                    ensure!(
                        checkpoint.parent().unwrap().join(declared) == *path,
                        "shard index mismatch for {name}"
                    );
                }
                tensors.insert(
                    name,
                    TensorInfo {
                        shape: t.shape.clone(),
                        dtype: t.dtype,
                        file: shards.len(),
                        offset: data_start + t.data_offsets.0,
                        len: t.data_offsets.1 - t.data_offsets.0,
                    },
                );
            }
            shards.push(Shard {
                registration: None,
                bytes,
            });
        }
        if let Some(i) = indexed {
            ensure!(
                i.weight_map.len() == tensors.len()
                    && i.weight_map.keys().all(|k| tensors.contains_key(k)),
                "incomplete shard index"
            );
        }
        startup_timing(
            &format!("{} checkpoint metadata/mmap", checkpoint.display()),
            start,
        );
        Ok(Self {
            shards,
            tensors,
            converted: BTreeMap::new(),
        })
    }
    pub fn validate(&self, name: &str, shape: &[usize]) -> Result<()> {
        let t = self
            .tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        ensure!(
            t.shape == shape,
            "{name}: expected shape {shape:?}, got {:?}",
            t.shape
        );
        ensure!(
            matches!(t.dtype, Dtype::BF16 | Dtype::F32),
            "{name}: unsupported dtype {:?}",
            t.dtype
        );
        Ok(())
    }
    pub fn host_f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self
            .tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        let bytes = &self.shards[t.file].bytes[t.offset..t.offset + t.len];
        match t.dtype {
            Dtype::BF16 => Ok(bytes
                .chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                .collect()),
            Dtype::F32 => Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()),
            d => bail!("cannot convert {name} from {d:?}"),
        }
    }
    pub fn prepare(
        &mut self,
        runtime: &Runtime,
        names: &[String],
        f32_storage: bool,
    ) -> Result<()> {
        let start = std::time::Instant::now();
        for name in names {
            let t = self
                .tensors
                .get(name)
                .with_context(|| format!("missing tensor {name}"))?;
            if f32_storage {
                self.converted
                    .insert(name.clone(), runtime.mem_f32(&self.host_f32(name)?)?);
            } else {
                if t.dtype == Dtype::F32 {
                    self.converted
                        .insert(name.clone(), runtime.mem_u16(&bf16(&self.host_f32(name)?))?);
                    continue;
                }
                ensure!(t.dtype == Dtype::BF16, "{name}: expected BF16 or F32");
                // CUDA weights live in device memory. HIP can map aligned BF16
                // tensors directly; unaligned tensors need an owned upload.
                if runtime.dialect() == GpuDialect::Cuda || t.offset % 2 != 0 {
                    let bytes = &self.shards[t.file].bytes[t.offset..t.offset + t.len];
                    let values: Vec<_> = bytes
                        .chunks_exact(2)
                        .map(|b| u16::from_le_bytes([b[0], b[1]]))
                        .collect();
                    self.converted
                        .insert(name.clone(), runtime.mem_u16(&values)?);
                } else if self.shards[t.file].registration.is_none() {
                    self.shards[t.file].registration = Some(GpuRegistration::new(
                        &self.shards[t.file].bytes,
                        runtime.dialect(),
                    )?);
                }
            }
        }
        startup_timing(
            if f32_storage {
                "weights F32 conversion/upload"
            } else {
                "weights BF16 registration/upload"
            },
            start,
        );
        Ok(())
    }
    pub(crate) fn insert_prepared(&mut self, name: &str, mem: MemOwn) -> Result<()> {
        ensure!(self.tensors.contains_key(name), "missing tensor {name}");
        self.converted.insert(name.to_owned(), mem);
        Ok(())
    }
    pub fn gpu_ref(&self, name: &str) -> Result<MemRef<'_>> {
        if let Some(m) = self.converted.get(name) {
            return Ok(m.as_ref());
        }
        let t = self
            .tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        self.shards[t.file]
            .registration
            .as_ref()
            .context("weights not prepared")?
            .reference(t.offset, t.len)
    }
    pub fn byte_len(&self) -> usize {
        self.shards.iter().map(|s| s.bytes.len()).sum()
    }
}

/// Round F32 to nearest-even BF16; preserve NaNs.
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        ((bits >> 16) as u16) | 0x0040
    } else {
        (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }
}
pub fn bf16(values: &[f32]) -> Vec<u16> {
    values.iter().copied().map(f32_to_bf16).collect()
}
pub(crate) fn startup_timing(stage: &str, start: std::time::Instant) {
    if std::env::var_os("CATENA_STARTUP_TIMINGS").is_some() {
        eprintln!("[startup] {stage}: {:.4}s", start.elapsed().as_secs_f64());
    }
}

pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// IEEE half precision, including subnormals and non-finite values.
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 31);
    let mantissa = u32::from(bits & 1023);
    if exponent == 0 {
        let value = mantissa as f32 * 2_f32.powi(-24);
        if sign == 0 { value } else { -value }
    } else {
        f32::from_bits(
            sign | (if exponent == 31 { 255 } else { exponent + 112 } << 23) | (mantissa << 13),
        )
    }
}

#[cfg(test)]
mod half_tests {
    #[test]
    fn half_conversion_covers_special_values_and_subnormals() {
        use super::f16_to_f32;
        assert_eq!(f16_to_f32(0x3c00), 1.);
        assert_eq!(f16_to_f32(0xc000), -2.);
        assert_eq!(f16_to_f32(1), 2_f32.powi(-24));
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0_f32).to_bits());
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
    }
}

#[cfg(test)]
mod header_tests {
    use super::*;

    #[test]
    fn header_only_reader_checks_ranges_against_the_complete_file() {
        let json = br#"{"v":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json);
        let size = bytes.len() as u64 + 4;
        let (start, metadata) = read_safetensors_header(bytes.as_slice(), size).unwrap();
        assert_eq!(start, bytes.len());
        assert_eq!(metadata.info("v").unwrap().data_offsets, (0, 4));
        for wrong_size in [size - 5, size - 1, size + 1] {
            assert!(read_safetensors_header(bytes.as_slice(), wrong_size).is_err());
        }
        assert!(read_safetensors_header(&bytes[..bytes.len() - 1], size).is_err());
        assert!(read_safetensors_header(&u64::MAX.to_le_bytes()[..], u64::MAX).is_err());
    }
}

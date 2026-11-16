//! Generate one linear-use Catena composition; Rust does not execute encoder layers.
use super::{VisionConfig, VisionKind};
use std::collections::{BTreeMap, HashMap};

fn c(n: usize) -> String {
    format!("(const.u64.0x{n:016X})")
}
fn f(x: f32) -> String {
    format!("(const.u32.0x{:08X} u32.bitcast-f32)", x.to_bits())
}
fn w(name: &str) -> String {
    format!("[.{name}]")
}
struct Op {
    name: String,
    args: Vec<String>,
    out: String,
    extra: Vec<String>,
}
struct Builder {
    ops: Vec<Op>,
    weights: HashMap<String, String>,
    bf16: bool,
}
impl Builder {
    fn op(&mut self, name: &str, args: Vec<String>) -> String {
        let out = format!("m{}", self.ops.len());
        self.ops.push(Op {
            name: name.into(),
            args,
            out: out.clone(),
            extra: vec![],
        });
        out
    }
    fn op_pair(&mut self, name: &str, args: Vec<String>) -> (String, String) {
        let first = self.op(name, args);
        let second = format!("{first}b");
        self.ops.last_mut().unwrap().extra.push(second.clone());
        (first, second)
    }
    fn weight(&self, name: &str) -> String {
        w(&self.weights[name])
    }
    fn linear(&mut self, x: &str, prefix: &str, rows: &str, ci: usize, co: usize) -> String {
        let bias = format!("{prefix}.bias");
        let bias = self.weight(if self.weights.contains_key(&bias) {
            &bias
        } else {
            "__zero"
        });
        self.op(
            if self.bf16 {
                "stdlib.vision.wmma.linear-bf16-own"
            } else {
                "stdlib.vision.wmma.linear-own"
            },
            vec![
                w(x),
                self.weight(&format!("{prefix}.weight")),
                bias,
                w(rows),
                c(ci),
                c(co),
            ],
        )
    }
    fn norm(&mut self, x: &str, prefix: &str, rows: &str, cols: usize, eps: f32) -> String {
        self.op(
            if self.bf16 {
                "stdlib.vision.qwen.norm-bf16"
            } else {
                "stdlib.vision.norm"
            },
            vec![
                w(x),
                self.weight(&format!("{prefix}.weight")),
                self.weight(&format!("{prefix}.bias")),
                w(rows),
                c(cols),
                c(1),
                c(0),
                c(2),
                f(eps),
            ],
        )
    }

    fn add(&mut self, x: &str, y: &str) -> String {
        self.op(
            if self.bf16 {
                "stdlib.vision.qwen.add"
            } else {
                "stdlib.vision.add"
            },
            vec![w(x), w(y), w("hidden-count")],
        )
    }
    fn render(self, result: &str, names: &[String], lfm3: bool) -> String {
        let mut last = BTreeMap::new();
        for (i, op) in self.ops.iter().enumerate() {
            last.insert(op.out.clone(), i);
            for output in &op.extra {
                last.insert(output.clone(), i);
            }
            for arg in &op.args {
                if let Some(name) = arg.strip_prefix("[.").and_then(|s| s.strip_suffix(']'))
                    && last.contains_key(name)
                {
                    last.insert(name.into(), i);
                }
            }
        }
        let mut aliases = BTreeMap::new();
        let mut body = String::new();
        for (i, op) in self.ops.iter().enumerate() {
            let args: Vec<_> = op
                .args
                .iter()
                .map(|a| {
                    if let Some(n) = a.strip_prefix("[.").and_then(|s| s.strip_suffix(']')) {
                        w(aliases.get(n).map(String::as_str).unwrap_or(n))
                    } else {
                        a.clone()
                    }
                })
                .collect();
            if op.extra.is_empty() {
                body += &format!(" ({{{}}} {} [{}.])\n", args.join(" "), op.name, op.out);
            } else {
                body += &format!(" ({{{}}} {} [pair-{i}-0.])\n", args.join(" "), op.name);
                let outputs = std::iter::once(&op.out)
                    .chain(op.extra.iter())
                    .collect::<Vec<_>>();
                for j in 0..outputs.len() - 1 {
                    let tail = if j + 2 == outputs.len() {
                        outputs[j + 1].clone()
                    } else {
                        format!("pair-{i}-{}", j + 1)
                    };
                    body += &format!(" ([.pair-{i}-{j}] *.elim [{} {tail}.])\n", outputs[j]);
                }
            }
            let mut continuation = op.out.clone();
            for (name, &end) in &last {
                if end != i || name == &op.out || op.extra.contains(name) || name == result {
                    continue;
                }
                let next = format!("free-{i}-{name}");
                body += &format!(
                    " ({{{} {}}} stdlib.models.resources.free-f32-before [{next}.])\n",
                    w(aliases.get(name).map(String::as_str).unwrap_or(name)),
                    w(&continuation)
                );
                continuation = next;
            }
            aliases.insert(op.out.clone(), continuation);
            for output in &op.extra {
                aliases.insert(output.clone(), output.clone());
            }
        }
        body += &w(aliases.get(result).map(String::as_str).unwrap_or(result));
        let mut types = vec!["((cap.ref mem) val)"; 5];
        if lfm3 {
            types.push("(u64 val)");
        }
        types.extend(vec!["((cap.ref mem) val)"; names.len()]);
        let types = types.join(" ");
        let width = if lfm3 { "width " } else { "" };
        let weight_args: Vec<_> = (0..names.len()).map(|i| format!("w{i}")).collect();
        format!(
            "\n(def program vision.encode : {{{types}}} -> ((cap.own mem) val)\n= ([patches positions indices coefficients frequencies {width}{}.]\n{body}))\n",
            weight_args.join(" ")
        )
    }
}
pub(crate) fn source(config: &VisionConfig, names: &[String]) -> String {
    if config.kind == VisionKind::Lfm3 {
        return lfm3_source(config, names);
    }
    let h = config.hidden_size;
    let dim = h / config.heads;
    assert!(
        matches!(dim, 64 | 72),
        "unsupported Qwen vision head width {dim}"
    );
    assert!(
        h.is_multiple_of(32)
            && config.intermediate_size.is_multiple_of(2)
            && config.output_size.is_multiple_of(32)
            && (3 * config.patch_size * config.patch_size).is_multiple_of(32)
    );
    let mut b = Builder {
        bf16: config.kind == VisionKind::Qwen35,
        ops: vec![],
        weights: names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), format!("w{i}")))
            .collect(),
    };
    // Shape scalars come from the patch buffer at runtime; they do not trigger recompilation.
    let mut prelude = format!(
        " ([.patches] stdlib.models.memory.ref-count-f32 [elements.])\n ({{[.elements] {}}} stdlib.models.memory.u64.div [rows.])\n ({{[.rows] {}}} stdlib.models.memory.u64.div [merged-rows.])\n ({{[.rows] {}}} stdlib.models.memory.u64.mul [hidden-count.])\n ({{[.rows] {}}} stdlib.models.memory.u64.mul [ff-count.])\n",
        c(3 * config.patch_size * config.patch_size),
        c(config.merge_size * config.merge_size),
        c(h),
        c(config.intermediate_size)
    );
    let ci = 3 * config.patch_size * config.patch_size;
    // Both temporal slices see the same patch for a still image; add the bias once.
    let patches = b.op(
        "stdlib.vision.wmma.m64n64k32.patch-bf16.linear-rr.native",
        vec![
            w("patches"),
            b.weight("v.patch_embd.weight"),
            b.weight("v.patch_embd.bias"),
            w("rows"),
            c(ci),
            c(h),
            b.weight("v.patch_embd.weight.1"),
        ],
    );
    let pos = b.op(
        "stdlib.vision.position",
        vec![
            b.weight("v.position_embd.weight"),
            w("indices"),
            w("coefficients"),
            w("hidden-count"),
            c(h),
        ],
    );
    let mut x = b.add(&patches, &pos);
    let mut prepared = None;
    for l in 0..config.layers {
        let prefix = format!("v.blk.{l}");
        let n = prepared
            .take()
            .unwrap_or_else(|| b.norm(&x, &format!("{prefix}.ln1"), "rows", h, config.eps));
        let qkv = b.linear(&n, &format!("{prefix}.attn_qkv"), "rows", h, 3 * h);
        let q = b.op(
            if dim == 64 {
                "stdlib.vision.qwen.qkv"
            } else {
                "stdlib.vision.qwen.qkv-d72"
            },
            vec![
                w(&qkv),
                w("positions"),
                w("frequencies"),
                w("hidden-count"),
                c(h),
            ],
        );
        let k = format!("{q}b");
        let v = format!("{q}c");
        b.ops.last_mut().unwrap().extra = vec![k.clone(), v.clone()];
        let dim = h / config.heads;
        let a = b.op(
            &format!("stdlib.vision.online.d{dim}.run"),
            vec![
                w(&q),
                w(&k),
                w(&v),
                w("rows"),
                w("rows"),
                c(config.heads),
                c(config.heads),
                f(1. / (dim as f32).sqrt()),
                c(0),
                c(0),
            ],
        );
        let a = b.linear(&a, &format!("{prefix}.attn_out"), "rows", h, h);
        let (residual, n) = b.op_pair(
            "stdlib.vision.qwen.add-ln",
            vec![
                w(&x),
                w(&a),
                b.weight(&format!("{prefix}.ln2.weight")),
                b.weight(&format!("{prefix}.ln2.bias")),
                w("rows"),
                c(h),
                f(config.eps),
            ],
        );
        x = residual;
        let up = b.op(
            "stdlib.vision.wmma.m64n64k32.gelu-tanh-bf16.linear-or.native",
            vec![
                w(&n),
                b.weight(&format!("{prefix}.ffn_up.weight")),
                b.weight(&format!("{prefix}.ffn_up.bias")),
                w("rows"),
                c(h),
                c(config.intermediate_size),
            ],
        );
        let down = b.linear(
            &up,
            &format!("{prefix}.ffn_down"),
            "rows",
            config.intermediate_size,
            h,
        );
        if l + 1 < config.layers {
            let next = format!("v.blk.{}.ln1", l + 1);
            let (residual, n) = b.op_pair(
                "stdlib.vision.qwen.add-ln",
                vec![
                    w(&x),
                    w(&down),
                    b.weight(&format!("{next}.weight")),
                    b.weight(&format!("{next}.bias")),
                    w("rows"),
                    c(h),
                    f(config.eps),
                ],
            );
            x = residual;
            prepared = Some(n);
        } else {
            x = b.add(&x, &down);
        }
    }
    // LayerNorm is applied before reshaping neighbouring patches into merged rows.
    x = b.norm(&x, "v.post_ln", "rows", h, 1e-6);
    let merged = h * config.merge_size * config.merge_size;
    x = b.op(
        "stdlib.vision.wmma.m64n64k32.gelu-bf16.linear-or.native",
        vec![
            w(&x),
            b.weight("mm.0.weight"),
            b.weight("mm.0.bias"),
            w("merged-rows"),
            c(merged),
            c(merged),
        ],
    );
    x = b.linear(&x, "mm.2", "merged-rows", merged, config.output_size);
    let source = b.render(&x, names, false);
    let marker = format!(
        "{}.]\n",
        (0..names.len())
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    prelude.insert_str(0, &marker);
    source.replacen(&marker, &prelude, 1)
}

fn lfm3_source(config: &VisionConfig, names: &[String]) -> String {
    let h = config.hidden_size;
    let d = h / config.heads;
    let k = d * config.kv_heads;
    // Preconditions of the generated matrix-core and fused head/row kernels.
    assert_eq!(d, 64, "LFM3 vision requires 64-channel accelerated heads");
    assert!(
        h.is_multiple_of(256)
            && config.intermediate_size.is_multiple_of(32)
            && config.output_size.is_multiple_of(32)
            && (3 * config.patch_size * config.patch_size).is_multiple_of(32),
        "unsupported LFM3 matrix-core dimensions"
    );
    let mut b = Builder {
        bf16: config.kind == VisionKind::Qwen35,
        ops: vec![],
        weights: names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), format!("w{i}")))
            .collect(),
    };
    let mut prelude = format!(
        " ([.patches] stdlib.models.memory.ref-count-f32 [elements.])\n ({{[.elements] {}}} stdlib.models.memory.u64.div [rows.])\n ({{[.rows] {}}} stdlib.models.memory.u64.div [merged-rows.])\n ({{[.rows] {}}} stdlib.models.memory.u64.mul [hidden-count.])\n ({{[.rows] {}}} stdlib.models.memory.u64.mul [ff-count.])\n ({{[.rows] {}}} stdlib.models.memory.u64.mul [head-rows.])\n ({{[.rows] {}}} stdlib.models.memory.u64.mul [key-head-rows.])\n\n",
        c(3 * config.patch_size * config.patch_size),
        c(4),
        c(h),
        c(config.intermediate_size),
        c(config.heads),
        c(config.kv_heads)
    );
    let ci = 3 * config.patch_size * config.patch_size;
    let patches = b.op(
        "stdlib.vision.wmma.m64n64k32.patch.linear-rr.native",
        vec![
            w("patches"),
            b.weight("v.patch_embd.weight"),
            b.weight("__zero"),
            w("rows"),
            c(ci),
            c(h),
            b.weight("v.patch_embd.weight.1"),
        ],
    );
    let mut x = b.op(
        "stdlib.lfm3.vision.rms",
        vec![
            w(&patches),
            b.weight("v.embedding_norm.weight"),
            w("rows"),
            c(h),
            f(config.eps),
        ],
    );
    let mut prepared = None;
    for i in 0..config.layers {
        let prefix = format!("v.blk.{i}");
        let n = prepared.take().unwrap_or_else(|| {
            b.op(
                "stdlib.lfm3.vision.rms",
                vec![
                    w(&x),
                    b.weight(&format!("{prefix}.ln1.weight")),
                    w("rows"),
                    c(h),
                    f(config.eps),
                ],
            )
        });
        let operator = if config.conv_layers.contains(&i) {
            let proj = b.linear(&n, &format!("{prefix}.conv2d_in_proj"), "rows", h, 3 * h);
            let spatial = b.op(
                "stdlib.lfm3.vision.spatial",
                vec![
                    w(&proj),
                    b.weight(&format!("{prefix}.conv2d_spatial.weight")),
                    w("rows"),
                    c(h),
                    w("width"),
                ],
            );
            b.linear(&spatial, &format!("{prefix}.conv2d_out_proj"), "rows", h, h)
        } else {
            let q = b.linear(&n, &format!("{prefix}.attn_q"), "rows", h, h);
            let key = b.linear(&n, &format!("{prefix}.attn_k"), "rows", h, k);
            let v = b.linear(&n, &format!("{prefix}.attn_v"), "rows", h, k);
            let q = b.op(
                "stdlib.lfm3.vision.rms-rope",
                vec![
                    w(&q),
                    b.weight(&format!("{prefix}.attn_q_norm.weight")),
                    w("positions"),
                    w("frequencies"),
                    w("head-rows"),
                    c(config.heads),
                    c(d),
                    f(config.eps),
                ],
            );
            let key = b.op(
                "stdlib.lfm3.vision.rms-rope",
                vec![
                    w(&key),
                    b.weight(&format!("{prefix}.attn_k_norm.weight")),
                    w("positions"),
                    w("frequencies"),
                    w("key-head-rows"),
                    c(config.kv_heads),
                    c(d),
                    f(config.eps),
                ],
            );
            let a = b.op(
                "stdlib.vision.online.d64.run",
                vec![
                    w(&q),
                    w(&key),
                    w(&v),
                    w("rows"),
                    w("rows"),
                    c(config.heads),
                    c(config.kv_heads),
                    f(1. / (d as f32).sqrt()),
                    c(0),
                    c(0),
                ],
            );
            b.linear(&a, &format!("{prefix}.attn_out"), "rows", h, h)
        };
        let (residual, n) = b.op_pair(
            "stdlib.lfm3.vision.add-rms",
            vec![
                w(&x),
                w(&operator),
                b.weight(&format!("{prefix}.ln2.weight")),
                w("rows"),
                c(h),
                f(config.eps),
            ],
        );
        x = residual;
        let gated = b.op(
            "stdlib.vision.wmma.m64n64k32.gate.linear-or.native",
            vec![
                w(&n),
                b.weight(&format!("{prefix}.ffn_gate.weight")),
                b.weight("__zero"),
                w("rows"),
                c(h),
                c(config.intermediate_size),
                b.weight(&format!("{prefix}.ffn_up.weight")),
                b.weight("__zero"),
            ],
        );
        let down = b.linear(
            &gated,
            &format!("{prefix}.ffn_down"),
            "rows",
            config.intermediate_size,
            h,
        );
        if i + 1 < config.layers {
            let (residual, next) = b.op_pair(
                "stdlib.lfm3.vision.add-rms",
                vec![
                    w(&x),
                    w(&down),
                    b.weight(&format!("v.blk.{}.ln1.weight", i + 1)),
                    w("rows"),
                    c(h),
                    f(config.eps),
                ],
            );
            x = residual;
            prepared = Some(next);
        } else {
            x = b.add(&x, &down);
        }
    }
    x = b.norm(&x, "mm.input_norm", "rows", h, 1e-6);
    x = b.op(
        "stdlib.lfm3.vision.merge",
        vec![w(&x), w("rows"), c(h), w("width")],
    );
    x = b.op(
        "stdlib.vision.wmma.m64n64k32.gelu.linear-or.native",
        vec![
            w(&x),
            b.weight("mm.0.weight"),
            b.weight("mm.0.bias"),
            w("merged-rows"),
            c(4 * h),
            c(4 * h),
        ],
    );
    x = b.linear(&x, "mm.2", "merged-rows", 4 * h, config.output_size);
    let source = b.render(&x, names, true);
    let marker = format!(
        "{}.]\n",
        (0..names.len())
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    prelude.insert_str(0, &marker);
    source.replacen(&marker, &prelude, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_qwen_projector_composes_without_tile_alignment_panics() {
        // Cached 35B-A3B projector: D72, a partial LayerNorm block and a K16 tail.
        let config = VisionConfig {
            kind: VisionKind::Qwen35,
            resize: crate::vision::ImageResize::PillowBicubic,
            kv_heads: 16,
            conv_layers: vec![],
            rope_theta: 10000.,
            hidden_size: 1152,
            intermediate_size: 4304,
            output_size: 2048,
            layers: 27,
            heads: 16,
            patch_size: 16,
            merge_size: 2,
            position_side: 48,
            eps: 1e-6,
            min_pixels: 65536,
            max_pixels: 16777216,
            mean: [0.5; 3],
            std: [0.5; 3],
        };
        let mut names = vec!["v.patch_embd.weight.1".into(), "__zero".into()];
        for prefix in [
            "v.patch_embd",
            "v.position_embd",
            "v.post_ln",
            "mm.0",
            "mm.2",
        ] {
            for suffix in ["weight", "bias"] {
                names.push(format!("{prefix}.{suffix}"));
            }
        }
        for layer in 0..config.layers {
            for prefix in ["ln1", "ln2", "attn_qkv", "attn_out", "ffn_up", "ffn_down"] {
                for suffix in ["weight", "bias"] {
                    names.push(format!("v.blk.{layer}.{prefix}.{suffix}"));
                }
            }
        }
        assert!(!source(&config, &names).is_empty());
    }
}

//! Writes a network as an ONNX model, the file format ONNX Runtime reads.
//! ONNX is protocol buffers; the few messages needed are encoded by hand.

use super::net::{GROUP_NORM_EPS, Net, Op};

/// The ONNX model for `net`, with spatial sizes left open. It returns
/// "logits": scores for each class, [1, classes, x, y, z].
pub fn model(net: &Net) -> Vec<u8> {
    let mut graph = Graph::default();
    let name = |i: usize| format!("t{i}");
    for (i, layer) in net.layers.iter().enumerate() {
        let out = name(i);
        match &layer.op {
            Op::Input => {}
            Op::Conv {
                x,
                weight,
                bias,
                k,
                stride,
                pad,
            } => {
                let cin = net.layers[*x].channels;
                let (w, b) = (format!("w{i}"), format!("b{i}"));
                graph.floats(&w, &[layer.channels, cin, *k, *k, *k], weight);
                graph.floats(&b, &[layer.channels], bias);
                graph.node(
                    "Conv",
                    &[&name(*x), &w, &b],
                    &out,
                    vec![
                        Attr::Ints("kernel_shape", vec![*k as i64; 3]),
                        Attr::Ints("strides", vec![*stride as i64; 3]),
                        Attr::Ints("pads", vec![*pad as i64; 6]),
                    ],
                );
            }
            Op::ConvTranspose {
                x,
                weight,
                bias,
                k,
                stride,
                pad,
                out_pad,
            } => {
                let cin = net.layers[*x].channels;
                let (w, b) = (format!("w{i}"), format!("b{i}"));
                graph.floats(&w, &[cin, layer.channels, *k, *k, *k], weight);
                graph.floats(&b, &[layer.channels], bias);
                graph.node(
                    "ConvTranspose",
                    &[&name(*x), &w, &b],
                    &out,
                    vec![
                        Attr::Ints("kernel_shape", vec![*k as i64; 3]),
                        Attr::Ints("strides", vec![*stride as i64; 3]),
                        Attr::Ints("pads", vec![*pad as i64; 6]),
                        Attr::Ints("output_padding", vec![*out_pad as i64; 3]),
                    ],
                );
            }
            Op::PRelu { x, slope } => {
                let s = format!("s{i}");
                graph.floats(&s, &[1], &[*slope]);
                graph.node("PRelu", &[&name(*x), &s], &out, vec![]);
            }
            Op::GroupNorm {
                x,
                groups,
                gamma,
                beta,
                relu,
            } => {
                // ONNX's own GroupNormalization changed meaning between
                // versions; instance norm over the groups is the same sum.
                let c = layer.channels;
                let p = format!("g{i}");
                graph.int64s(&format!("{p}_grouped"), &[3], &[1, *groups as i64, -1]);
                graph.floats(&format!("{p}_ones"), &[*groups], &vec![1.0; *groups]);
                graph.floats(&format!("{p}_zeros"), &[*groups], &vec![0.0; *groups]);
                graph.floats(&format!("{p}_gamma"), &[c, 1, 1, 1], gamma);
                graph.floats(&format!("{p}_beta"), &[c, 1, 1, 1], beta);
                graph.node("Shape", &[&name(*x)], &format!("{p}_shape"), vec![]);
                graph.node(
                    "Reshape",
                    &[&name(*x), &format!("{p}_grouped")],
                    &format!("{p}_r"),
                    vec![],
                );
                graph.node(
                    "InstanceNormalization",
                    &[
                        &format!("{p}_r"),
                        &format!("{p}_ones"),
                        &format!("{p}_zeros"),
                    ],
                    &format!("{p}_n"),
                    vec![Attr::Float("epsilon", GROUP_NORM_EPS)],
                );
                graph.node(
                    "Reshape",
                    &[&format!("{p}_n"), &format!("{p}_shape")],
                    &format!("{p}_back"),
                    vec![],
                );
                graph.node(
                    "Mul",
                    &[&format!("{p}_back"), &format!("{p}_gamma")],
                    &format!("{p}_scaled"),
                    vec![],
                );
                let shifted = if *relu {
                    format!("{p}_shifted")
                } else {
                    out.clone()
                };
                graph.node(
                    "Add",
                    &[&format!("{p}_scaled"), &format!("{p}_beta")],
                    &shifted,
                    vec![],
                );
                if *relu {
                    graph.node("Relu", &[&shifted], &out, vec![]);
                }
            }
            Op::Add { a, b } => graph.node("Add", &[&name(*a), &name(*b)], &out, vec![]),
            Op::Concat { a, b } => graph.node(
                "Concat",
                &[&name(*a), &name(*b)],
                &out,
                vec![Attr::Int("axis", 1)],
            ),
            Op::Upsample { x } => {
                let s = format!("u{i}");
                graph.floats(&s, &[5], &[1.0, 1.0, 2.0, 2.0, 2.0]);
                graph.node(
                    "Resize",
                    &[&name(*x), "", &s],
                    &out,
                    vec![
                        Attr::Str("mode", "linear"),
                        Attr::Str("coordinate_transformation_mode", "half_pixel"),
                    ],
                );
            }
        }
    }
    graph.node("Identity", &[&name(net.output)], "logits", vec![]);
    // The input is t0, the input layer's name.
    let input = value_info("t0", FLOAT, net.layers[0].channels);
    let output = value_info("logits", FLOAT, net.classes());
    let mut g = Pb::default();
    for n in &graph.nodes {
        g.bytes(1, n);
    }
    g.string(2, "net");
    for t in &graph.initializers {
        g.bytes(5, t);
    }
    g.bytes(11, &input);
    g.bytes(12, &output);
    let mut m = Pb::default();
    m.int(1, 8); // IR version
    m.string(2, "quick_dicom");
    m.bytes(7, &g.0);
    let mut opset = Pb::default();
    opset.string(1, "");
    opset.int(2, 17);
    m.bytes(8, &opset.0);
    m.0
}

const FLOAT: i64 = 1;
const INT64: i64 = 7;

/// A tensor's name and type, with the batch fixed at 1 and the spatial
/// sizes left open.
fn value_info(name: &str, elem_type: i64, channels: usize) -> Vec<u8> {
    let mut shape = Pb::default();
    for (i, dim) in [Some(1), Some(channels as i64), None, None, None]
        .iter()
        .enumerate()
    {
        let mut d = Pb::default();
        match dim {
            Some(v) => d.int(1, *v),
            None => d.string(2, ["x", "y", "z"][i - 2]),
        }
        shape.bytes(1, &d.0);
    }
    let mut tensor = Pb::default();
    tensor.int(1, elem_type);
    tensor.bytes(2, &shape.0);
    let mut ty = Pb::default();
    ty.bytes(1, &tensor.0);
    let mut info = Pb::default();
    info.string(1, name);
    info.bytes(2, &ty.0);
    info.0
}

enum Attr {
    Int(&'static str, i64),
    Ints(&'static str, Vec<i64>),
    Float(&'static str, f32),
    Str(&'static str, &'static str),
}

#[derive(Default)]
struct Graph {
    nodes: Vec<Vec<u8>>,
    initializers: Vec<Vec<u8>>,
}

impl Graph {
    fn node(&mut self, op: &str, inputs: &[&str], output: &str, attrs: Vec<Attr>) {
        let mut n = Pb::default();
        for i in inputs {
            n.string(1, i);
        }
        n.string(2, output);
        n.string(3, output);
        n.string(4, op);
        for a in attrs {
            let mut p = Pb::default();
            match a {
                Attr::Int(name, v) => {
                    p.string(1, name);
                    p.int(3, v);
                    p.int(20, 2);
                }
                Attr::Ints(name, v) => {
                    p.string(1, name);
                    for x in v {
                        p.int(8, x);
                    }
                    p.int(20, 7);
                }
                Attr::Float(name, v) => {
                    p.string(1, name);
                    p.key(2, 5);
                    p.0.extend_from_slice(&v.to_le_bytes());
                    p.int(20, 1);
                }
                Attr::Str(name, v) => {
                    p.string(1, name);
                    p.string(4, v);
                    p.int(20, 3);
                }
            }
            n.bytes(5, &p.0);
        }
        self.nodes.push(n.0);
    }

    fn floats(&mut self, name: &str, dims: &[usize], data: &[f32]) {
        debug_assert_eq!(dims.iter().product::<usize>(), data.len());
        let mut t = Pb::default();
        for &d in dims {
            t.int(1, d as i64);
        }
        t.int(2, FLOAT);
        t.string(8, name);
        t.bytes(9, bytemuck::cast_slice(data));
        self.initializers.push(t.0);
    }

    fn int64s(&mut self, name: &str, dims: &[usize], data: &[i64]) {
        let mut t = Pb::default();
        for &d in dims {
            t.int(1, d as i64);
        }
        t.int(2, INT64);
        t.string(8, name);
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        t.bytes(9, &raw);
        self.initializers.push(t.0);
    }
}

/// A protocol buffer message being written.
#[derive(Default)]
struct Pb(Vec<u8>);

impl Pb {
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.0.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.0.push(v as u8);
    }

    fn key(&mut self, field: u32, wire: u8) {
        self.varint(((field as u64) << 3) | wire as u64);
    }

    fn int(&mut self, field: u32, v: i64) {
        self.key(field, 0);
        self.varint(v as u64);
    }

    fn bytes(&mut self, field: u32, b: &[u8]) {
        self.key(field, 2);
        self.varint(b.len() as u64);
        self.0.extend_from_slice(b);
    }

    fn string(&mut self, field: u32, s: &str) {
        self.bytes(field, s.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varints_follow_the_protobuf_rules() {
        let mut p = Pb::default();
        p.varint(1);
        p.varint(300);
        p.int(1, -1);
        assert_eq!(&p.0[..3], &[1, 0xAC, 0x02]);
        // Negative int64s take ten bytes.
        assert_eq!(p.0.len(), 3 + 1 + 10);
    }
}

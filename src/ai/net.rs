//! The segmentation networks as a list of layers, built from a state dict,
//! and a plain CPU implementation of each layer to check the fast ones
//! against.
//!
//! Volumes are laid out channel by channel, then along the model's first,
//! second and third axes, the third varying fastest: the order of a
//! PyTorch tensor of shape [1, C, X, Y, Z].

use std::collections::HashMap;

#[cfg(test)]
use rayon::prelude::*;

use super::torch::Tensor;

/// A layer's index in [`Net::layers`]. Layer 0 is the input.
pub type Id = usize;

#[derive(Clone, Debug)]
pub enum Op {
    Input,
    /// 3D convolution with a cubic kernel and zero padding. Weights are
    /// [out, in, k, k, k]. Batch norm, where the network has it, is folded
    /// in.
    Conv {
        x: Id,
        weight: Vec<f32>,
        bias: Vec<f32>,
        k: usize,
        stride: usize,
        pad: usize,
    },
    /// Transposed 3D convolution; weights are [in, out, k, k, k]. Output
    /// sides are `(n - 1) * stride - 2 * pad + k + out_pad`.
    ConvTranspose {
        x: Id,
        weight: Vec<f32>,
        bias: Vec<f32>,
        k: usize,
        stride: usize,
        pad: usize,
        out_pad: usize,
    },
    /// Leaky ReLU with a learned slope shared by all channels.
    PRelu {
        x: Id,
        slope: f32,
    },
    /// Group normalisation, then ReLU if `relu`.
    GroupNorm {
        x: Id,
        groups: usize,
        gamma: Vec<f32>,
        beta: Vec<f32>,
        relu: bool,
    },
    Add {
        a: Id,
        b: Id,
    },
    /// Channels of `a`, then of `b`.
    Concat {
        a: Id,
        b: Id,
    },
    /// Twice the size along each axis, trilinear, sampling at pixel centres
    /// (PyTorch's `align_corners=False`).
    Upsample {
        x: Id,
    },
}

#[derive(Clone, Debug)]
pub struct Layer {
    pub op: Op,
    pub channels: usize,
    /// The layer's size is the input's divided by 2 to this power.
    pub level: u32,
}

#[derive(Clone, Debug)]
pub struct Net {
    pub layers: Vec<Layer>,
    pub output: Id,
}

pub const GROUP_NORM_EPS: f32 = 1e-5;
const BATCH_NORM_EPS: f32 = 1e-5;

impl Net {
    pub fn classes(&self) -> usize {
        self.layers[self.output].channels
    }

    /// Input sides must be a multiple of this.
    #[cfg(test)]
    pub fn multiple(&self) -> usize {
        1 << self.layers.iter().map(|l| l.level).max().unwrap_or(0)
    }

    fn push(&mut self, op: Op, channels: usize, level: u32) -> Id {
        self.layers.push(Layer {
            op,
            channels,
            level,
        });
        self.layers.len() - 1
    }

    fn channels(&self, id: Id) -> usize {
        self.layers[id].channels
    }

    fn level(&self, id: Id) -> u32 {
        self.layers[id].level
    }
}

/// Where a network's weights come from.
pub trait Weights {
    /// The tensor called `name`, which must have this shape.
    fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String>;
}

impl Weights for HashMap<String, Tensor> {
    fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let t = self
            .remove(name)
            .ok_or_else(|| format!("the weights have no {name}"))?;
        if t.shape != shape {
            return Err(format!("{name} is {:?}, not {shape:?}", t.shape));
        }
        Ok(t.data)
    }
}

/// Checks that building the network used every weight in the file, so a
/// file for a different architecture is not half read.
fn all_used(w: &dyn std::any::Any) -> Result<(), String> {
    if let Some(map) = w.downcast_ref::<HashMap<String, Tensor>>()
        && let Some(name) = map.keys().min()
    {
        return Err(format!(
            "the weights have {name}, which this network does not use"
        ));
    }
    Ok(())
}

/// MONAI's `UNet` with residual units (two per level), batch norm and
/// PReLU, as in the spleen model.
pub fn unet<W: Weights + 'static>(
    w: &mut W,
    in_channels: usize,
    out_channels: usize,
    channels: &[usize],
) -> Result<Net, String> {
    let mut net = Net {
        layers: Vec::new(),
        output: 0,
    };
    let x = net.push(Op::Input, in_channels, 0);
    net.output = unet_block(&mut net, w, "model", x, out_channels, channels, true)?;
    all_used(&*w)?;
    Ok(net)
}

/// One level of the U: down, the levels below with a skip around them,
/// then up. MONAI's `UNet._create_block`.
fn unet_block(
    net: &mut Net,
    w: &mut dyn Weights,
    prefix: &str,
    x: Id,
    out: usize,
    channels: &[usize],
    top: bool,
) -> Result<Id, String> {
    let c = channels[0];
    let down = residual_unit(net, w, &format!("{prefix}.0"), x, c, 2, 2, false)?;
    let below = format!("{prefix}.1.submodule");
    let sub = if channels.len() > 2 {
        unet_block(net, w, &below, down, c, &channels[1..], false)?
    } else {
        residual_unit(net, w, &below, down, channels[1], 1, 2, false)?
    };
    let joined = net.push(
        Op::Concat { a: down, b: sub },
        net.channels(down) + net.channels(sub),
        net.level(down),
    );
    let up = conv_unit(
        net,
        w,
        &format!("{prefix}.2.0"),
        joined,
        out,
        2,
        true,
        false,
    )?;
    residual_unit(net, w, &format!("{prefix}.2.1"), up, out, 1, 1, top)
}

/// MONAI's `ResidualUnit`: `units` convolutions, the first with `stride`,
/// plus the input, through a convolution if its shape changes.
#[allow(clippy::too_many_arguments)]
fn residual_unit(
    net: &mut Net,
    w: &mut dyn Weights,
    prefix: &str,
    x: Id,
    out: usize,
    stride: usize,
    units: usize,
    last_conv_only: bool,
) -> Result<Id, String> {
    let cin = net.channels(x);
    let mut y = x;
    for u in 0..units {
        let s = if u == 0 { stride } else { 1 };
        let conv_only = last_conv_only && u == units - 1;
        y = conv_unit(
            net,
            w,
            &format!("{prefix}.conv.unit{u}"),
            y,
            out,
            s,
            false,
            conv_only,
        )?;
    }
    let residual = if stride != 1 || cin != out {
        let k = if stride == 1 { 1 } else { 3 };
        let weight = w.take(&format!("{prefix}.residual.weight"), &[out, cin, k, k, k])?;
        let bias = w.take(&format!("{prefix}.residual.bias"), &[out])?;
        let level = net.level(x) + (stride == 2) as u32;
        net.push(
            Op::Conv {
                x,
                weight,
                bias,
                k,
                stride,
                pad: k / 2,
            },
            out,
            level,
        )
    } else {
        x
    };
    Ok(net.push(Op::Add { a: y, b: residual }, out, net.level(y)))
}

/// MONAI's `Convolution`: a 3x3x3 convolution, transposed or not, then
/// batch norm and PReLU unless `conv_only`.
#[allow(clippy::too_many_arguments)]
fn conv_unit(
    net: &mut Net,
    w: &mut dyn Weights,
    prefix: &str,
    x: Id,
    out: usize,
    stride: usize,
    transposed: bool,
    conv_only: bool,
) -> Result<Id, String> {
    let cin = net.channels(x);
    let k = 3;
    let shape = if transposed {
        [cin, out, k, k, k]
    } else {
        [out, cin, k, k, k]
    };
    let mut weight = w.take(&format!("{prefix}.conv.weight"), &shape)?;
    let mut bias = w.take(&format!("{prefix}.conv.bias"), &[out])?;
    if !conv_only {
        // Fold batch norm into the convolution.
        let n = |w: &mut dyn Weights, name: &str| w.take(&format!("{prefix}.adn.N.{name}"), &[out]);
        let (gamma, beta) = (n(w, "weight")?, n(w, "bias")?);
        let (mean, var) = (n(w, "running_mean")?, n(w, "running_var")?);
        let per = k * k * k;
        for o in 0..out {
            let scale = gamma[o] / (var[o] + BATCH_NORM_EPS).sqrt();
            bias[o] = (bias[o] - mean[o]) * scale + beta[o];
            if transposed {
                for i in 0..cin {
                    let at = (i * out + o) * per;
                    weight[at..at + per].iter_mut().for_each(|v| *v *= scale);
                }
            } else {
                weight[o * cin * per..(o + 1) * cin * per]
                    .iter_mut()
                    .for_each(|v| *v *= scale);
            }
        }
    }
    let level = net.level(x);
    let y = if transposed {
        net.push(
            Op::ConvTranspose {
                x,
                weight,
                bias,
                k,
                stride,
                pad: 1,
                out_pad: stride - 1,
            },
            out,
            level - (stride == 2) as u32,
        )
    } else {
        net.push(
            Op::Conv {
                x,
                weight,
                bias,
                k,
                stride,
                pad: 1,
            },
            out,
            level + (stride == 2) as u32,
        )
    };
    if conv_only {
        return Ok(y);
    }
    let slope = w.take(&format!("{prefix}.adn.A.weight"), &[1])?[0];
    Ok(net.push(Op::PRelu { x: y, slope }, out, net.level(y)))
}

/// MONAI's `SegResNet` with group norm (8 groups), ReLU and trilinear
/// upsampling, as in the whole-body model.
pub fn segresnet<W: Weights + 'static>(
    w: &mut W,
    in_channels: usize,
    out_channels: usize,
    filters: usize,
    blocks_down: &[usize],
    blocks_up: &[usize],
) -> Result<Net, String> {
    let mut net = Net {
        layers: Vec::new(),
        output: 0,
    };
    let input = net.push(Op::Input, in_channels, 0);
    let mut x = plain_conv(&mut net, w, "convInit.conv", input, filters, 3, 1, false)?;
    let mut skips = Vec::new();
    for (i, &blocks) in blocks_down.iter().enumerate() {
        let ch = filters << i;
        if i > 0 {
            x = plain_conv(
                &mut net,
                w,
                &format!("down_layers.{i}.0.conv"),
                x,
                ch,
                3,
                2,
                false,
            )?;
        }
        for j in 0..blocks {
            x = res_block(&mut net, w, &format!("down_layers.{i}.{}", j + 1), x)?;
        }
        skips.push(x);
    }
    skips.reverse();
    for (i, &blocks) in blocks_up.iter().enumerate() {
        let ch = filters << (blocks_up.len() - i - 1);
        x = plain_conv(
            &mut net,
            w,
            &format!("up_samples.{i}.0.conv"),
            x,
            ch,
            1,
            1,
            false,
        )?;
        x = net.push(Op::Upsample { x }, ch, net.level(x) - 1);
        x = net.push(
            Op::Add {
                a: x,
                b: skips[i + 1],
            },
            ch,
            net.level(x),
        );
        for j in 0..blocks {
            x = res_block(&mut net, w, &format!("up_layers.{i}.{j}"), x)?;
        }
    }
    x = group_norm(&mut net, w, "conv_final.0", x)?;
    net.output = plain_conv(
        &mut net,
        w,
        "conv_final.2.conv",
        x,
        out_channels,
        1,
        1,
        true,
    )?;
    all_used(&*w)?;
    Ok(net)
}

/// SegResNet's `ResBlock`: norm, ReLU, convolution, twice, plus the input.
fn res_block(net: &mut Net, w: &mut dyn Weights, prefix: &str, x: Id) -> Result<Id, String> {
    let ch = net.channels(x);
    let y = group_norm(net, w, &format!("{prefix}.norm1"), x)?;
    let y = plain_conv(net, w, &format!("{prefix}.conv1.conv"), y, ch, 3, 1, false)?;
    let y = group_norm(net, w, &format!("{prefix}.norm2"), y)?;
    let y = plain_conv(net, w, &format!("{prefix}.conv2.conv"), y, ch, 3, 1, false)?;
    Ok(net.push(Op::Add { a: y, b: x }, ch, net.level(x)))
}

fn group_norm(net: &mut Net, w: &mut dyn Weights, prefix: &str, x: Id) -> Result<Id, String> {
    let ch = net.channels(x);
    let gamma = w.take(&format!("{prefix}.weight"), &[ch])?;
    let beta = w.take(&format!("{prefix}.bias"), &[ch])?;
    Ok(net.push(
        Op::GroupNorm {
            x,
            groups: 8,
            gamma,
            beta,
            relu: true,
        },
        ch,
        net.level(x),
    ))
}

#[allow(clippy::too_many_arguments)]
fn plain_conv(
    net: &mut Net,
    w: &mut dyn Weights,
    prefix: &str,
    x: Id,
    out: usize,
    k: usize,
    stride: usize,
    has_bias: bool,
) -> Result<Id, String> {
    let cin = net.channels(x);
    let weight = w.take(&format!("{prefix}.weight"), &[out, cin, k, k, k])?;
    let bias = if has_bias {
        w.take(&format!("{prefix}.bias"), &[out])?
    } else {
        vec![0.0; out]
    };
    let level = net.level(x) + (stride == 2) as u32;
    Ok(net.push(
        Op::Conv {
            x,
            weight,
            bias,
            k,
            stride,
            pad: (k - stride).div_ceil(2),
        },
        out,
        level,
    ))
}

/// Runs the network on the CPU, one layer at a time, simply. `size` is the
/// input's three sides. Returns the output's values.
#[cfg(test)]
pub fn run_cpu(net: &Net, input: &[f32], size: [usize; 3]) -> Vec<f32> {
    let mut values: Vec<Option<Vec<f32>>> = vec![None; net.layers.len()];
    let mut sizes = vec![[0usize; 3]; net.layers.len()];
    // The last layer that reads each layer, so values are dropped once used.
    let mut last_use = vec![0usize; net.layers.len()];
    for (i, layer) in net.layers.iter().enumerate() {
        for x in inputs(&layer.op) {
            last_use[x] = i;
        }
    }
    for (i, layer) in net.layers.iter().enumerate() {
        let get = |id: Id| (values[id].as_deref().unwrap(), sizes[id]);
        let (out, out_size) = match &layer.op {
            Op::Input => (input.to_vec(), size),
            Op::Conv {
                x,
                weight,
                bias,
                k,
                stride,
                pad,
            } => {
                let (v, s) = get(*x);
                conv(
                    v,
                    s,
                    net.layers[*x].channels,
                    weight,
                    bias,
                    layer.channels,
                    *k,
                    *stride,
                    *pad,
                )
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
                let (v, s) = get(*x);
                let cin = net.layers[*x].channels;
                conv_transpose(
                    v,
                    s,
                    cin,
                    weight,
                    bias,
                    layer.channels,
                    *k,
                    *stride,
                    *pad,
                    *out_pad,
                )
            }
            Op::PRelu { x, slope } => {
                let (v, s) = get(*x);
                (
                    v.iter()
                        .map(|&a| if a >= 0.0 { a } else { a * slope })
                        .collect(),
                    s,
                )
            }
            Op::GroupNorm {
                x,
                groups,
                gamma,
                beta,
                relu,
            } => {
                let (v, s) = get(*x);
                (
                    group_norm_cpu(v, layer.channels, *groups, gamma, beta, *relu),
                    s,
                )
            }
            Op::Add { a, b } => {
                let ((va, s), (vb, _)) = (get(*a), get(*b));
                (va.iter().zip(vb).map(|(p, q)| p + q).collect(), s)
            }
            Op::Concat { a, b } => {
                let ((va, s), (vb, _)) = (get(*a), get(*b));
                ([va, vb].concat(), s)
            }
            Op::Upsample { x } => {
                let (v, s) = get(*x);
                upsample(v, s, layer.channels)
            }
        };
        values[i] = Some(out);
        sizes[i] = out_size;
        for x in inputs(&layer.op) {
            if last_use[x] == i {
                values[x] = None;
            }
        }
    }
    values[net.output].take().unwrap()
}

/// The layers an op reads.
pub fn inputs(op: &Op) -> Vec<Id> {
    match op {
        Op::Input => vec![],
        Op::Conv { x, .. }
        | Op::ConvTranspose { x, .. }
        | Op::PRelu { x, .. }
        | Op::GroupNorm { x, .. }
        | Op::Upsample { x } => vec![*x],
        Op::Add { a, b } | Op::Concat { a, b } => vec![*a, *b],
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn conv(
    v: &[f32],
    s: [usize; 3],
    cin: usize,
    weight: &[f32],
    bias: &[f32],
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
) -> (Vec<f32>, [usize; 3]) {
    let o = s.map(|n| (n + 2 * pad - k) / stride + 1);
    let plane = o[0] * o[1] * o[2];
    let mut out = vec![0.0f32; cout * plane];
    out.par_chunks_mut(plane).enumerate().for_each(|(co, dst)| {
        dst.iter_mut().for_each(|d| *d = bias[co]);
        for ci in 0..cin {
            let src = &v[ci * s[0] * s[1] * s[2]..(ci + 1) * s[0] * s[1] * s[2]];
            for a in 0..k {
                for b in 0..k {
                    for c in 0..k {
                        let wv = weight[(((co * cin + ci) * k + a) * k + b) * k + c];
                        for x in 0..o[0] {
                            let sx = (x * stride + a) as isize - pad as isize;
                            if sx < 0 || sx >= s[0] as isize {
                                continue;
                            }
                            for y in 0..o[1] {
                                let sy = (y * stride + b) as isize - pad as isize;
                                if sy < 0 || sy >= s[1] as isize {
                                    continue;
                                }
                                let row = (sx as usize * s[1] + sy as usize) * s[2];
                                let drow = (x * o[1] + y) * o[2];
                                for z in 0..o[2] {
                                    let sz = (z * stride + c) as isize - pad as isize;
                                    if sz >= 0 && sz < s[2] as isize {
                                        dst[drow + z] += wv * src[row + sz as usize];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    });
    (out, o)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn conv_transpose(
    v: &[f32],
    s: [usize; 3],
    cin: usize,
    weight: &[f32],
    bias: &[f32],
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
    out_pad: usize,
) -> (Vec<f32>, [usize; 3]) {
    let o = s.map(|n| (n - 1) * stride + k + out_pad - 2 * pad);
    let plane = o[0] * o[1] * o[2];
    let mut out = vec![0.0f32; cout * plane];
    out.par_chunks_mut(plane).enumerate().for_each(|(co, dst)| {
        dst.iter_mut().for_each(|d| *d = bias[co]);
        for ci in 0..cin {
            let src = &v[ci * s[0] * s[1] * s[2]..(ci + 1) * s[0] * s[1] * s[2]];
            for a in 0..k {
                for b in 0..k {
                    for c in 0..k {
                        let wv = weight[(((ci * cout + co) * k + a) * k + b) * k + c];
                        for x in 0..s[0] {
                            let ox = (x * stride + a) as isize - pad as isize;
                            if ox < 0 || ox >= o[0] as isize {
                                continue;
                            }
                            for y in 0..s[1] {
                                let oy = (y * stride + b) as isize - pad as isize;
                                if oy < 0 || oy >= o[1] as isize {
                                    continue;
                                }
                                let row = (x * s[1] + y) * s[2];
                                let drow = (ox as usize * o[1] + oy as usize) * o[2];
                                for z in 0..s[2] {
                                    let oz = (z * stride + c) as isize - pad as isize;
                                    if oz >= 0 && oz < o[2] as isize {
                                        dst[drow + oz as usize] += wv * src[row + z];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    });
    (out, o)
}

#[cfg(test)]
fn group_norm_cpu(
    v: &[f32],
    channels: usize,
    groups: usize,
    gamma: &[f32],
    beta: &[f32],
    relu: bool,
) -> Vec<f32> {
    let per_channel = v.len() / channels;
    let per_group = channels / groups;
    let mut out = vec![0.0f32; v.len()];
    out.par_chunks_mut(per_channel * per_group)
        .enumerate()
        .for_each(|(g, dst)| {
            let src = &v[g * per_channel * per_group..(g + 1) * per_channel * per_group];
            let n = src.len() as f64;
            let mean = src.iter().map(|&a| a as f64).sum::<f64>() / n;
            let var = src.iter().map(|&a| (a as f64 - mean).powi(2)).sum::<f64>() / n;
            let inv = 1.0 / (var + GROUP_NORM_EPS as f64).sqrt();
            for (i, d) in dst.iter_mut().enumerate() {
                let c = g * per_group + i / per_channel;
                let y = ((src[i] as f64 - mean) * inv) as f32 * gamma[c] + beta[c];
                *d = if relu { y.max(0.0) } else { y };
            }
        });
    out
}

#[cfg(test)]
fn upsample(v: &[f32], s: [usize; 3], channels: usize) -> (Vec<f32>, [usize; 3]) {
    let o = s.map(|n| n * 2);
    // Source position and weight for each output index along one axis.
    let taps = |n: usize| -> Vec<(usize, usize, f32)> {
        (0..n * 2)
            .map(|i| {
                let src = ((i as f32 + 0.5) / 2.0 - 0.5).max(0.0);
                let i0 = (src.floor() as usize).min(n - 1);
                let i1 = (i0 + 1).min(n - 1);
                (i0, i1, src - i0 as f32)
            })
            .collect()
    };
    let (tx, ty, tz) = (taps(s[0]), taps(s[1]), taps(s[2]));
    let plane = o[0] * o[1] * o[2];
    let mut out = vec![0.0f32; channels * plane];
    out.par_chunks_mut(plane).enumerate().for_each(|(c, dst)| {
        let src = &v[c * s[0] * s[1] * s[2]..(c + 1) * s[0] * s[1] * s[2]];
        let at = |x: usize, y: usize, z: usize| src[(x * s[1] + y) * s[2] + z];
        for (x, &(x0, x1, fx)) in tx.iter().enumerate() {
            for (y, &(y0, y1, fy)) in ty.iter().enumerate() {
                for (z, &(z0, z1, fz)) in tz.iter().enumerate() {
                    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
                    let c00 = lerp(at(x0, y0, z0), at(x0, y0, z1), fz);
                    let c01 = lerp(at(x0, y1, z0), at(x0, y1, z1), fz);
                    let c10 = lerp(at(x1, y0, z0), at(x1, y0, z1), fz);
                    let c11 = lerp(at(x1, y1, z0), at(x1, y1, z1), fz);
                    dst[(x * o[1] + y) * o[2] + z] =
                        lerp(lerp(c00, c01, fy), lerp(c10, c11, fy), fx);
                }
            }
        }
    });
    (out, o)
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// Weights made up on demand: repeatable values of a sensible size.
    pub struct Random(pub u64);

    impl Weights for Random {
        fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
            let n: usize = shape.iter().product();
            let fan_in = if shape.len() == 5 {
                shape[1] * shape[2] * shape[3] * shape[4]
            } else {
                1
            };
            let scale = 1.0 / (fan_in as f32).sqrt();
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                let u = (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                out.push(u * 2.0 * scale);
            }
            // Variances must be positive; scales near 1.
            if name.ends_with("running_var") {
                out.iter_mut().for_each(|v| *v = v.abs() + 0.5);
            }
            if name.ends_with("N.weight") || name.contains("norm") && name.ends_with("weight") {
                out.iter_mut().for_each(|v| *v += 1.0);
            }
            Ok(out)
        }
    }

    pub fn small_unet() -> Net {
        unet(&mut Random(7), 1, 3, &[4, 8, 16]).unwrap()
    }

    pub fn small_segresnet() -> Net {
        segresnet(&mut Random(11), 1, 5, 8, &[1, 2, 1], &[1, 1]).unwrap()
    }

    pub fn random_input(n: usize, seed: u64) -> Vec<f32> {
        Random(seed)
            .take("input", &[n])
            .unwrap()
            .iter()
            .map(|v| v * 4.0)
            .collect()
    }

    #[test]
    fn networks_have_the_right_shape() {
        let u = small_unet();
        assert_eq!(u.classes(), 3);
        assert_eq!(u.multiple(), 4);
        let out = run_cpu(&u, &random_input(8 * 8 * 8, 3), [8, 8, 8]);
        assert_eq!(out.len(), 3 * 8 * 8 * 8);
        let s = small_segresnet();
        assert_eq!(s.classes(), 5);
        assert_eq!(s.multiple(), 4);
        let out = run_cpu(&s, &random_input(8 * 8 * 12, 4), [8, 8, 12]);
        assert_eq!(out.len(), 5 * 8 * 8 * 12);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn weights_must_match_the_network() {
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert(
            "convInit.conv.weight".into(),
            Tensor {
                shape: vec![8, 1, 3, 3, 3],
                data: vec![0.0; 216],
            },
        );
        let err = segresnet(&mut map, 1, 5, 4, &[1], &[]).unwrap_err();
        assert!(
            err.contains("convInit.conv.weight is [8, 1, 3, 3, 3]"),
            "{err}"
        );
    }

    #[test]
    fn upsampling_matches_pytorch() {
        // torch.nn.functional.interpolate(torch.tensor([[[[[0., 1., 4.]]]]]),
        // scale_factor=2, mode="trilinear", align_corners=False) along the
        // last axis.
        let (out, size) = upsample(&[0.0, 1.0, 4.0], [1, 1, 3], 1);
        assert_eq!(size, [2, 2, 6]);
        assert_eq!(&out[..6], &[0.0, 0.25, 0.75, 1.75, 3.25, 4.0]);
    }

    /// The real models against PyTorch's output for the same input, from
    /// files made by a Python script. Run with
    /// `AI_REF=<folder> cargo test real_weights -- --ignored`.
    #[test]
    #[ignore]
    fn real_weights_match_pytorch() {
        let dir = std::path::PathBuf::from(std::env::var("AI_REF").unwrap());
        let floats = |name: &str| -> Vec<f32> {
            let b = std::fs::read(dir.join(name)).unwrap();
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };
        let input = floats("input32.f32");
        for (name, file) in [("spleen", "spleen.pt"), ("wholebody", "wholebody.pt")] {
            let mut weights =
                super::super::torch::read(&std::fs::read(dir.join(file)).unwrap()).unwrap();
            let net = if name == "spleen" {
                unet(&mut weights, 1, 2, &[16, 32, 64, 128, 256]).unwrap()
            } else {
                segresnet(&mut weights, 1, 105, 32, &[1, 2, 2, 4], &[1, 1, 1]).unwrap()
            };
            let want = floats(&format!("{name}_logits32.f32"));
            let got = run_cpu(&net, &input, [32, 32, 32]);
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let size = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            println!("{name}: worst difference {worst} in values up to {size}");
            assert!(worst < 1e-3 * size.max(1.0), "{name}: {worst}");
        }
    }
}

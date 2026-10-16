// Compute shaders for the built-in AI models. Each entry point is built as
// its own module, after the shared parameters below; the Rust side splits
// this file at the "//@" markers. Volumes are channel by channel, then
// along x, y and z, z fastest.

// Sixteen numbers whose meaning depends on the kernel.
struct P {
    a: vec4<u32>,
    b: vec4<u32>,
    c: vec4<u32>,
    d: vec4<u32>,
};
@group(0) @binding(0) var<uniform> p: P;

// The flat index of this invocation: workgroups may be laid out in two
// dimensions, `across` wide, when there are too many for one.
fn index_of(wg: vec3<u32>, li: u32, across: u32, size: u32) -> u32 {
    return (wg.y * across + wg.x) * size + li;
}

//@ conv
// 3D convolution, plain or transposed, for eight output channels of one
// voxel. a: input size, input channels. b: output size, output channels.
// c: kernel side, stride, padding, flags (bit 0 transposed; bits 1-2 the
// activation: 1 PReLU, 2 ReLU). d: PReLU slope's bits, workgroups across,
// first voxel (large outputs run in slices).
// Weights are [block of 8 outputs][input channel][tap][8].
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let ins = p.a.xyz;
    let cin = p.a.w;
    let outs = p.b.xyz;
    let cout = p.b.w;
    let k = p.c.x;
    let transposed = (p.c.w & 1u) != 0u;
    let act = p.c.w >> 1u;
    let voxels = outs.x * outs.y * outs.z;
    let v = p.d.z + index_of(wg, li, p.d.y, 64u);
    if v >= voxels {
        return;
    }
    let cb = wg.z;
    let o = vec3<i32>(i32(v / (outs.z * outs.y)), i32((v / outs.z) % outs.y), i32(v % outs.z));
    let s = vec3<i32>(i32(p.c.y));
    let pad = vec3<i32>(i32(p.c.z));
    let n = vec3<i32>(ins);
    let plane = ins.x * ins.y * ins.z;
    let taps = k * k * k;
    var lo = bias[cb * 2u];
    var hi = bias[cb * 2u + 1u];
    for (var t = 0u; t < taps; t++) {
        let kk = vec3<i32>(i32(t / (k * k)), i32((t / k) % k), i32(t % k));
        var q: vec3<i32>;
        if transposed {
            // Output o takes input q where o = q * stride - pad + kk.
            let num = o + pad - kk;
            if any(num < vec3<i32>(0)) || any(num % s != vec3<i32>(0)) {
                continue;
            }
            q = num / s;
        } else {
            q = o * s - pad + kk;
        }
        if any(q < vec3<i32>(0)) || any(q >= n) {
            continue;
        }
        let at = (u32(q.x) * ins.y + u32(q.y)) * ins.z + u32(q.z);
        var wi = (cb * cin * taps + t) * 2u;
        let wstep = taps * 2u;
        for (var ci = 0u; ci < cin; ci++) {
            let xv = x[ci * plane + at];
            lo += xv * w[wi];
            hi += xv * w[wi + 1u];
            wi += wstep;
        }
    }
    if act == 1u {
        let slope = bitcast<f32>(p.d.x);
        lo = select(lo * slope, lo, lo >= vec4<f32>(0.0));
        hi = select(hi * slope, hi, hi >= vec4<f32>(0.0));
    } else if act == 2u {
        lo = max(lo, vec4<f32>(0.0));
        hi = max(hi, vec4<f32>(0.0));
    }
    let base = cb * 8u;
    let values = array<f32, 8>(lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w);
    for (var i = 0u; i < 8u; i++) {
        if base + i < cout {
            y[(base + i) * voxels + v] = values[i];
        }
    }
}

//@ gn_partial
// Count, mean and squared deviations of one stretch of one group, in a
// single pass (Welford), merged across the workgroup (Chan). a: values
// per group, values per stretch, stretches per group.
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial: array<vec4<f32>>;

var<workgroup> wn: array<f32, 256>;
var<workgroup> wmean: array<f32, 256>;
var<workgroup> wm2: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let g = wg.y;
    let chunk = wg.x;
    let start = g * p.a.x + chunk * p.a.y;
    let end = min(start + p.a.y, (g + 1u) * p.a.x);
    var n = 0.0;
    var mean = 0.0;
    var m2 = 0.0;
    for (var i = start + li; i < end; i += 256u) {
        let v = x[i];
        n += 1.0;
        let d = v - mean;
        mean += d / n;
        m2 += d * (v - mean);
    }
    wn[li] = n;
    wmean[li] = mean;
    wm2[li] = m2;
    workgroupBarrier();
    for (var span = 128u; span > 0u; span /= 2u) {
        if li < span {
            let na = wn[li];
            let nb = wn[li + span];
            let total = na + nb;
            if total > 0.0 {
                let d = wmean[li + span] - wmean[li];
                wmean[li] += d * nb / total;
                wm2[li] += wm2[li + span] + d * d * na * nb / total;
                wn[li] = total;
            }
        }
        workgroupBarrier();
    }
    if li == 0u {
        partial[g * p.a.z + chunk] = vec4<f32>(wn[0], wmean[0], wm2[0], 0.0);
    }
}

//@ gn_final
// Merges a group's stretches into its mean and 1 / standard deviation.
// a: stretches per group. d.x: epsilon's bits.
@group(0) @binding(1) var<storage, read> partial: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> stats: array<vec2<f32>>;

var<workgroup> wn: array<f32, 64>;
var<workgroup> wmean: array<f32, 64>;
var<workgroup> wm2: array<f32, 64>;

fn merge(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let total = a.x + b.x;
    if total == 0.0 {
        return a;
    }
    let d = b.y - a.y;
    return vec3<f32>(total, a.y + d * b.x / total, a.z + b.z + d * d * a.x * b.x / total);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let g = wg.x;
    var acc = vec3<f32>(0.0);
    for (var i = li; i < p.a.x; i += 64u) {
        acc = merge(acc, partial[g * p.a.x + i].xyz);
    }
    wn[li] = acc.x;
    wmean[li] = acc.y;
    wm2[li] = acc.z;
    workgroupBarrier();
    for (var span = 32u; span > 0u; span /= 2u) {
        if li < span {
            let m = merge(vec3<f32>(wn[li], wmean[li], wm2[li]), vec3<f32>(wn[li + span], wmean[li + span], wm2[li + span]));
            wn[li] = m.x;
            wmean[li] = m.y;
            wm2[li] = m.z;
        }
        workgroupBarrier();
    }
    if li == 0u {
        let variance = wm2[0] / max(wn[0], 1.0);
        stats[g] = vec2<f32>(wmean[0], inverseSqrt(variance + bitcast<f32>(p.d.x)));
    }
}

//@ gn_apply
// Normalises each value by its group, scales and shifts it by its
// channel, then ReLU if asked. a: values, voxels per channel, channels
// per group, workgroups across. b.x: ReLU (0 or 1).
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> affine: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = index_of(wg, li, p.a.w, 256u);
    if i >= p.a.x {
        return;
    }
    let c = i / p.a.y;
    let s = stats[c / p.a.z];
    let ab = affine[c];
    var v = (x[i] - s.x) * s.y * ab.x + ab.y;
    if p.b.x == 1u {
        v = max(v, 0.0);
    }
    y[i] = v;
}

//@ add
// a: values, workgroups across.
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = index_of(wg, li, p.a.y, 256u);
    if i < p.a.x {
        y[i] = a[i] + b[i];
    }
}

//@ upsample
// Twice the size, trilinear, sampling at voxel centres. a: input size,
// channels. b.x: workgroups across.
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

// The two source indices and the weight of the second, along one axis.
fn tap(o: u32, n: u32) -> vec3<f32> {
    let src = max((f32(o) + 0.5) / 2.0 - 0.5, 0.0);
    let i0 = min(u32(floor(src)), n - 1u);
    let i1 = min(i0 + 1u, n - 1u);
    return vec3<f32>(f32(i0), f32(i1), src - f32(i0));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let ins = p.a.xyz;
    let outs = ins * 2u;
    let voxels = outs.x * outs.y * outs.z;
    let i = index_of(wg, li, p.b.x, 256u);
    if i >= voxels * p.a.w {
        return;
    }
    let c = i / voxels;
    let v = i % voxels;
    let tx = tap(v / (outs.y * outs.z), ins.x);
    let ty = tap((v / outs.z) % outs.y, ins.y);
    let tz = tap(v % outs.z, ins.z);
    let base = c * ins.x * ins.y * ins.z;
    var acc = 0.0;
    for (var n = 0u; n < 8u; n++) {
        let ix = u32(select(tx.x, tx.y, (n & 4u) != 0u));
        let iy = u32(select(ty.x, ty.y, (n & 2u) != 0u));
        let iz = u32(select(tz.x, tz.y, (n & 1u) != 0u));
        let wx = select(1.0 - tx.z, tx.z, (n & 4u) != 0u);
        let wy = select(1.0 - ty.z, ty.z, (n & 2u) != 0u);
        let wz = select(1.0 - tz.z, tz.z, (n & 1u) != 0u);
        acc += wx * wy * wz * x[base + (ix * ins.y + iy) * ins.z + iz];
    }
    y[i] = acc;
}

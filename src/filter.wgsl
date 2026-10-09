// Filters run once per frame at the image's own resolution, before window
// and level. Denoising is a bilateral filter: a mean of the neighbours,
// weighted by distance and by how close their values are, so edges stay.
// Sharpening is unsharp masking: a Gaussian blur across, then down, then
// the image plus some of the detail the blur took out, with detail near
// the noise level mostly left alone. Colour images are judged and sharpened
// by brightness. `src/filter.rs` does the same on the CPU for videos.

struct Params {
    // Denoise: spatial sigma in pixels, taps each side of the centre, range
    // sigma in texture units, unused. Sharpen: blur sigma in pixels, taps
    // each side of the centre, amount, noise threshold in texture units.
    p: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var source: texture_2d<f32>;
// The blur across, read by the last sharpening pass. The other passes bind
// the source here too, since they may draw into this texture.
@group(0) @binding(2) var across: texture_2d<f32>;

@vertex
fn vs_full(@builtin(vertex_index) v: u32) -> @builtin(position) vec4<f32> {
    // One triangle that covers the whole target: (-1,-1) (3,-1) (-1,3).
    let x = f32((v << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(v & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

// Pixels beyond the edge repeat the edge. Both textures are the image's
// size. (One function per texture: not every backend takes textures as
// function arguments.)
fn clamped(p: vec2<i32>) -> vec2<i32> {
    let size = vec2<i32>(textureDimensions(source));
    return clamp(p, vec2<i32>(0), size - vec2<i32>(1));
}

fn load_source(p: vec2<i32>) -> vec4<f32> {
    return textureLoad(source, clamped(p), 0);
}

fn load_across(p: vec2<i32>) -> vec4<f32> {
    return textureLoad(across, clamped(p), 0);
}

@fragment
fn fs_denoise_gray(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return denoise(pos, false);
}

@fragment
fn fs_denoise_colour(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return denoise(pos, true);
}

fn denoise(pos: vec4<f32>, colour: bool) -> vec4<f32> {
    let p = vec2<i32>(pos.xy);
    let spatial = params.p.x;
    let r = i32(params.p.y);
    let range = params.p.z;
    let c = load_source(p);
    if range <= 0.0 {
        return c;
    }
    let to_spatial = -0.5 / (spatial * spatial);
    let to_range = -0.5 / (range * range);
    var acc = vec3<f32>(0.0);
    var total = 0.0;
    for (var dy = -r; dy <= r; dy++) {
        for (var dx = -r; dx <= r; dx++) {
            let q = load_source(p + vec2<i32>(dx, dy)).rgb;
            let d = q - c.rgb;
            // Colour difference is the mean square over the channels, so a
            // grey image in RGB filters as it would in greyscale.
            let d2 = select(d.r * d.r, dot(d, d) / 3.0, colour);
            let w = exp(f32(dx * dx + dy * dy) * to_spatial + d2 * to_range);
            acc += w * q;
            total += w;
        }
    }
    return vec4<f32>(acc / total, 1.0);
}

fn weight(i: i32) -> f32 {
    let s = params.p.x;
    return exp(-f32(i * i) / (2.0 * s * s));
}

@fragment
fn fs_across(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(pos.xy);
    let r = i32(params.p.y);
    var acc = vec4<f32>(0.0);
    var total = 0.0;
    for (var i = -r; i <= r; i++) {
        let w = weight(i);
        acc += w * load_source(p + vec2<i32>(i, 0));
        total += w;
    }
    return acc / total;
}

// Soft coring: detail at the threshold keeps half its size, detail well
// above it passes whole and detail well below it fades to nothing. The
// ratio is capped so that the fourth power cannot overflow.
fn core(d: f32) -> f32 {
    let t = params.p.w;
    if t <= 0.0 {
        return d;
    }
    let x = min(abs(d), 1e4 * t) / t;
    let x4 = (x * x) * (x * x);
    return d * x4 / (x4 + 1.0);
}

@fragment
fn fs_down_gray(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return down(pos, false);
}

@fragment
fn fs_down_colour(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return down(pos, true);
}

fn down(pos: vec4<f32>, colour: bool) -> vec4<f32> {
    let p = vec2<i32>(pos.xy);
    let r = i32(params.p.y);
    var blur = vec4<f32>(0.0);
    var total = 0.0;
    for (var i = -r; i <= r; i++) {
        let w = weight(i);
        blur += w * load_across(p + vec2<i32>(0, i));
        total += w;
    }
    let v = load_source(p);
    let d = v - blur / total;
    if !colour {
        return vec4<f32>(v.r + params.p.z * core(d.r), 0.0, 0.0, 1.0);
    }
    // Each channel gains the same, so edges between colours keep their
    // hues and gain no fringes.
    let detail = dot(d.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    return vec4<f32>(v.rgb + params.p.z * core(detail), 1.0);
}

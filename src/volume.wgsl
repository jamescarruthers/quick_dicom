// The 3D view: every slice of the volume is drawn as a quad, farthest
// first, into an offscreen target; a final pass puts the result on screen.
// Dark voxels are transparent and bright ones opaque, so the window and
// level decide what can be seen through.

struct Uniforms {
    // Clip-space x and y of a model point (X, Y, Z, 1), as two dot products.
    mx: vec4<f32>,
    my: vec4<f32>,
    // Width, height, layer count, current layer.
    dims: vec4<f32>,
    // Gap between slices (already scaled by the transition), the layer that
    // sits at Z = 0, transition progress, 1 to draw the last layer first.
    slices: vec4<f32>,
    // Window centre, window width, invert (0 or 1), colour map.
    wl: vec4<f32>,
    // Density, mode (0 blend, 1 maximum intensity), sRGB target, unused.
    look: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var volume: texture_2d_array<f32>;
@group(0) @binding(2) var volume_sampler: sampler;

struct SliceOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) layer: i32,
};

@vertex
fn vs_slice(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> SliceOut {
    let layers = u32(u.dims.z);
    let layer = select(i, layers - 1u - i, u.slices.w > 0.5);
    let c = vec2<f32>(f32(v & 1u), f32((v >> 1u) & 1u));
    let p = vec4<f32>(
        (c.x - 0.5) * u.dims.x,
        (c.y - 0.5) * u.dims.y,
        (f32(layer) - u.slices.y) * u.slices.x,
        1.0,
    );
    var out: SliceOut;
    out.pos = vec4<f32>(dot(u.mx, p), dot(u.my, p), 0.5, 1.0);
    out.uv = c;
    out.layer = i32(layer);
    return out;
}

@fragment
fn fs_slice(in: SliceOut) -> @location(0) vec4<f32> {
    let v = textureSample(volume, volume_sampler, in.uv, in.layer).r;
    let width = max(u.wl.y, 1e-6);
    var g = clamp((v - (u.wl.x - 0.5 * width)) / width, 0.0, 1.0);
    let current = in.layer == i32(u.dims.w);
    let t = u.slices.z;

    if u.look.y > 0.5 {
        // Maximum intensity: the blend keeps the largest value; inversion
        // and colour are applied in the final pass.
        let k = select(t, 1.0, current);
        return vec4<f32>(g * k);
    }

    if u.wl.z > 0.5 {
        g = 1.0 - g;
    }
    let rgb = colormap(g, u.wl.w);
    // Opacity grows with brightness: black is clear, white is solid. The
    // exponent spreads the density over the number of slices.
    let full = 1.0 - pow(max(1.0 - g, 1e-6), u.look.x / u.dims.z);
    // During the transition the current slice starts at full brightness and
    // the others fade in. Nothing blocks the view until the stack has
    // spread, so the current slice's black never darkens what lies behind.
    let glow = select(full * t, mix(1.0, full, t), current);
    return vec4<f32>(rgb * glow, full * t);
}

// The final pass shares the uniforms; its bind group holds the offscreen
// image at binding 3 instead of the volume and sampler.
@group(0) @binding(3) var offscreen: texture_2d<f32>;

struct CompositeOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_composite(@builtin(vertex_index) v: u32) -> CompositeOut {
    let k = vec2<f32>(f32(v & 1u), f32((v >> 1u) & 1u));
    var out: CompositeOut;
    out.pos = vec4<f32>(k.x * 2.0 - 1.0, 1.0 - k.y * 2.0, 0.0, 1.0);
    out.uv = k;
    return out;
}

@fragment
fn fs_composite(in: CompositeOut) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(offscreen));
    let p = clamp(vec2<i32>(in.uv * size), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
    let s = textureLoad(offscreen, p, 0);
    var rgb: vec3<f32>;
    if u.look.y > 0.5 {
        var g = s.r;
        if u.wl.z > 0.5 {
            g = 1.0 - g;
        }
        rgb = colormap(g, u.wl.w);
    } else {
        // Premultiplied colour over a black background.
        rgb = s.rgb;
    }
    if u.look.z > 0.5 {
        rgb = srgb_to_linear(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)));
    }
    return vec4<f32>(rgb, 1.0);
}

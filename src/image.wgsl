// Draws one DICOM frame. Raw modality values live in the texture; window,
// level, local contrast, inversion and filtering all happen here, so
// changing them never touches the CPU copy.

struct Uniforms {
    // Image corners in clip space: x0, y0 (top-left), x1, y1 (bottom-right).
    rect: vec4<f32>,
    // Texture width, height, then local contrast tiles across and down (0
    // when local contrast is off).
    tex: vec4<f32>,
    // Window centre, window width, value scale, invert (0 or 1).
    wl: vec4<f32>,
    // Colour (0 or 1), smooth (0 or 1), sRGB target (0 or 1), colour map.
    mode: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var img: texture_2d<f32>;
// Local contrast curves: one row of 256 per tile, row after row of tiles.
@group(0) @binding(2) var curves: texture_2d<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    // Triangle strip: (0,0) (1,0) (0,1) (1,1).
    let c = vec2<f32>(f32(i & 1u), f32((i >> 1u) & 1u));
    var out: VsOut;
    out.pos = vec4<f32>(mix(u.rect.xy, u.rect.zw, c), 0.0, 1.0);
    out.uv = c;
    return out;
}

fn fetch(p: vec2<i32>) -> vec3<f32> {
    let size = vec2<i32>(u.tex.xy);
    let s = textureLoad(img, clamp(p, vec2<i32>(0), size - vec2<i32>(1)), 0);
    return select(vec3<f32>(s.r), s.rgb, u.mode.x > 0.5);
}

// Bilinear filtering by hand: 32-bit float textures are not filterable on
// every GPU, but they can always be loaded.
fn bilinear(t: vec2<f32>) -> vec3<f32> {
    let f = t - 0.5;
    let i = vec2<i32>(floor(f));
    let w = fract(f);
    let a = fetch(i);
    let b = fetch(i + vec2<i32>(1, 0));
    let c = fetch(i + vec2<i32>(0, 1));
    let d = fetch(i + vec2<i32>(1, 1));
    return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}

// One tile's curve at brightness g, between its two nearest steps.
fn curve(tile: vec2<i32>, g: f32) -> f32 {
    let row = tile.y * i32(u.tex.z) + tile.x;
    let b = clamp(g * 256.0 - 0.5, 0.0, 255.0);
    let i = i32(floor(b));
    let lo = textureLoad(curves, vec2<i32>(i, row), 0).r;
    let hi = textureLoad(curves, vec2<i32>(min(i + 1, 255), row), 0).r;
    return mix(lo, hi, b - f32(i));
}

// Local contrast at texel position t: the four nearest tiles' curves,
// blended by distance, as `Curves::map` does in src/contrast.rs.
fn local_contrast(g: f32, t: vec2<f32>) -> f32 {
    let grid = u.tex.zw;
    let p = clamp(t / u.tex.xy * grid - 0.5, vec2<f32>(0.0), grid - 1.0);
    let i = vec2<i32>(floor(p));
    let f = p - vec2<f32>(i);
    let j = min(i + vec2<i32>(1), vec2<i32>(grid) - vec2<i32>(1));
    let top = mix(curve(i, g), curve(vec2<i32>(j.x, i.y), g), f.x);
    let bottom = mix(curve(vec2<i32>(i.x, j.y), g), curve(j, g), f.x);
    return mix(top, bottom, f.y);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let t = in.uv * u.tex.xy;
    let dx = dpdx(t);
    let dy = dpdy(t);

    var v: vec3<f32>;
    if u.mode.y < 0.5 {
        v = fetch(vec2<i32>(floor(t)));
    } else {
        // When zoomed out, average several samples per screen pixel so fine
        // detail does not alias.
        // The 1% margin keeps rounding at 100% zoom from averaging four
        // samples in some 2x2 blocks of pixels and not others.
        let footprint = max(length(dx), length(dy));
        let n = i32(clamp(ceil(footprint - 0.01), 1.0, 4.0));
        if n == 1 {
            v = bilinear(t);
        } else {
            var acc = vec3<f32>(0.0);
            for (var j = 0; j < n; j++) {
                for (var k = 0; k < n; k++) {
                    let o = (vec2<f32>(f32(k), f32(j)) + 0.5) / f32(n) - 0.5;
                    acc += bilinear(t + dx * o.x + dy * o.y);
                }
            }
            v = acc / f32(n * n);
        }
    }

    let x = v * u.wl.z;
    let width = max(u.wl.y, 1e-6);
    var g = clamp((x - (u.wl.x - 0.5 * width)) / width, vec3<f32>(0.0), vec3<f32>(1.0));
    if u.tex.z > 0.5 {
        if u.mode.x > 0.5 {
            // Colour: change only the brightness of grey parts, adding the
            // same to each channel. Colour such as Doppler stays as it is.
            let lum = dot(g, vec3<f32>(0.2126, 0.7152, 0.0722));
            let spread = max(g.r, max(g.g, g.b)) - min(g.r, min(g.g, g.b));
            let grey = clamp(1.0 - 10.0 * spread, 0.0, 1.0);
            let shift = (local_contrast(lum, t) - lum) * grey;
            g = clamp(g + shift, vec3<f32>(0.0), vec3<f32>(1.0));
        } else {
            g = vec3<f32>(local_contrast(g.r, t));
        }
    }
    if u.wl.w > 0.5 {
        g = 1.0 - g;
    }
    if u.mode.x < 0.5 {
        g = colormap(g.r, u.mode.w);
    }
    if u.mode.z > 0.5 {
        g = srgb_to_linear(g);
    }
    return vec4<f32>(g, 1.0);
}

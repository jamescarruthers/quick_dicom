// The 3D view: a ray from every pixel of an offscreen image marches through
// the volume, a 3D texture, front to back. Window and level map each sample
// to a brightness; bright samples are opaque and dark ones clear, so the
// window decides what can be seen through. A final pass puts the image on
// screen.
//
// Model space is in volume pixels: X right, Y down, Z from slice to slice,
// with the centre of rotation at the origin.

struct Uniforms {
    // The screen's right, down and into-the-screen directions in model space.
    right: vec4<f32>,
    down: vec4<f32>,
    into: vec4<f32>,
    // Model units to clip space: x scale, y scale, x and y of the centre.
    view: vec4<f32>,
    // Width, height, layer count, current layer.
    dims: vec4<f32>,
    // Gap between layers (already scaled by the transition), the layer at
    // Z = 0, transition progress, samples per voxel.
    slices: vec4<f32>,
    // Window centre, window width, invert (0 or 1), colour map.
    wl: vec4<f32>,
    // Density, mode (0 blend, 1 maximum intensity), sRGB target, shading.
    look: vec4<f32>,
    // Segmentation labels shown (0 or 1), every how many label slices the
    // volume took one, a version of which labels are hidden, and flags:
    // 1 for smooth surfaces, 2 for labels above 255.
    labels: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var volume: texture_3d<f32>;
@group(0) @binding(2) var volume_sampler: sampler;
// Segmentation labels at the scan's full resolution, 0 for none, each
// with three distances across its slice, in eighths of a pixel up to FAR:
// inside its own region, from the structure in the voxel above, and from
// the one below, where those differ (see `surface.rs`). One byte each
// when labels fit in a byte; else the label and the first distance in
// the first word, 16 bits each, the others in the second.
@group(1) @binding(0) var label_map: texture_3d<u32>;
// Which labels are shown: label l at (l % 256, l / 256), 0 for hidden.
@group(1) @binding(1) var label_shown: texture_2d<u32>;

struct RayIn {
    @builtin(position) pos: vec4<f32>,
    @location(0) clip: vec2<f32>,
};

@vertex
fn vs_ray(@builtin(vertex_index) v: u32) -> RayIn {
    // One triangle that covers the whole target: (-1,-1) (3,-1) (-1,3).
    let p = vec2<f32>(f32((v << 1u) & 2u), f32(v & 2u)) * 2.0 - 1.0;
    var out: RayIn;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.clip = p;
    return out;
}

fn gap() -> f32 {
    return max(u.slices.x, 1e-4);
}

// Texture coordinates of a model point.
fn to_texture(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        p.x / u.dims.x + 0.5,
        p.y / u.dims.y + 0.5,
        (p.z / gap() + u.slices.y + 0.5) / u.dims.z,
    );
}

fn value(p: vec3<f32>) -> f32 {
    return textureSampleLevel(volume, volume_sampler, to_texture(p), 0.0).r;
}

// Twenty-four pixels, as far as distances go (`surface::FAR`).
const FAR: u32 = 192u;
// A smooth surface's slope is measured this many label pixels each way:
// enough to even out the pixel steps in each slice's outline.
const SLOPE_REACH: f32 = 4.0;

// The labels cover the same box as the volume, at their own resolution,
// and the volume may hold only every so many of their slices. This is a
// model point's place among the label voxels, with voxel centres at whole
// numbers: x, y and slice.
fn label_grid(p: vec3<f32>) -> vec3<f32> {
    let size = vec3<f32>(textureDimensions(label_map));
    let tex = to_texture(p);
    let layer = tex.z * u.dims.z - 0.5;
    return vec3<f32>(tex.x * size.x - 0.5, tex.y * size.y - 0.5, layer * u.labels.y);
}

fn smooth_surfaces() -> bool {
    return (u32(u.labels.w) & 1u) != 0u;
}

struct Voxel {
    id: u32,
    // Inside its own region, from the structure above, from the one below.
    d: vec3<f32>,
}

// A label voxel; beyond the labels, background far from anything.
fn label_voxel(q: vec3<i32>) -> Voxel {
    let size = vec3<i32>(textureDimensions(label_map));
    if any(q < vec3<i32>(0)) || any(q >= size) {
        return Voxel(0u, vec3<f32>(f32(FAR), 0.0, 0.0));
    }
    let v = textureLoad(label_map, q, 0);
    if (u32(u.labels.w) & 2u) != 0u {
        let d = vec3<u32>(v.x >> 16u, v.y & 0xffu, (v.y >> 8u) & 0xffu);
        return Voxel(v.x & 0xffffu, vec3<f32>(d));
    }
    let d = vec3<u32>(v.x >> 8u, v.x >> 16u, v.x >> 24u) & vec3<u32>(0xffu);
    return Voxel(v.x & 0xffu, vec3<f32>(d));
}

fn shows(id: u32) -> bool {
    return textureLoad(label_shown, vec2<i32>(i32(id % 256u), i32(id / 256u)), 0).r != 0u;
}

struct Smooth {
    id: u32,
    inside: f32,
}

// The eight label voxels round grid point g, lower slice first, with
// their trilinear weights.
struct Corners {
    v: array<Voxel, 8>,
    w: array<f32, 8>,
}

fn corners(g: vec3<f32>) -> Corners {
    let b = floor(g);
    let f = g - b;
    let q = vec3<i32>(b);
    var c: Corners;
    for (var i = 0; i < 8; i++) {
        let o = vec3<i32>(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        c.v[i] = label_voxel(q + o);
        let t = select(1.0 - f, f, o == vec3<i32>(1));
        c.w[i] = t.x * t.y * t.z;
    }
    return c;
}

// How far inside label a a point lies, from its eight voxels: for their
// own label, their distance inside it; for the label of the voxel across
// the slab from them, their distance from it; for any other, their
// distance inside their own region, against.
fn inside_corners(c: Corners, a: u32) -> f32 {
    var inside = 0.0;
    for (var j = 0; j < 8; j++) {
        let v = c.v[j];
        var d: f32;
        if v.id == a {
            d = v.d.x;
        } else if c.v[j ^ 4].id == a {
            // Corners 0 to 3 are on the lower slice: their partner is above.
            d = -select(v.d.z, v.d.y, j < 4);
        } else {
            d = -v.d.x;
        }
        inside += c.w[j] * d;
    }
    return inside;
}

// The label whose smooth surface holds label grid point g, and how far
// inside it g lies; or 0. Each of the eight voxels round g counts its
// distance for its own label and against every other, weighted as in
// trilinear interpolation, and the label with the largest sum wins if
// the sum is positive. `surface::smooth_label` does the same on the CPU.
fn smooth_label(g: vec3<f32>) -> Smooth {
    let c = corners(g);
    // Most points lie among voxels of one label, or none.
    var same = true;
    for (var i = 1; i < 8; i++) {
        same = same && c.v[i].id == c.v[0].id;
    }
    if same {
        if c.v[0].id == 0u {
            return Smooth(0u, -1.0);
        }
        return Smooth(c.v[0].id, inside_corners(c, c.v[0].id));
    }
    var best = Smooth(0u, -1e30);
    for (var i = 0; i < 8; i++) {
        let a = c.v[i].id;
        var seen = a == 0u;
        for (var j = 0; j < i; j++) {
            seen = seen || c.v[j].id == a;
        }
        if seen {
            continue;
        }
        let inside = inside_corners(c, a);
        if inside > best.inside {
            best = Smooth(a, inside);
        }
    }
    if best.inside <= 0.0 {
        best.id = 0u;
    }
    return best;
}

// The label at a model point, or 0 where there is none or it is hidden:
// with smooth surfaces, the one whose surface holds the point, else the
// nearest voxel's.
fn label_at(p: vec3<f32>) -> u32 {
    let g = label_grid(p);
    var id: u32;
    if smooth_surfaces() {
        id = smooth_label(g).id;
    } else {
        id = label_voxel(vec3<i32>(floor(g + 0.5))).id;
    }
    return select(0u, id, shows(id));
}

// How far inside a label a point lies, for the slope of its surface:
// with smooth surfaces the blended distance, else 1 inside and 0 out.
fn label_depth(p: vec3<f32>, id: u32) -> f32 {
    let g = label_grid(p);
    if smooth_surfaces() {
        return inside_corners(corners(g), id);
    }
    return select(0.0, 1.0, label_voxel(vec3<i32>(floor(g + 0.5))).id == id);
}

// Where a ray, outside label `id` at distance a and inside it at b, meets
// its smooth surface, found by halving the gap.
fn surface_point(origin: vec3<f32>, dir: vec3<f32>, a: f32, b: f32, id: u32) -> vec3<f32> {
    var lo = a;
    var hi = b;
    for (var j = 0; j < 5; j++) {
        let mid = 0.5 * (lo + hi);
        if label_depth(origin + dir * mid, id) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return origin + dir * hi;
}

// Light on a labelled region, from the slope of the region's own edge, so
// organs show their shape whatever the scan looks like there.
fn lit_label(c: vec3<f32>, p: vec3<f32>, dir: vec3<f32>, id: u32) -> vec3<f32> {
    var e = vec3<f32>(1.5, 1.5, 1.5 * gap());
    if smooth_surfaces() {
        // SLOPE_REACH label pixels across, a label slice deep.
        let size = vec2<f32>(textureDimensions(label_map).xy);
        e = vec3<f32>(SLOPE_REACH * u.dims.xy / size, gap() / u.labels.y);
    }
    let grad = vec3<f32>(
        label_depth(p + vec3<f32>(e.x, 0.0, 0.0), id) - label_depth(p - vec3<f32>(e.x, 0.0, 0.0), id),
        label_depth(p + vec3<f32>(0.0, e.y, 0.0), id) - label_depth(p - vec3<f32>(0.0, e.y, 0.0), id),
        label_depth(p + vec3<f32>(0.0, 0.0, e.z), id) - label_depth(p - vec3<f32>(0.0, 0.0, e.z), id),
    ) / e;
    let size = length(grad);
    if size < 1e-6 {
        return c;
    }
    let facing = abs(dot(grad / size, dir));
    return c * (0.35 + 0.65 * facing) + vec3<f32>(0.1 * pow(facing, 8.0));
}

// Brightness through the window, before inversion.
fn level(v: f32) -> f32 {
    let width = max(u.wl.y, 1e-6);
    return clamp((v - (u.wl.x - 0.5 * width)) / width, 0.0, 1.0);
}

fn shown(v: f32) -> f32 {
    let g = level(v);
    return select(g, 1.0 - g, u.wl.z > 0.5);
}

// Light from the viewer, falling on the surface that the gradient of the
// volume implies. Flat regions, where the gradient is only noise, keep
// their colour unlit. The gradient spans two voxels each way: one voxel
// each way gives normals that jump at voxel boundaries, which shows as
// rings on smooth surfaces.
fn lit(c: vec3<f32>, p: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let e = vec3<f32>(2.0, 2.0, 2.0 * gap());
    let grad = vec3<f32>(
        value(p + vec3<f32>(e.x, 0.0, 0.0)) - value(p - vec3<f32>(e.x, 0.0, 0.0)),
        value(p + vec3<f32>(0.0, e.y, 0.0)) - value(p - vec3<f32>(0.0, e.y, 0.0)),
        value(p + vec3<f32>(0.0, 0.0, e.z)) - value(p - vec3<f32>(0.0, 0.0, e.z)),
    ) / (2.0 * e * max(u.wl.y, 1e-6));
    let size = length(grad);
    let n = grad / max(size, 1e-6);
    let facing = abs(dot(n, dir));
    let shaded = c * (0.3 + 0.7 * facing) + vec3<f32>(0.15 * pow(facing, 8.0));
    return mix(c, shaded, smoothstep(0.01, 0.06, size));
}

@fragment
fn fs_ray(in: RayIn) -> @location(0) vec4<f32> {
    let w = u.dims.x;
    let h = u.dims.y;
    let d = u.dims.z;
    let t = u.slices.z;
    let maximum = u.look.y > 0.5;

    // The ray through this pixel, in model space.
    let sx = (in.clip.x - u.view.z) / u.view.x;
    let sy = -(in.clip.y - u.view.w) / u.view.y;
    let origin = sx * u.right.xyz + sy * u.down.xyz;
    let dir = u.into.xyz;

    // Where it enters and leaves the volume's box.
    let lo = vec3<f32>(-0.5 * w, -0.5 * h, (-0.5 - u.slices.y) * gap());
    let hi = vec3<f32>(0.5 * w, 0.5 * h, (d - 0.5 - u.slices.y) * gap());
    let safe = select(dir, vec3<f32>(1e-12), abs(dir) < vec3<f32>(1e-12));
    let a = (lo - origin) / safe;
    let b = (hi - origin) / safe;
    let near = max(max(min(a.x, b.x), min(a.y, b.y)), min(a.z, b.z));
    let far = min(min(max(a.x, b.x), max(a.y, b.y)), max(a.z, b.z));
    if near > far {
        return vec4<f32>(0.0);
    }

    // During the turn into 3D the current slice shows as a plane, fading
    // as the volume comes in.
    let plane_z = (u.dims.w - u.slices.y) * gap();
    var plane_at = 1e30;
    var plane = 0.0;
    if abs(dir.z) > 1e-6 {
        let s = (plane_z - origin.z) / dir.z;
        let q = origin + dir * s;
        if abs(q.x) <= 0.5 * w && abs(q.y) <= 0.5 * h && s >= near - 1e-3 && s <= far + 1e-3 {
            plane_at = s;
            let uvw = vec3<f32>(q.x / w + 0.5, q.y / h + 0.5, (u.dims.w + 0.5) / d);
            plane = textureSampleLevel(volume, volume_sampler, uvw, 0.0).r;
        }
    }

    // One or two samples per voxel along the axis the ray crosses fastest,
    // starting at a fixed random offset for each pixel so the layers of
    // samples do not show as bands.
    let step = 1.0 / (u.slices.w * max(max(abs(dir.x), abs(dir.y)), abs(dir.z) / gap()));
    let count = min(i32(ceil((far - near) / step)), 16384);
    let jitter = fract(sin(dot(in.pos.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    // Opacity per step: density is how opaque the volume's whole thickness
    // would be, seen face on, if it were all at the top of the window.
    let k = u.look.x * step / (d * gap());

    if maximum {
        var m = 0.0;
        var s = near + step * jitter;
        for (var i = 0; i < count; i++) {
            m = max(m, level(value(origin + dir * s)));
            s += step;
        }
        if plane_at < 1e29 {
            m = max(m * t, level(plane));
        } else {
            m *= t;
        }
        // Inversion and colour are applied in the final pass.
        return vec4<f32>(m);
    }

    let labels_on = u.labels.x > 0.5;
    // Labelled tissue is nine tenths opaque per voxel.
    let solid = 1.0 - pow(0.1, 1.0 / u.slices.w);
    // With labels shown, the rest of the scan is a faint surround: in all
    // it lets at least this much light through.
    let surround = 0.75;
    var clear = 1.0;
    var colour = vec3<f32>(0.0);
    var alpha = 0.0;
    var plane_left = plane_at < 1e29;
    let plane_colour = colormap(shown(plane), u.wl.w) * (1.0 - t);
    // The label of the last sample, and the colour of the one the ray is
    // in, lit where the ray entered it.
    var last_id = 0u;
    var label_lit = vec3<f32>(0.0);
    var s = near + step * jitter;
    for (var i = 0; i < count; i++) {
        if plane_left && s >= plane_at {
            colour += (1.0 - alpha) * plane_colour;
            plane_left = false;
        }
        let p = origin + dir * s;
        let g = shown(value(p));
        var opacity = t * (1.0 - pow(max(1.0 - g, 1e-6), k));
        var id = 0u;
        if labels_on {
            id = label_at(p);
            if id > 0u {
                opacity = t * solid;
            } else {
                opacity = min(opacity, 1.0 - surround / clear);
                clear *= 1.0 - opacity;
            }
        }
        if opacity > 0.002 {
            var c: vec3<f32>;
            if id > 0u {
                if id != last_id {
                    label_lit = label_colour(id);
                    if u.look.w > 0.5 {
                        // A smooth surface is lit where the ray meets it,
                        // not at the sample past it, so the light does
                        // not vary with where the samples fall.
                        var at = p;
                        if smooth_surfaces() && i > 0 {
                            at = surface_point(origin, dir, s - step, s, id);
                        }
                        label_lit = lit_label(label_lit, at, dir, id);
                    }
                }
                c = label_lit;
            } else {
                c = colormap(g, u.wl.w);
                if u.look.w > 0.5 {
                    c = lit(c, p, dir);
                }
            }
            colour += (1.0 - alpha) * opacity * c;
            alpha += (1.0 - alpha) * opacity;
            if alpha > 0.99 {
                break;
            }
        }
        last_id = id;
        s += step;
    }
    if plane_left {
        colour += (1.0 - alpha) * plane_colour;
    }
    return vec4<f32>(colour, alpha);
}

// The final pass shares the uniforms; its bind group holds the offscreen
// image at binding 3, which may be smaller than the screen while the view
// moves.
@group(0) @binding(3) var offscreen: texture_2d<f32>;
@group(0) @binding(4) var offscreen_sampler: sampler;

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
    let s = textureSampleLevel(offscreen, offscreen_sampler, in.uv, 0.0);
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

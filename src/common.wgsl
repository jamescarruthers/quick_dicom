// Shared by the 2D and 3D shaders. `colormap` and `label_colour` must match
// src/colormap.rs.

fn colormap(g_in: f32, kind: f32) -> vec3<f32> {
    let g = clamp(g_in, 0.0, 1.0);
    if kind < 0.5 {
        return vec3<f32>(g);
    }
    if kind < 1.5 {
        // Hot iron: black, red, yellow, white.
        return clamp(vec3<f32>(g * 3.0, g * 3.0 - 1.0, g * 3.0 - 2.0), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    // Rainbow: eight evenly spaced stops from black to white.
    var stops = array<vec3<f32>, 8>(
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.5, 0.0, 1.0),
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 1.0, 1.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(1.0, 1.0, 1.0),
    );
    let x = g * 7.0;
    let i = min(i32(floor(x)), 6);
    return mix(stops[i], stops[i + 1], x - f32(i));
}

// The colour of a segmentation label: hues a golden-ratio turn apart.
fn label_colour(id: u32) -> vec3<f32> {
    let h = fract(f32(id) * 0.618034);
    let k = (vec3<f32>(5.0, 3.0, 1.0) + h * 6.0) % vec3<f32>(6.0);
    return 1.0 - 0.65 * clamp(min(k, 4.0 - k), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

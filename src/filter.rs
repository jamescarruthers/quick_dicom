//! Filters that run at the image's own resolution, before window and level:
//! denoising, with an edge-preserving bilateral filter, and sharpening, by
//! unsharp masking. Both scale to each image's measured noise, so they need
//! no tuning. Local contrast, which works on the windowed image, is in
//! `contrast.rs`.
//!
//! The viewer runs these filters on the GPU (`GpuFilters` and
//! `src/filter.wgsl`), once per frame. `apply` is the CPU twin, used for
//! videos; keep the two in step.

use eframe::egui_wgpu::wgpu;

use crate::decode::{Frame, Pixels};

/// How strongly a filter works.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Level {
    #[default]
    Off,
    Low,
    Medium,
    High,
}

impl Level {
    pub const ALL: [Level; 4] = [Level::Off, Level::Low, Level::Medium, Level::High];

    pub fn name(self) -> &'static str {
        match self {
            Level::Off => "Off",
            Level::Low => "Low",
            Level::Medium => "Medium",
            Level::High => "High",
        }
    }

    pub fn next(self) -> Self {
        let i = Self::ALL.iter().position(|&s| s == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }

    /// Denoising: the spatial sigma in pixels, and the range sigma in
    /// multiples of the noise.
    fn denoise(self) -> Option<(f32, f32)> {
        match self {
            Level::Off => None,
            Level::Low => Some((1.0, 1.5)),
            Level::Medium => Some((1.5, 2.0)),
            Level::High => Some((2.5, 2.5)),
        }
    }

    /// Sharpening: how much of the detail is added back.
    fn sharpen(self) -> Option<f32> {
        match self {
            Level::Off => None,
            Level::Low => Some(0.6),
            Level::Medium => Some(1.2),
            Level::High => Some(2.5),
        }
    }
}

/// The filters chosen in the viewer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Filters {
    pub denoise: Level,
    pub sharpen: Level,
    /// Local contrast; see `contrast.rs`.
    pub contrast: Level,
}

impl Filters {
    /// Whether any filter that runs at image size is on.
    pub fn at_image_size(self) -> bool {
        self.denoise != Level::Off || self.sharpen != Level::Off
    }
}

/// BT.709 weights of red, green and blue in brightness.
pub const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Widest sharpening blur, in image pixels.
const MAX_SIGMA: f32 = 8.0;
/// Detail this many times the noise level gets half the full boost.
const NOISE_FACTOR: f32 = 2.0;

/// Width (standard deviation) of the sharpening blur, in image pixels, for
/// an image drawn at `scale` screen pixels per image pixel. It is one screen
/// pixel, but never less than one image pixel, so the effect shows at any
/// zoom. It moves in half-octave steps, so zooming seldom redoes the work.
pub fn sigma_for(scale: f32) -> f32 {
    let screen_pixel = (1.0 / scale.max(1e-3)).max(1.0);
    let steps = (screen_pixel.log2() * 2.0).round();
    2f32.powf(steps / 2.0).min(MAX_SIGMA)
}

/// Taps each side of the centre for sharpening: the kernel reaches three
/// sigmas.
fn radius(sigma: f32) -> usize {
    (3.0 * sigma).ceil() as usize
}

/// Taps each side of the centre for denoising: two sigmas, as the weights
/// beyond are small and every tap costs a whole row.
fn denoise_radius(spatial: f32) -> usize {
    (2.0 * spatial).ceil() as usize
}

/// Standard deviation of the pixel noise, estimated from the median
/// difference between horizontal neighbours, over a grid of at most 256 by
/// 256 pairs. Equal pairs are skipped, so flat padding round the anatomy
/// does not pass for a noiseless image.
pub fn noise(frame: &Frame) -> f32 {
    let (w, h) = (frame.width as usize, frame.height as usize);
    if w < 2 || h == 0 {
        return 0.0;
    }
    let value = |i: usize| match &frame.pixels {
        Pixels::Gray(v) => v[i],
        Pixels::Rgba(v) => (0..3).map(|c| LUMA[c] * v[i * 4 + c] as f32).sum(),
    };
    let mut diffs = Vec::with_capacity(256 * 256);
    for y in (0..h).step_by(h.div_ceil(256)) {
        for x in (0..w - 1).step_by((w - 1).div_ceil(256)) {
            let i = y * w + x;
            let d = (value(i + 1) - value(i)).abs();
            if d > 0.0 && d.is_finite() {
                diffs.push(d);
            }
        }
    }
    if diffs.is_empty() {
        return 0.0;
    }
    let mid = diffs.len() / 2;
    let (_, median, _) = diffs.select_nth_unstable_by(mid, f32::total_cmp);
    // For Gaussian noise the difference of two pixels has a standard
    // deviation of sigma times root 2, and the median of its size is 0.6745
    // times that.
    *median / (0.6745 * std::f32::consts::SQRT_2)
}

/// Denoise and then sharpen a frame in place, as the GPU does. `sigma` is
/// the sharpening blur, in image pixels.
pub fn apply(frame: &mut Frame, filters: Filters, sigma: f32) {
    if !filters.at_image_size() {
        return;
    }
    let noise = noise(frame);
    if let Some((spatial, range)) = filters.denoise.denoise() {
        denoise(frame, spatial, range * noise);
    }
    if let Some(amount) = filters.sharpen.sharpen() {
        sharpen(frame, amount, sigma, NOISE_FACTOR * noise);
    }
}

/// Bilateral filter: each pixel becomes a weighted mean of its neighbours,
/// weighted both by distance (`spatial`, in pixels) and by how close their
/// value is to its own (`range`, in the frame's units). Neighbours across an
/// edge differ by much more than the noise, so they count for little.
fn denoise(frame: &mut Frame, spatial: f32, range: f32) {
    if range <= 0.0 {
        return;
    }
    let (w, h) = (frame.width as i64, frame.height as i64);
    let r = denoise_radius(spatial) as i64;
    let (to_spatial, to_range) = (-0.5 / (spatial * spatial), -0.5 / (range * range));
    let at = |x: i64, y: i64| (y.clamp(0, h - 1) * w + x.clamp(0, w - 1)) as usize;
    match &mut frame.pixels {
        Pixels::Gray(v) => {
            let src = v.clone();
            for y in 0..h {
                for x in 0..w {
                    let c = src[at(x, y)];
                    let (mut acc, mut total) = (0.0, 0.0);
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let q = src[at(x + dx, y + dy)];
                            let d = q - c;
                            let wt =
                                ((dx * dx + dy * dy) as f32 * to_spatial + d * d * to_range).exp();
                            acc += wt * q;
                            total += wt;
                        }
                    }
                    v[at(x, y)] = acc / total;
                }
            }
        }
        Pixels::Rgba(v) => {
            // Colour difference is the mean square over the three channels,
            // so a grey image in RGB filters as it would in greyscale.
            let src = v.clone();
            let px = |i: usize| [0, 1, 2].map(|c| src[i * 4 + c] as f32);
            for y in 0..h {
                for x in 0..w {
                    let c = px(at(x, y));
                    let (mut acc, mut total) = ([0.0f32; 3], 0.0);
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let q = px(at(x + dx, y + dy));
                            let d2 = (0..3).map(|k| (q[k] - c[k]).powi(2)).sum::<f32>() / 3.0;
                            let wt =
                                ((dx * dx + dy * dy) as f32 * to_spatial + d2 * to_range).exp();
                            (0..3).for_each(|k| acc[k] += wt * q[k]);
                            total += wt;
                        }
                    }
                    let i = at(x, y);
                    for k in 0..3 {
                        v[i * 4 + k] = (acc[k] / total).round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    }
}

/// Unsharp masking: add back `amount` times the detail that a Gaussian blur
/// of `sigma` pixels takes out, cored against `threshold`.
fn sharpen(frame: &mut Frame, amount: f32, sigma: f32, threshold: f32) {
    let (w, h) = (frame.width as usize, frame.height as usize);
    let weights = kernel(sigma);
    match &mut frame.pixels {
        Pixels::Gray(v) => {
            let blurred = blur(v, w, h, &weights);
            for (x, b) in v.iter_mut().zip(blurred) {
                *x += amount * core(*x - b, threshold);
            }
        }
        Pixels::Rgba(v) => {
            // Only brightness is sharpened: each channel gains the same, so
            // edges between colours keep their hues and gain no fringes.
            let blurred: Vec<Vec<f32>> = (0..3)
                .map(|c| {
                    let plane: Vec<f32> = v.iter().skip(c).step_by(4).map(|&x| x as f32).collect();
                    blur(&plane, w, h, &weights)
                })
                .collect();
            for (i, px) in v.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let detail: f32 = (0..3)
                    .map(|c| LUMA[c] * (px[c] as f32 - blurred[c][i]))
                    .sum();
                let add = amount * core(detail, threshold);
                for c in &mut px[..3] {
                    *c = (*c as f32 + add).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
}

/// Gaussian weights from the centre outwards, summing to 1 over both sides.
fn kernel(sigma: f32) -> Vec<f32> {
    let r = radius(sigma);
    let mut w: Vec<f32> = (0..=r)
        .map(|i| (-((i * i) as f32) / (2.0 * sigma * sigma)).exp())
        .collect();
    let total = w[0] + 2.0 * w[1..].iter().sum::<f32>();
    w.iter_mut().for_each(|v| *v /= total);
    w
}

/// Gaussian blur of one channel, across and then down. Pixels beyond the
/// edge repeat the edge, as the GPU's clamped loads do.
fn blur(v: &[f32], w: usize, h: usize, weights: &[f32]) -> Vec<f32> {
    let mut across = vec![0.0f32; w * h];
    for y in 0..h {
        let row = &v[y * w..(y + 1) * w];
        for x in 0..w {
            let mut acc = weights[0] * row[x];
            for (k, &wk) in weights.iter().enumerate().skip(1) {
                acc += wk * (row[x.saturating_sub(k)] + row[(x + k).min(w - 1)]);
            }
            across[y * w + x] = acc;
        }
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = weights[0] * across[y * w + x];
            for (k, &wk) in weights.iter().enumerate().skip(1) {
                let up = y.saturating_sub(k);
                let down = (y + k).min(h - 1);
                acc += wk * (across[up * w + x] + across[down * w + x]);
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// Soft coring: detail at the threshold keeps half its size, detail well
/// above it passes whole and detail well below it fades to nothing. The
/// ratio is capped so that the fourth power cannot overflow.
fn core(d: f32, threshold: f32) -> f32 {
    if threshold <= 0.0 {
        return d;
    }
    let x = d.abs().min(1e4 * threshold) / threshold;
    let x4 = (x * x) * (x * x);
    d * x4 / (x4 + 1.0)
}

/// The GPU side: pipelines for both kinds of image, made once.
pub struct GpuFilters {
    layout: wgpu::BindGroupLayout,
    /// Denoising and sharpening run in one submission, so each needs its
    /// own parameters.
    denoise_params: wgpu::Buffer,
    sharpen_params: wgpu::Buffer,
    /// Denoise, blur across, blur down and sharpen.
    gray: [wgpu::RenderPipeline; 3],
    colour: [wgpu::RenderPipeline; 3],
}

/// Textures holding one image's filtered versions, made when first needed
/// and dropped when their filter is turned off.
#[derive(Default)]
pub struct Targets {
    denoised: Option<wgpu::TextureView>,
    across: Option<wgpu::TextureView>,
    sharpened: Option<wgpu::TextureView>,
}

/// Formats for the denoised image, the blur across and the sharpened
/// image. Greyscale keeps 32-bit floats; colour keeps 8 bits, as its source
/// and the videos do.
fn formats(gray: bool) -> [wgpu::TextureFormat; 3] {
    use wgpu::TextureFormat::*;
    if gray {
        [R32Float, R32Float, R32Float]
    } else {
        [Rgba8Unorm, Rgba16Float, Rgba8Unorm]
    }
}

impl GpuFilters {
    /// `None` when the GPU cannot draw into the textures this needs.
    pub fn new(device: &wgpu::Device, adapter: &wgpu::Adapter) -> Option<Self> {
        let drawable = |f: wgpu::TextureFormat| {
            adapter
                .get_texture_format_features(f)
                .allowed_usages
                .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
        };
        if !formats(true)
            .into_iter()
            .chain(formats(false))
            .all(drawable)
        {
            return None;
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("filters"),
            source: wgpu::ShaderSource::Wgsl(include_str!("filter.wgsl").into()),
        });
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("filters"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                texture(1),
                texture(2),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("filters"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str, format| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_full"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let kind = |gray: bool| {
            let [denoised, across, sharpened] = formats(gray);
            let suffix = if gray { "gray" } else { "colour" };
            [
                pipeline(&format!("fs_denoise_{suffix}"), denoised),
                pipeline("fs_across", across),
                pipeline(&format!("fs_down_{suffix}"), sharpened),
            ]
        };
        let params = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Some(GpuFilters {
            gray: kind(true),
            colour: kind(false),
            denoise_params: params("denoise params"),
            sharpen_params: params("sharpen params"),
            layout,
        })
    }

    /// Record the passes that filter `image`, a greyscale (R32Float) or
    /// colour (Rgba8Unorm) texture of `size`. `noise` is in the frame's own
    /// units, as `noise` gives it, and `sigma` is the sharpening blur.
    /// Returns the view that holds the result, or `None` when no filter
    /// here is on.
    #[allow(clippy::too_many_arguments)]
    pub fn run<'t>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        targets: &'t mut Targets,
        image: &'t wgpu::TextureView,
        size: (u32, u32),
        gray: bool,
        filters: Filters,
        sigma: f32,
        noise: f32,
    ) -> Option<&'t wgpu::TextureView> {
        // Colour textures hold 0 to 1, not 0 to 255.
        let noise = if gray { noise } else { noise / 255.0 };
        let [denoised_format, across_format, sharpened_format] = formats(gray);
        let pipelines = if gray { &self.gray } else { &self.colour };
        let denoise = filters.denoise.denoise();
        let sharpen = filters.sharpen.sharpen();
        if denoise.is_none() {
            targets.denoised = None;
        }
        if sharpen.is_none() {
            targets.across = None;
            targets.sharpened = None;
        }

        if let Some((spatial, range)) = denoise {
            let params = [spatial, denoise_radius(spatial) as f32, range * noise, 0.0];
            queue.write_buffer(&self.denoise_params, 0, bytemuck::cast_slice(&params));
            let out = targets
                .denoised
                .get_or_insert_with(|| make_target(device, size, denoised_format));
            self.pass(
                device,
                encoder,
                &pipelines[0],
                &self.denoise_params,
                out,
                image,
                image,
            );
        }
        if let Some(amount) = sharpen {
            let params = [sigma, radius(sigma) as f32, amount, NOISE_FACTOR * noise];
            queue.write_buffer(&self.sharpen_params, 0, bytemuck::cast_slice(&params));
            let source = targets.denoised.as_ref().unwrap_or(image);
            let across = targets
                .across
                .get_or_insert_with(|| make_target(device, size, across_format));
            self.pass(
                device,
                encoder,
                &pipelines[1],
                &self.sharpen_params,
                across,
                source,
                source,
            );
            let out = targets
                .sharpened
                .get_or_insert_with(|| make_target(device, size, sharpened_format));
            self.pass(
                device,
                encoder,
                &pipelines[2],
                &self.sharpen_params,
                out,
                source,
                across,
            );
        }
        targets.sharpened.as_ref().or(targets.denoised.as_ref())
    }

    /// One full-screen pass into `out`, reading `source` and `second`.
    #[allow(clippy::too_many_arguments)]
    fn pass(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        params: &wgpu::Buffer,
        out: &wgpu::TextureView,
        source: &wgpu::TextureView,
        second: &wgpu::TextureView,
    ) {
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("filters"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(second),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("filter"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: out,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// A texture the filters draw into and later read.
fn make_target(
    device: &wgpu::Device,
    size: (u32, u32),
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("filtered"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            // Tests copy results out.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{self, colour, gaussian, gray};

    fn pixels(frame: &Frame) -> &[f32] {
        match &frame.pixels {
            Pixels::Gray(v) => v,
            Pixels::Rgba(_) => panic!("not greyscale"),
        }
    }

    /// Standard deviation of columns `x0..x1` over all rows.
    fn spread(v: &[f32], w: usize, x0: usize, x1: usize) -> f32 {
        let flat: Vec<f32> = v.chunks(w).flat_map(|row| row[x0..x1].to_vec()).collect();
        let mean = flat.iter().sum::<f32>() / flat.len() as f32;
        (flat.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / flat.len() as f32).sqrt()
    }

    /// A step of 100 halfway across, with noise of 5 on both sides.
    fn noisy_step(w: usize, h: usize) -> Vec<f32> {
        let noise = gaussian(w * h, 5.0, 3);
        (0..w * h)
            .map(|i| if i % w < w / 2 { 0.0 } else { 100.0 } + noise[i])
            .collect()
    }

    #[test]
    fn sigma_is_one_screen_pixel_in_half_octaves() {
        assert_eq!(sigma_for(1.0), 1.0);
        assert_eq!(sigma_for(4.0), 1.0);
        assert_eq!(sigma_for(0.5), 2.0);
        assert!((sigma_for(0.35) - 2.828).abs() < 0.01);
        assert_eq!(sigma_for(0.01), MAX_SIGMA);
    }

    #[test]
    fn kernel_sums_to_one() {
        for sigma in [1.0, 1.414, 4.0] {
            let w = kernel(sigma);
            assert_eq!(w.len(), radius(sigma) + 1);
            let total = w[0] + 2.0 * w[1..].iter().sum::<f32>();
            assert!((total - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn estimates_noise_and_ignores_flat_padding() {
        let (w, h) = (300, 200);
        let mut values: Vec<f32> = gaussian(w * h, 12.0, 7).iter().map(|v| v + 40.0).collect();
        let sd = noise(&gray(w as u32, h as u32, values.clone()));
        assert!((sd - 12.0).abs() < 1.0, "estimated {sd}");
        // The left two thirds of each row become padding.
        for row in values.chunks_mut(w) {
            row[..200].fill(-2048.0);
        }
        let sd = noise(&gray(w as u32, h as u32, values));
        assert!((sd - 12.0).abs() < 1.5, "estimated {sd} with padding");
        assert_eq!(noise(&gray(4, 4, vec![5.0; 16])), 0.0);
    }

    #[test]
    fn sharpens_edges_more_than_noise() {
        let (w, h) = (64usize, 64usize);
        let values = noisy_step(w, h);
        let mut frame = gray(w as u32, h as u32, values.clone());
        let t = NOISE_FACTOR * noise(&frame);
        sharpen(&mut frame, 1.2, 1.0, t);
        let out = pixels(&frame);
        // Across the edge the step grows: dark side darker, bright side brighter.
        let col_mean = |v: &[f32], x: usize| (0..h).map(|y| v[y * w + x]).sum::<f32>() / h as f32;
        assert!(col_mean(out, w / 2 - 1) < col_mean(&values, w / 2 - 1) - 10.0);
        assert!(col_mean(out, w / 2) > col_mean(&values, w / 2) + 10.0);
        // Away from the edge the noise grows by about a quarter.
        let (before, after) = (spread(&values, w, 4, 24), spread(out, w, 4, 24));
        assert!(after < before * 1.35, "noise {before} became {after}");
        // Without the threshold the same amount makes it far grainier.
        let mut plain = gray(w as u32, h as u32, values.clone());
        sharpen(&mut plain, 1.2, 1.0, 0.0);
        assert!(spread(pixels(&plain), w, 4, 24) > before * 1.8);
    }

    #[test]
    fn denoising_smooths_noise_but_keeps_edges() {
        let (w, h) = (64usize, 64usize);
        let values = noisy_step(w, h);
        let before = spread(&values, w, 4, 24);
        for (level, most) in [(Level::Low, 0.6), (Level::Medium, 0.4), (Level::High, 0.3)] {
            let mut frame = gray(w as u32, h as u32, values.clone());
            let filters = Filters {
                denoise: level,
                ..Filters::default()
            };
            apply(&mut frame, filters, 1.0);
            let out = pixels(&frame);
            let after = spread(out, w, 4, 24);
            assert!(
                after < before * most,
                "{level:?}: noise {before} became {after}"
            );
            // The step stays a step: the columns either side keep their levels.
            let col_mean = |x: usize| (0..h).map(|y| out[y * w + x]).sum::<f32>() / h as f32;
            assert!(
                col_mean(w / 2 - 1) < 5.0 && col_mean(w / 2) > 95.0,
                "{level:?}"
            );
        }
    }

    #[test]
    fn denoising_keeps_colour_edges() {
        // Orange beside grey of about the same brightness: their colours
        // differ, so neither bleeds into the other.
        let (orange, grey) = ([200u8, 110, 40, 255], [124u8, 124, 124, 255]);
        let rgba: Vec<u8> = (0..64)
            .flat_map(|i| if i % 8 < 4 { orange } else { grey })
            .collect();
        let mut frame = colour(8, 8, rgba.clone());
        // Noise of 2 grey levels, so the range is a few levels wide.
        denoise(&mut frame, 1.5, 4.0);
        let Pixels::Rgba(v) = &frame.pixels else {
            unreachable!()
        };
        assert_eq!(v, &rgba);
    }

    #[test]
    fn flat_images_and_colour() {
        let mut flat = gray(8, 8, vec![42.0; 64]);
        sharpen(&mut flat, 2.5, 2.0, 0.0);
        assert!(pixels(&flat).iter().all(|&x| (x - 42.0).abs() < 1e-3));
        let filters = Filters {
            denoise: Level::High,
            sharpen: Level::High,
            ..Filters::default()
        };
        apply(&mut flat, filters, 1.0);
        assert!(pixels(&flat).iter().all(|&x| (x - 42.0).abs() < 1e-3));

        // A white square on black: its corner pixel overshoots and clips.
        let mut rgba = vec![0u8; 8 * 8 * 4];
        for y in 2..6 {
            for x in 2..6 {
                rgba[(y * 8 + x) * 4..(y * 8 + x) * 4 + 3].fill(200);
            }
        }
        rgba.iter_mut().skip(3).step_by(4).for_each(|a| *a = 255);
        let mut square = colour(8, 8, rgba);
        sharpen(&mut square, 1.2, 1.0, 0.0);
        let Pixels::Rgba(v) = &square.pixels else {
            unreachable!()
        };
        let at = |x: usize, y: usize| v[(y * 8 + x) * 4];
        assert!(at(2, 2) > 200 && at(1, 1) == 0 && at(4, 4) >= 200);
        assert_eq!(v[(2 * 8 + 2) * 4 + 3], 255, "alpha is untouched");

        // An orange square on grey: the grey beside it darkens but stays
        // grey, and the orange keeps its hue.
        let (orange, grey) = ([200u8, 110, 40, 255], [100u8, 100, 100, 255]);
        let rgba: Vec<u8> = (0..64)
            .flat_map(|i| {
                if (2..6).contains(&(i % 8)) && (2..6).contains(&(i / 8)) {
                    orange
                } else {
                    grey
                }
            })
            .collect();
        let mut square = colour(8, 8, rgba);
        sharpen(&mut square, 1.2, 1.0, 0.0);
        let Pixels::Rgba(v) = &square.pixels else {
            unreachable!()
        };
        let px = |x: usize, y: usize| &v[(y * 8 + x) * 4..(y * 8 + x) * 4 + 3];
        let outside = px(1, 3);
        assert!(outside[0] < 100 && outside[0] == outside[1] && outside[1] == outside[2]);
        let inside = px(2, 3);
        let shift = inside[0] as i32 - 200;
        assert!(shift > 0);
        assert_eq!(inside[1] as i32 - 110, shift);
        assert_eq!(inside[2] as i32 - 40, shift);
    }

    /// Runs the GPU passes and checks them against the CPU twin. Skipped
    /// when there is no GPU, as on most CI runners.
    #[test]
    fn gpu_matches_cpu() {
        let Some((adapter, device, queue)) = testing::gpu() else {
            eprintln!("no GPU: skipped");
            return;
        };
        let Some(gpu) = GpuFilters::new(&device, &adapter) else {
            eprintln!("this GPU cannot filter: skipped");
            return;
        };

        let (w, h) = (64u32, 48u32);
        let n = (w * h) as usize;
        let speckle = gaussian(n, 5.0, 11);
        let grey = || {
            let step = |i: usize| if i % 64 > 30 { 1000.0 } else { -200.0 };
            gray(w, h, (0..n).map(|i| step(i) + speckle[i]).collect())
        };
        let checks = || {
            let rgba = (0..n).flat_map(|i| {
                let (x, y) = (i % 64, i / 64);
                let check = if (x / 8 + y / 8) % 2 == 0 { 220 } else { 30 };
                let grain = (speckle[i] * 2.0) as i32;
                let c = |v: i32| (v + grain).clamp(0, 255) as u8;
                [c(x as i32 * 4), c(y as i32 * 5), c(check), 255]
            });
            colour(w, h, rgba.collect())
        };
        let both = |denoise, sharpen| Filters {
            denoise,
            sharpen,
            contrast: Level::Off,
        };
        let cases: [(&dyn Fn() -> Frame, Filters, f32); 6] = [
            (&grey, both(Level::Off, Level::Medium), 1.0),
            (&grey, both(Level::Off, Level::High), 2.828),
            (&grey, both(Level::High, Level::Off), 1.0),
            (&grey, both(Level::Medium, Level::Medium), 1.0),
            (&checks, both(Level::Off, Level::Medium), 1.414),
            (&checks, both(Level::Medium, Level::High), 1.0),
        ];
        for (make, filters, sigma) in cases {
            let mut frame = make();
            let gray = !frame.is_color();
            let image = testing::upload(&device, &queue, &frame);
            let mut targets = Targets::default();
            let mut encoder = device.create_command_encoder(&Default::default());
            let level = noise(&frame);
            let out = gpu
                .run(
                    &device,
                    &queue,
                    &mut encoder,
                    &mut targets,
                    &image,
                    (w, h),
                    gray,
                    filters,
                    sigma,
                    level,
                )
                .expect("a filter is on")
                .texture()
                .clone();
            let got = testing::read(&device, &queue, encoder, &out);

            apply(&mut frame, filters, sigma);
            match &frame.pixels {
                Pixels::Gray(want) => {
                    let got: &[f32] = bytemuck::cast_slice(&got);
                    for (i, (g, c)) in got.iter().zip(want).enumerate() {
                        assert!(
                            (g - c).abs() < 0.05,
                            "{filters:?} pixel {i}: GPU {g}, CPU {c}"
                        );
                    }
                }
                Pixels::Rgba(want) => {
                    for (i, (g, c)) in got.iter().zip(want).enumerate().filter(|(i, _)| i % 4 < 3) {
                        assert!(
                            g.abs_diff(*c) <= 2,
                            "{filters:?} byte {i}: GPU {g}, CPU {c}"
                        );
                    }
                }
            }
        }
    }
}

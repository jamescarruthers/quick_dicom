//! Sharpening by unsharp masking: the image plus some of the detail that a
//! Gaussian blur takes out. Detail no bigger than the image's noise is
//! mostly left alone, so edges sharpen without the grain growing with them.
//!
//! The viewer sharpens on the GPU (`GpuSharpen` and `src/sharpen.wgsl`),
//! once per frame at the image's own resolution. `apply` is the CPU twin,
//! used for videos; keep the two in step.

use eframe::egui_wgpu::wgpu;

use crate::decode::{Frame, Pixels};

/// How strongly to sharpen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Sharpen {
    #[default]
    Off,
    Low,
    Medium,
    High,
}

impl Sharpen {
    pub const ALL: [Sharpen; 4] = [Sharpen::Off, Sharpen::Low, Sharpen::Medium, Sharpen::High];

    pub fn name(self) -> &'static str {
        match self {
            Sharpen::Off => "Off",
            Sharpen::Low => "Low",
            Sharpen::Medium => "Medium",
            Sharpen::High => "High",
        }
    }

    pub fn next(self) -> Self {
        let i = Self::ALL.iter().position(|&s| s == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }

    /// How much of the detail is added back; 0 when off.
    pub fn amount(self) -> f32 {
        match self {
            Sharpen::Off => 0.0,
            Sharpen::Low => 0.6,
            Sharpen::Medium => 1.2,
            Sharpen::High => 2.5,
        }
    }
}

/// BT.709 weights of red, green and blue in brightness.
const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Widest blur, in image pixels.
const MAX_SIGMA: f32 = 8.0;
/// Detail this many times the noise level gets half the full boost.
const NOISE_FACTOR: f32 = 2.0;

/// Width (standard deviation) of the blur, in image pixels, for an image
/// drawn at `scale` screen pixels per image pixel. It is one screen pixel,
/// but never less than one image pixel, so the effect shows at any zoom. It
/// moves in half-octave steps, so zooming seldom redoes the work.
pub fn sigma_for(scale: f32) -> f32 {
    let screen_pixel = (1.0 / scale.max(1e-3)).max(1.0);
    let steps = (screen_pixel.log2() * 2.0).round();
    2f32.powf(steps / 2.0).min(MAX_SIGMA)
}

/// Taps each side of the centre: the kernel reaches three sigmas.
fn radius(sigma: f32) -> usize {
    (3.0 * sigma).ceil() as usize
}

/// Detail smaller than this is mostly left alone. It is in the frame's own
/// units: raw values for greyscale, 0 to 255 for colour.
pub fn threshold(frame: &Frame) -> f32 {
    NOISE_FACTOR * noise(frame)
}

/// Standard deviation of the pixel noise, estimated from the median
/// difference between horizontal neighbours, over a grid of at most 256 by
/// 256 pairs. Equal pairs are skipped, so flat padding round the anatomy
/// does not pass for a noiseless image.
fn noise(frame: &Frame) -> f32 {
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

/// Sharpen a frame in place, as the GPU does. `sigma` is in image pixels
/// and `threshold` in the frame's own units.
pub fn apply(frame: &mut Frame, amount: f32, sigma: f32, threshold: f32) {
    if amount <= 0.0 {
        return;
    }
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
pub struct GpuSharpen {
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    gray: [wgpu::RenderPipeline; 2],
    colour: [wgpu::RenderPipeline; 2],
}

/// Textures for sharpening images of one size and format.
pub struct Target {
    half: wgpu::TextureView,
    /// The sharpened image, in the source's units.
    pub out: wgpu::TextureView,
    across_group: wgpu::BindGroup,
    down_group: wgpu::BindGroup,
    gray: bool,
}

/// Formats for the horizontal blur and for the result. Greyscale keeps
/// 32-bit floats; colour keeps 8 bits, as its source and the videos do.
fn formats(gray: bool) -> [wgpu::TextureFormat; 2] {
    if gray {
        [wgpu::TextureFormat::R32Float, wgpu::TextureFormat::R32Float]
    } else {
        [
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::Rgba8Unorm,
        ]
    }
}

impl GpuSharpen {
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
            label: Some("sharpen"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sharpen.wgsl").into()),
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
            label: Some("sharpen"),
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
            label: Some("sharpen"),
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
        let both = |gray| {
            let [half, out] = formats(gray);
            let down = if gray {
                "fs_down_gray"
            } else {
                "fs_down_colour"
            };
            [pipeline("fs_across", half), pipeline(down, out)]
        };
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sharpen params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(GpuSharpen {
            gray: both(true),
            colour: both(false),
            layout,
            params,
        })
    }

    /// Textures to sharpen `image`, a greyscale (R32Float) or colour
    /// (Rgba8Unorm) texture of `size`.
    pub fn target(
        &self,
        device: &wgpu::Device,
        image: &wgpu::TextureView,
        size: (u32, u32),
        gray: bool,
    ) -> Target {
        let [half_format, out_format] = formats(gray);
        let make = |label, format| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: size.0,
                        height: size.1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let half = make("sharpen across", half_format);
        let out = make("sharpened", out_format);
        let group = |second: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sharpen"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(image),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(second),
                    },
                ],
            })
        };
        // The first pass draws into `half`, so it cannot also read it; it
        // reads the image in both slots.
        let across_group = group(image);
        let down_group = group(&half);
        Target {
            half,
            out,
            across_group,
            down_group,
            gray,
        }
    }

    /// Record the two passes that fill `target.out`. `threshold` is in the
    /// frame's own units, as `threshold` gives it.
    pub fn run(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &Target,
        amount: f32,
        sigma: f32,
        threshold: f32,
    ) {
        // Colour textures hold 0 to 1, not 0 to 255.
        let threshold = if target.gray {
            threshold
        } else {
            threshold / 255.0
        };
        let params = [sigma, radius(sigma) as f32, amount, threshold];
        queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&params));
        let pipelines = if target.gray {
            &self.gray
        } else {
            &self.colour
        };
        for (pipeline, view, group) in [
            (&pipelines[0], &target.half, &target.across_group),
            (&pipelines[1], &target.out, &target.down_group),
        ] {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sharpen"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(width: u32, height: u32, values: Vec<f32>) -> Frame {
        Frame {
            id: 0,
            width,
            height,
            pixels: Pixels::Gray(values),
            min: 0.0,
            max: 0.0,
            window: (0.0, 1.0),
            inverted: false,
            spacing: None,
            position: None,
        }
    }

    /// Repeatable Gaussian noise: xorshift and Box-Muller.
    fn gaussian(n: usize, sd: f32, seed: u64) -> Vec<f32> {
        let mut s = seed;
        let mut uniform = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let (a, b) = (uniform(), uniform());
                ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32 * sd
            })
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
        // A step of 100 with noise of 5 on both sides.
        let (w, h) = (64usize, 64usize);
        let noisy = gaussian(w * h, 5.0, 3);
        let values: Vec<f32> = (0..w * h)
            .map(|i| if i % w < w / 2 { 0.0 } else { 100.0 } + noisy[i])
            .collect();
        let mut frame = gray(w as u32, h as u32, values.clone());
        let t = threshold(&frame);
        apply(&mut frame, 1.2, 1.0, t);
        let Pixels::Gray(out) = &frame.pixels else {
            unreachable!()
        };
        // Across the edge the step grows: dark side darker, bright side brighter.
        let col_mean = |v: &[f32], x: usize| (0..h).map(|y| v[y * w + x]).sum::<f32>() / h as f32;
        assert!(col_mean(out, w / 2 - 1) < col_mean(&values, w / 2 - 1) - 10.0);
        assert!(col_mean(out, w / 2) > col_mean(&values, w / 2) + 10.0);
        // Away from the edge the noise grows by about a quarter.
        let spread = |v: &[f32]| {
            let flat: Vec<f32> = (0..h)
                .flat_map(|y| (4..24).map(move |x| v[y * w + x]))
                .collect();
            let mean = flat.iter().sum::<f32>() / flat.len() as f32;
            (flat.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / flat.len() as f32).sqrt()
        };
        let (before, after) = (spread(&values), spread(out));
        assert!(after < before * 1.35, "noise {before} became {after}");
        // Without the threshold the same amount makes it far grainier.
        let mut plain = gray(w as u32, h as u32, values.clone());
        apply(&mut plain, 1.2, 1.0, 0.0);
        let Pixels::Gray(plain) = &plain.pixels else {
            unreachable!()
        };
        assert!(spread(plain) > before * 1.8);
    }

    #[test]
    fn flat_images_and_colour() {
        let mut flat = gray(8, 8, vec![42.0; 64]);
        apply(&mut flat, 2.5, 2.0, 0.0);
        assert!(
            matches!(&flat.pixels, Pixels::Gray(v) if v.iter().all(|&x| (x - 42.0).abs() < 1e-3))
        );

        // A white square on black: its corner pixel overshoots and clips.
        let mut rgba = vec![0u8; 8 * 8 * 4];
        for y in 2..6 {
            for x in 2..6 {
                rgba[(y * 8 + x) * 4..(y * 8 + x) * 4 + 3].fill(200);
            }
        }
        rgba.iter_mut().skip(3).step_by(4).for_each(|a| *a = 255);
        let mut colour = Frame {
            pixels: Pixels::Rgba(rgba),
            ..gray(8, 8, vec![])
        };
        apply(&mut colour, 1.2, 1.0, 0.0);
        let Pixels::Rgba(v) = &colour.pixels else {
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
        let mut colour = Frame {
            pixels: Pixels::Rgba(rgba),
            ..gray(8, 8, vec![])
        };
        apply(&mut colour, 1.2, 1.0, 0.0);
        let Pixels::Rgba(v) = &colour.pixels else {
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
        let instance = wgpu::Instance::default();
        let options = wgpu::RequestAdapterOptions::default();
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&options)) else {
            eprintln!("no GPU: skipped");
            return;
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("the adapter gives a device");
        let Some(gpu) = GpuSharpen::new(&device, &adapter) else {
            eprintln!("this GPU cannot sharpen: skipped");
            return;
        };

        // 64 pixels across makes rows of 256 bytes, as copies need.
        let (w, h) = (64u32, 48u32);
        let n = (w * h) as usize;
        let noisy = gaussian(n, 5.0, 11);
        let step = |i: usize| if i % 64 > 30 { 1000.0 } else { -200.0 };
        let grey = || Pixels::Gray((0..n).map(|i| step(i) + noisy[i]).collect());
        let colour = Pixels::Rgba(
            (0..n)
                .flat_map(|i| {
                    let (x, y) = (i % 64, i / 64);
                    let check = if (x / 8 + y / 8) % 2 == 0 { 220 } else { 30 };
                    [(x * 4) as u8, (y * 5) as u8, check, 255]
                })
                .collect(),
        );
        for (pixels, amount, sigma) in [
            (grey(), 1.2, 1.0),
            (grey(), 2.5, 2.828),
            (colour, 1.2, 1.414),
        ] {
            let mut frame = Frame {
                pixels,
                ..gray(w, h, vec![])
            };
            let is_gray = !frame.is_color();
            let (format, bytes): (_, Vec<u8>) = match &frame.pixels {
                Pixels::Gray(v) => (
                    wgpu::TextureFormat::R32Float,
                    bytemuck::cast_slice(v).to_vec(),
                ),
                Pixels::Rgba(v) => (wgpu::TextureFormat::Rgba8Unorm, v.clone()),
            };
            let size = wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            };
            let layout = wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            };
            let image = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(image.as_image_copy(), &bytes, layout, size);
            let view = image.create_view(&wgpu::TextureViewDescriptor::default());
            let target = gpu.target(&device, &view, (w, h), is_gray);
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (w * h * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let t = threshold(&frame);
            let mut encoder = device.create_command_encoder(&Default::default());
            gpu.run(&queue, &mut encoder, &target, amount, sigma, t);
            encoder.copy_texture_to_buffer(
                target.out.texture().as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout,
                },
                size,
            );
            queue.submit([encoder.finish()]);
            readback.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("the GPU finishes");
            let got = readback.get_mapped_range(..).unwrap().to_vec();

            apply(&mut frame, amount, sigma, t);
            match &frame.pixels {
                Pixels::Gray(want) => {
                    let got: &[f32] = bytemuck::cast_slice(&got);
                    for (i, (g, c)) in got.iter().zip(want).enumerate() {
                        assert!((g - c).abs() < 0.05, "pixel {i}: GPU {g}, CPU {c}");
                    }
                }
                Pixels::Rgba(want) => {
                    for (i, (g, c)) in got.iter().zip(want).enumerate().filter(|(i, _)| i % 4 < 3) {
                        assert!(g.abs_diff(*c) <= 1, "byte {i}: GPU {g}, CPU {c}");
                    }
                }
            }
        }
    }
}

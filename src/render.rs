//! GPU drawing of the current frame through an egui paint callback.

use std::sync::Arc;

use eframe::egui_wgpu::{self, wgpu};

use crate::contrast::{self, BINS, Curves, MAX_TILES};
use crate::decode::{Frame, Pixels};
use crate::filter::{self, Filters, GpuFilters, Targets};

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub rect: [f32; 4],
    pub tex: [f32; 4],
    pub wl: [f32; 4],
    pub mode: [f32; 4],
}

struct Texture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
    format: wgpu::TextureFormat,
    /// Shows the frame as uploaded.
    plain: wgpu::BindGroup,
    /// The frame's filtered versions.
    targets: Targets,
    /// Shows the filtered frame, when a filter is on.
    filtered: Option<wgpu::BindGroup>,
    /// Filters and sharpening blur that `filtered` was made with, for the
    /// frame now uploaded.
    made: Option<(Filters, f32)>,
}

/// GPU state, stored in egui's callback resources.
pub struct ImageRenderer {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    texture: Option<Texture>,
    uploaded: u64,
    srgb_target: bool,
    /// `None` when the GPU cannot run the filters.
    filters: Option<GpuFilters>,
    /// Noise level of the uploaded frame, worked out when first needed.
    noise: Option<f32>,
    /// Local contrast curves: a row of `BINS` per tile.
    curves: wgpu::Texture,
    curves_view: wgpu::TextureView,
    /// Frame, window and clip limit the curves were made for, and their
    /// tiles across and down.
    curves_made: Option<(u64, [f32; 2], f32)>,
    tiles: (usize, usize),
    /// Draw the filtered frame this time.
    show_filtered: bool,
}

impl ImageRenderer {
    /// Returns whether the GPU can run the denoise and sharpen filters.
    pub fn install(rs: &egui_wgpu::RenderState) -> bool {
        let renderer = Self::new(&rs.device, &rs.adapter, rs.target_format);
        let can_filter = renderer.filters.is_some();
        rs.renderer.write().callback_resources.insert(renderer);
        can_filter
    }

    fn new(device: &wgpu::Device, adapter: &wgpu::Adapter, target: wgpu::TextureFormat) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dicom image"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(include_str!("common.wgsl"), include_str!("image.wgsl")).into(),
            ),
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
            label: Some("dicom image"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
            label: Some("dicom image"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dicom image"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dicom image uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let curves = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("local contrast curves"),
            size: wgpu::Extent3d {
                width: BINS as u32,
                height: (MAX_TILES * MAX_TILES) as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let curves_view = curves.create_view(&wgpu::TextureViewDescriptor::default());

        ImageRenderer {
            pipeline,
            layout,
            uniforms,
            texture: None,
            uploaded: 0,
            srgb_target: target.is_srgb(),
            filters: GpuFilters::new(device, adapter),
            noise: None,
            curves,
            curves_view,
            curves_made: None,
            tiles: (0, 0),
            show_filtered: false,
        }
    }

    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, frame: &Frame) {
        let (format, bytes): (_, &[u8]) = match &frame.pixels {
            Pixels::Gray(v) => (wgpu::TextureFormat::R32Float, bytemuck::cast_slice(v)),
            Pixels::Rgba(v) => (wgpu::TextureFormat::Rgba8Unorm, v),
        };
        let size = (frame.width, frame.height);
        let reuse = self
            .texture
            .as_ref()
            .is_some_and(|t| t.size == size && t.format == format);
        if !reuse {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("dicom frame"),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.texture = Some(Texture {
                plain: bind_group(
                    device,
                    &self.layout,
                    &self.uniforms,
                    &self.curves_view,
                    &view,
                ),
                texture,
                view,
                size,
                format,
                targets: Targets::default(),
                filtered: None,
                made: None,
            });
        }
        let t = self.texture.as_mut().unwrap();
        t.made = None;
        self.noise = None;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &t.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.0 * 4),
                rows_per_image: Some(size.1),
            },
            wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
        );
        self.uploaded = frame.id;
    }

    /// Upload the frame, run the filters and make the local contrast curves
    /// as far as they have changed, then set the uniforms. `sigma` is the
    /// sharpening blur, in image pixels.
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &Frame,
        uniforms: Uniforms,
        filters: Filters,
        sigma: f32,
    ) {
        if self.uploaded != frame.id {
            self.upload(device, queue, frame);
        }
        self.run_filters(device, queue, encoder, frame, filters, sigma);

        let mut u = uniforms;
        if let Some(clip) = contrast::clip_limit(filters.contrast) {
            let window = [u.wl[0], u.wl[1]];
            if self.curves_made != Some((frame.id, window, clip)) {
                let curves = Curves::new(frame, (window[0], window[1]), clip);
                queue.write_texture(
                    self.curves.as_image_copy(),
                    bytemuck::cast_slice(&curves.table),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(BINS as u32 * 4),
                        rows_per_image: None,
                    },
                    wgpu::Extent3d {
                        width: BINS as u32,
                        height: (curves.nx * curves.ny) as u32,
                        depth_or_array_layers: 1,
                    },
                );
                self.tiles = (curves.nx, curves.ny);
                self.curves_made = Some((frame.id, window, clip));
            }
            u.tex[2] = self.tiles.0 as f32;
            u.tex[3] = self.tiles.1 as f32;
        }
        u.mode[2] = if self.srgb_target { 1.0 } else { 0.0 };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));
    }

    /// Denoise and sharpen the uploaded frame on the GPU, unless the result
    /// is already there. Frees the filtered copies when both are off.
    fn run_filters(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &Frame,
        filters: Filters,
        sigma: f32,
    ) {
        let Some(t) = &mut self.texture else {
            return;
        };
        let wanted = self.filters.is_some() && filters.at_image_size();
        self.show_filtered = wanted;
        if !wanted {
            t.targets = Targets::default();
            t.filtered = None;
            t.made = None;
            return;
        }
        // Local contrast does not change what the filters make, nor does
        // the sharpening blur when sharpening is off.
        let key = (
            Filters {
                contrast: Default::default(),
                ..filters
            },
            if filters.sharpen == Default::default() {
                0.0
            } else {
                sigma
            },
        );
        if t.made == Some(key) {
            return;
        }
        let noise = *self.noise.get_or_insert_with(|| filter::noise(frame));
        let gray = t.format == wgpu::TextureFormat::R32Float;
        let gpu = self.filters.as_ref().unwrap();
        let out = gpu.run(
            device,
            queue,
            encoder,
            &mut t.targets,
            &t.view,
            t.size,
            gray,
            filters,
            sigma,
            noise,
        );
        t.filtered = out.map(|view| {
            bind_group(
                device,
                &self.layout,
                &self.uniforms,
                &self.curves_view,
                view,
            )
        });
        t.made = Some(key);
    }

    fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        let Some(t) = &self.texture else {
            return;
        };
        let group = match &t.filtered {
            Some(filtered) if self.show_filtered => filtered,
            _ => &t.plain,
        };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..4, 0..1);
    }
}

/// A bind group that shows `view` with the image pipeline.
fn bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniforms: &wgpu::Buffer,
    curves: &wgpu::TextureView,
    view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dicom image"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(curves),
            },
        ],
    })
}

/// One frame's draw request. The texture is uploaded only when the frame
/// changes, and filtered only when the frame or the filters change.
pub struct ImageCallback {
    pub frame: Arc<Frame>,
    pub uniforms: Uniforms,
    pub filters: Filters,
    /// Sharpening blur, in image pixels.
    pub sigma: f32,
}

impl egui_wgpu::CallbackTrait for ImageCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(r) = resources.get_mut::<ImageRenderer>() {
            r.prepare(
                device,
                queue,
                encoder,
                &self.frame,
                self.uniforms,
                self.filters,
                self.sigma,
            );
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: eframe::egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(r) = resources.get::<ImageRenderer>() {
            r.draw(pass);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colormap::Colormap;
    use crate::export::{self, Look};
    use crate::filter::Level;
    use crate::testing::{self, colour, gaussian, gray};

    /// Draws frames through the viewer's shaders at one screen pixel per
    /// image pixel and checks them against the video's drawing code, with
    /// every filter on. Skipped when there is no GPU.
    #[test]
    fn screen_matches_video() {
        let Some((adapter, device, queue)) = testing::gpu() else {
            eprintln!("no GPU: skipped");
            return;
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut renderer = ImageRenderer::new(&device, &adapter, format);
        if renderer.filters.is_none() {
            eprintln!("this GPU cannot filter: skipped");
            return;
        }
        let (w, h) = (64u32, 48u32);
        let n = (w * h) as usize;
        let speckle = gaussian(n, 6.0, 21);
        // A ramp with a bright disc and a dark bar, and noise.
        let grey = || {
            let values = (0..n).map(|i| {
                let (x, y) = ((i % 64) as f32, (i / 64) as f32);
                let disc = if (x - 40.0).powi(2) + (y - 20.0).powi(2) < 64.0 {
                    150.0
                } else {
                    0.0
                };
                let bar = if (10.0..14.0).contains(&x) {
                    -80.0
                } else {
                    0.0
                };
                x * 3.0 + disc + bar + speckle[i]
            });
            gray(w, h, values.collect())
        };
        let doppler = || {
            let rgba = (0..n).flat_map(|i| {
                let (x, y) = ((i % 64) as f32, (i / 64) as f32);
                let g = (60.0 + x * 2.0 + speckle[i] * 2.0).clamp(0.0, 255.0) as u8;
                if (x - 30.0).powi(2) + (y - 24.0).powi(2) < 100.0 {
                    [210, 100, 30, 255]
                } else {
                    [g, g, g, 255]
                }
            });
            colour(w, h, rgba.collect())
        };
        let all = Filters {
            denoise: Level::Medium,
            sharpen: Level::Medium,
            contrast: Level::High,
        };
        let only = |f: fn(&mut Filters)| {
            let mut filters = Filters::default();
            f(&mut filters);
            filters
        };
        // Frame, filters, window, invert, colour map.
        type Case<'a> = (&'a dyn Fn() -> Frame, Filters, (f32, f32), bool, Colormap);
        let cases: [Case; 6] = [
            (
                &grey,
                only(|f| f.denoise = Level::Medium),
                (100.0, 300.0),
                false,
                Colormap::Grey,
            ),
            (
                &grey,
                only(|f| f.sharpen = Level::High),
                (100.0, 300.0),
                false,
                Colormap::Grey,
            ),
            (&grey, all, (100.0, 300.0), false, Colormap::HotIron),
            (
                &grey,
                only(|f| f.contrast = Level::Medium),
                (100.0, 300.0),
                true,
                Colormap::Grey,
            ),
            (
                &grey,
                Filters::default(),
                (100.0, 300.0),
                false,
                Colormap::Rainbow,
            ),
            (&doppler, all, (127.5, 255.0), false, Colormap::Grey),
        ];
        for (k, (make, filters, window, invert, colormap)) in cases.into_iter().enumerate() {
            let mut frame = make();
            frame.id = k as u64 + 1;
            let is_colour = frame.is_color();
            let uniforms = Uniforms {
                rect: [-1.0, 1.0, 1.0, -1.0],
                tex: [w as f32, h as f32, 0.0, 0.0],
                wl: [
                    window.0,
                    window.1,
                    if is_colour { 255.0 } else { 1.0 },
                    invert as u8 as f32,
                ],
                mode: [is_colour as u8 as f32, 1.0, 0.0, colormap.shader_id()],
            };
            let target = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = target.create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = device.create_command_encoder(&Default::default());
            renderer.prepare(
                &device,
                &queue,
                &mut encoder,
                &frame,
                uniforms,
                filters,
                1.0,
            );
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                renderer.draw(&mut pass);
            }
            let got = testing::read(&device, &queue, encoder, &target);

            let curves = contrast::clip_limit(filters.contrast)
                .map(|clip| Curves::new(&frame, window, clip));
            filter::apply(&mut frame, filters, 1.0);
            let look = Look {
                window,
                invert,
                smooth: true,
                colormap,
            };
            let want = export::draw(&frame, w as usize, h as usize, &look, curves.as_ref());
            let (mut worst, mut total) = (0.0f32, 0.0f32);
            for (i, px) in want.iter().enumerate() {
                for c in 0..3 {
                    let d = (got[i * 4 + c] as f32 / 255.0 - px[c]).abs();
                    worst = worst.max(d);
                    total += d;
                }
            }
            let mean = total / (n * 3) as f32;
            // Within the rounding of an 8-bit screen.
            assert!(
                worst < 1.0 / 255.0,
                "case {k}: worst difference {}/255",
                worst * 255.0
            );
            assert!(
                mean < 0.4 / 255.0,
                "case {k}: mean difference {}/255",
                mean * 255.0
            );
        }
    }
}

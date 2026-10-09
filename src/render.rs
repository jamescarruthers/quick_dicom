//! GPU drawing of the current frame through an egui paint callback.

use std::sync::Arc;

use eframe::egui_wgpu::{self, wgpu};

use crate::decode::{Frame, Pixels};
use crate::sharpen::{self, GpuSharpen, Target};

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub rect: [f32; 4],
    pub tex: [f32; 4],
    pub wl: [f32; 4],
    pub mode: [f32; 4],
}

struct Texture {
    bind_group: wgpu::BindGroup,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
    format: wgpu::TextureFormat,
    /// Textures for the sharpened image and the bind group that shows it,
    /// made the first time sharpening is asked for.
    sharp: Option<(Target, wgpu::BindGroup)>,
    /// Amount and sigma that `sharp` was made with, for the frame now
    /// uploaded.
    sharp_made: Option<(f32, f32)>,
}

/// GPU state, stored in egui's callback resources.
pub struct ImageRenderer {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    texture: Option<Texture>,
    uploaded: u64,
    srgb_target: bool,
    /// `None` when the GPU cannot sharpen.
    sharpener: Option<GpuSharpen>,
    /// Noise threshold of the uploaded frame, worked out when first needed.
    threshold: Option<f32>,
    /// Whether this frame's paint shows the sharpened image.
    show_sharp: bool,
}

impl ImageRenderer {
    /// Returns whether the GPU can sharpen.
    pub fn install(rs: &egui_wgpu::RenderState) -> bool {
        let device = &rs.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dicom image"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(include_str!("common.wgsl"), include_str!("image.wgsl")).into(),
            ),
        });
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
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
                    format: rs.target_format,
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

        let sharpener = GpuSharpen::new(device, &rs.adapter);
        let can_sharpen = sharpener.is_some();
        rs.renderer
            .write()
            .callback_resources
            .insert(ImageRenderer {
                pipeline,
                layout,
                uniforms,
                texture: None,
                uploaded: 0,
                srgb_target: rs.target_format.is_srgb(),
                sharpener,
                threshold: None,
                show_sharp: false,
            });
        can_sharpen
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
                bind_group: bind_group(device, &self.layout, &self.uniforms, &view),
                texture,
                view,
                size,
                format,
                sharp: None,
                sharp_made: None,
            });
        }
        let t = self.texture.as_mut().unwrap();
        t.sharp_made = None;
        self.threshold = None;
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

    /// Fill the sharpened texture for the uploaded frame, unless it already
    /// holds this amount and sigma. Returns whether there is one to show.
    fn sharpen(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &Frame,
        (amount, sigma): (f32, f32),
    ) -> bool {
        let (Some(gpu), Some(t)) = (&self.sharpener, &mut self.texture) else {
            return false;
        };
        if t.sharp_made == Some((amount, sigma)) {
            return true;
        }
        let (target, _) = t.sharp.get_or_insert_with(|| {
            let gray = t.format == wgpu::TextureFormat::R32Float;
            let target = gpu.target(device, &t.view, t.size, gray);
            let show = bind_group(device, &self.layout, &self.uniforms, &target.out);
            (target, show)
        });
        let threshold = *self
            .threshold
            .get_or_insert_with(|| sharpen::threshold(frame));
        gpu.run(queue, encoder, target, amount, sigma, threshold);
        t.sharp_made = Some((amount, sigma));
        true
    }
}

/// A bind group that shows `view` with the image pipeline.
fn bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniforms: &wgpu::Buffer,
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
        ],
    })
}

/// One frame's draw request. The texture is uploaded only when the frame changes.
pub struct ImageCallback {
    pub frame: Arc<Frame>,
    pub uniforms: Uniforms,
    /// Amount and blur sigma in image pixels, when sharpening.
    pub sharpen: Option<(f32, f32)>,
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
        let Some(r) = resources.get_mut::<ImageRenderer>() else {
            return Vec::new();
        };
        if r.uploaded != self.frame.id {
            r.upload(device, queue, &self.frame);
        }
        r.show_sharp = match self.sharpen {
            Some(s) => r.sharpen(device, queue, encoder, &self.frame, s),
            None => {
                // Give back the GPU memory: two more copies of the image.
                if let Some(t) = &mut r.texture {
                    t.sharp = None;
                    t.sharp_made = None;
                }
                false
            }
        };
        let mut u = self.uniforms;
        u.mode[2] = if r.srgb_target { 1.0 } else { 0.0 };
        queue.write_buffer(&r.uniforms, 0, bytemuck::bytes_of(&u));
        Vec::new()
    }

    fn paint(
        &self,
        _info: eframe::egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(r) = resources.get::<ImageRenderer>() else {
            return;
        };
        let Some(t) = &r.texture else {
            return;
        };
        let group = match &t.sharp {
            Some((_, show)) if r.show_sharp => show,
            _ => &t.bind_group,
        };
        pass.set_pipeline(&r.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..4, 0..1);
    }
}

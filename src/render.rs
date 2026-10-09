//! GPU drawing of the current frame through an egui paint callback.

use std::sync::Arc;

use eframe::egui_wgpu::{self, wgpu};

use crate::decode::{Frame, Pixels};

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
    size: (u32, u32),
    format: wgpu::TextureFormat,
}

/// GPU state, stored in egui's callback resources.
pub struct ImageRenderer {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    texture: Option<Texture>,
    uploaded: u64,
    srgb_target: bool,
}

impl ImageRenderer {
    pub fn install(rs: &egui_wgpu::RenderState) {
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
            });
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
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dicom image"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniforms.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                ],
            });
            self.texture = Some(Texture {
                bind_group,
                texture,
                size,
                format,
            });
        }
        let t = self.texture.as_ref().unwrap();
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
}

/// One frame's draw request. The texture is uploaded only when the frame changes.
pub struct ImageCallback {
    pub frame: Arc<Frame>,
    pub uniforms: Uniforms,
}

impl egui_wgpu::CallbackTrait for ImageCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(r) = resources.get_mut::<ImageRenderer>() else {
            return Vec::new();
        };
        if r.uploaded != self.frame.id {
            r.upload(device, queue, &self.frame);
        }
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
        pass.set_pipeline(&r.pipeline);
        pass.set_bind_group(0, &t.bind_group, &[]);
        pass.draw(0..4, 0..1);
    }
}

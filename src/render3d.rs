//! GPU drawing of the 3D view through an egui paint callback.
//!
//! `prepare` draws every slice into an offscreen image (16-bit float where
//! the GPU can blend it, so faint slices still add up); `paint` copies that
//! image onto the screen, applying the colour map for maximum intensity.

use std::sync::Arc;

use eframe::egui_wgpu::{self, wgpu};

use crate::volume::Volume;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumeUniforms {
    pub mx: [f32; 4],
    pub my: [f32; 4],
    pub dims: [f32; 4],
    pub slices: [f32; 4],
    pub wl: [f32; 4],
    pub look: [f32; 4],
}

struct VolumeTexture {
    bind_group: wgpu::BindGroup,
    layers: u32,
}

struct Offscreen {
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    size: (u32, u32),
}

pub struct VolumeRenderer {
    slice_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    blend: wgpu::RenderPipeline,
    maximum: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    sampler: wgpu::Sampler,
    format: wgpu::TextureFormat,
    volume: Option<VolumeTexture>,
    /// The volume on the GPU, to tell when a new one needs uploading.
    uploaded: Option<Arc<Volume>>,
    offscreen: Option<Offscreen>,
    srgb_target: bool,
}

impl VolumeRenderer {
    pub fn install(rs: &egui_wgpu::RenderState) {
        let device = &rs.device;
        // Blending many faint slices needs more than 8 bits per channel.
        let wanted = wgpu::TextureFormat::Rgba16Float;
        let features = rs.adapter.get_texture_format_features(wanted);
        let format = if features
            .flags
            .contains(wgpu::TextureFormatFeatureFlags::BLENDABLE)
            && features
                .allowed_usages
                .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
        {
            wanted
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dicom volume"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(include_str!("common.wgsl"), include_str!("volume.wgsl")).into(),
            ),
        });
        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let slice_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume slices"),
            entries: &[
                uniform_entry,
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume composite"),
            entries: &[
                uniform_entry,
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        let pipeline = |layout: &wgpu::BindGroupLayout,
                        vs: &str,
                        fs: &str,
                        format: wgpu::TextureFormat,
                        blend: Option<wgpu::BlendState>| {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("dicom volume"),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(fs),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(vs),
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
                    entry_point: Some(fs),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let over = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let max = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Max,
        };
        let blend = pipeline(
            &slice_layout,
            "vs_slice",
            "fs_slice",
            format,
            Some(wgpu::BlendState {
                color: over,
                alpha: over,
            }),
        );
        let maximum = pipeline(
            &slice_layout,
            "vs_slice",
            "fs_slice",
            format,
            Some(wgpu::BlendState {
                color: max,
                alpha: max,
            }),
        );
        let composite = pipeline(
            &composite_layout,
            "vs_composite",
            "fs_composite",
            rs.target_format,
            None,
        );

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volume uniforms"),
            size: std::mem::size_of::<VolumeUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volume"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        rs.renderer
            .write()
            .callback_resources
            .insert(VolumeRenderer {
                slice_layout,
                composite_layout,
                blend,
                maximum,
                composite,
                uniforms,
                sampler,
                format,
                volume: None,
                uploaded: None,
                offscreen: None,
                srgb_target: rs.target_format.is_srgb(),
            });
    }

    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, volume: &Arc<Volume>) {
        let size = wgpu::Extent3d {
            width: volume.width,
            height: volume.height,
            depth_or_array_layers: volume.depth,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dicom volume"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&volume.voxels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(volume.width * 2),
                rows_per_image: Some(volume.height),
            },
            size,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volume slices"),
            layout: &self.slice_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.volume = Some(VolumeTexture {
            bind_group,
            layers: volume.depth,
        });
        self.uploaded = Some(volume.clone());
    }

    fn ensure_offscreen(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        if self.offscreen.as_ref().is_some_and(|o| o.size == size) {
            return;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volume offscreen"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volume composite"),
            layout: &self.composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });
        self.offscreen = Some(Offscreen {
            view,
            bind_group,
            size,
        });
    }
}

/// One frame of the 3D view.
pub struct VolumeCallback {
    pub volume: Arc<Volume>,
    pub uniforms: VolumeUniforms,
    /// Size of the view in physical pixels.
    pub size_px: (u32, u32),
    pub maximum_intensity: bool,
}

impl egui_wgpu::CallbackTrait for VolumeCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(r) = resources.get_mut::<VolumeRenderer>() else {
            return Vec::new();
        };
        if self.size_px.0 == 0 || self.size_px.1 == 0 {
            return Vec::new();
        }
        if r.uploaded
            .as_ref()
            .is_none_or(|v| !Arc::ptr_eq(v, &self.volume))
        {
            r.upload(device, queue, &self.volume);
        }
        r.ensure_offscreen(device, self.size_px);
        let mut u = self.uniforms;
        u.look[2] = if r.srgb_target { 1.0 } else { 0.0 };
        queue.write_buffer(&r.uniforms, 0, bytemuck::bytes_of(&u));

        let (Some(volume), Some(offscreen)) = (&r.volume, &r.offscreen) else {
            return Vec::new();
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("volume slices"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &offscreen.view,
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
        pass.set_pipeline(if self.maximum_intensity {
            &r.maximum
        } else {
            &r.blend
        });
        pass.set_bind_group(0, &volume.bind_group, &[]);
        pass.draw(0..4, 0..volume.layers);
        Vec::new()
    }

    fn paint(
        &self,
        _info: eframe::egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(r) = resources.get::<VolumeRenderer>() else {
            return;
        };
        let Some(offscreen) = &r.offscreen else {
            return;
        };
        pass.set_pipeline(&r.composite);
        pass.set_bind_group(0, &offscreen.bind_group, &[]);
        pass.draw(0..4, 0..1);
    }
}

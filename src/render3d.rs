//! GPU drawing of the 3D view through an egui paint callback.
//!
//! `prepare` casts a ray through the volume for every pixel of an
//! offscreen image (16-bit float where the GPU allows), but only when the
//! view has changed; `paint` copies that image onto the screen, applying
//! the colour map for maximum intensity. While the view moves, the
//! offscreen image has half the resolution, so turning stays smooth.

use std::sync::{Arc, Mutex};

use eframe::egui_wgpu::{self, wgpu};

use crate::monai::Segmentation;
use crate::volume::Volume;

#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumeUniforms {
    pub right: [f32; 4],
    pub down: [f32; 4],
    pub into: [f32; 4],
    pub view: [f32; 4],
    pub dims: [f32; 4],
    pub slices: [f32; 4],
    pub wl: [f32; 4],
    pub look: [f32; 4],
    /// Segmentation labels shown (0 or 1), and every how many of the
    /// labels' slices the volume took one. Set by the renderer.
    pub labels: [f32; 4],
}

/// What the 3D view shows, and from where.
pub struct Scene {
    /// Size of the view, in any unit that `centre` and `scale` share.
    pub view: (f32, f32),
    /// Where the centre of rotation sits in the view, from its top left.
    pub centre: (f32, f32),
    /// View units per volume pixel.
    pub scale: f32,
    pub yaw: f32,
    pub pitch: f32,
    /// Distance between layers, in volume pixels.
    pub gap: f32,
    /// The layer shown in 2D.
    pub current: f32,
    /// How far the turn into 3D has got: 0 is the flat slice, 1 the volume.
    pub t: f32,
    /// Window centre and width.
    pub window: (f32, f32),
    pub invert: bool,
    pub colormap: f32,
    /// How opaque the volume's whole thickness would be, seen face on, if
    /// it were all at the top of the window.
    pub density: f32,
    pub maximum: bool,
    pub shading: bool,
    /// Two samples per voxel rather than one: smoother, at twice the cost.
    pub fine: bool,
}

impl VolumeUniforms {
    pub fn new(scene: &Scene, width: f32, height: f32, depth: f32) -> Self {
        let t = scene.t;
        let r = rotation(scene.yaw * t, scene.pitch * t);
        // Turn about the current slice at first, the middle of the volume
        // at the end, as the volume thickens from that slice.
        let pivot = scene.current + ((depth - 1.0) * 0.5 - scene.current) * t;
        let (vw, vh) = scene.view;
        let row = |i: usize| [r[i][0], r[i][1], r[i][2], 0.0];
        VolumeUniforms {
            right: row(0),
            down: row(1),
            into: row(2),
            view: [
                2.0 * scene.scale / vw,
                2.0 * scene.scale / vh,
                2.0 * scene.centre.0 / vw - 1.0,
                1.0 - 2.0 * scene.centre.1 / vh,
            ],
            dims: [width, height, depth, scene.current],
            slices: [scene.gap * t, pivot, t, if scene.fine { 2.0 } else { 1.0 }],
            wl: [
                scene.window.0,
                scene.window.1,
                scene.invert as u8 as f32,
                scene.colormap,
            ],
            look: [
                scene.density,
                scene.maximum as u8 as f32,
                0.0,
                scene.shading as u8 as f32,
            ],
            labels: [0.0; 4],
        }
    }
}

/// Rotation turning by `yaw` about the screen's vertical axis, then by
/// `pitch` about its horizontal axis. Rows are screen x (right), y (down)
/// and z (into the screen), in model space.
pub fn rotation(yaw: f32, pitch: f32) -> [[f32; 3]; 3] {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    [
        [cy, 0.0, sy],
        [sp * sy, cp, -sp * cy],
        [-cp * sy, sp, cp * cy],
    ]
}

struct Offscreen {
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    size: (u32, u32),
}

pub struct VolumeRenderer {
    ray_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    ray: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    sampler: wgpu::Sampler,
    format: wgpu::TextureFormat,
    /// Binds the volume's 3D texture for the ray pass.
    volume: Option<wgpu::BindGroup>,
    labels_layout: wgpu::BindGroupLayout,
    /// The segmentation labels as a 3D texture, or a single empty voxel.
    labels: wgpu::BindGroup,
    /// Which segmentation `labels` holds, or failed to hold.
    labels_for: Option<u64>,
    labels_ok: bool,
    /// The volume on the GPU, to tell when a new one needs uploading.
    uploaded: Option<Arc<Volume>>,
    offscreen: Option<Offscreen>,
    /// What the offscreen image shows, so an unchanged view is not cast
    /// again.
    drawn: Option<VolumeUniforms>,
    srgb_target: bool,
}

impl VolumeRenderer {
    pub fn install(rs: &egui_wgpu::RenderState) {
        let renderer = Self::new(&rs.device, &rs.adapter, rs.target_format);
        rs.renderer.write().callback_resources.insert(renderer);
    }

    fn new(device: &wgpu::Device, adapter: &wgpu::Adapter, target: wgpu::TextureFormat) -> Self {
        // Faint colour adds up over many samples, so keep more than 8 bits
        // where the GPU can draw into them.
        let wanted = wgpu::TextureFormat::Rgba16Float;
        let format = if adapter
            .get_texture_format_features(wanted)
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
        let texture = |binding, view_dimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension,
                multisampled: false,
            },
            count: None,
        };
        let sampler_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let ray_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume rays"),
            entries: &[
                uniform_entry,
                texture(1, wgpu::TextureViewDimension::D3),
                sampler_entry(2),
            ],
        });
        let labels_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume labels"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume composite"),
            entries: &[
                uniform_entry,
                texture(3, wgpu::TextureViewDimension::D2),
                sampler_entry(4),
            ],
        });

        let pipeline = |layouts: &[Option<&wgpu::BindGroupLayout>],
                        vs: &str,
                        fs: &str,
                        format: wgpu::TextureFormat| {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("dicom volume"),
                bind_group_layouts: layouts,
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
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let ray = pipeline(
            &[Some(&ray_layout), Some(&labels_layout)],
            "vs_ray",
            "fs_ray",
            format,
        );
        let composite = pipeline(
            &[Some(&composite_layout)],
            "vs_composite",
            "fs_composite",
            target,
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

        VolumeRenderer {
            ray_layout,
            composite_layout,
            ray,
            composite,
            uniforms,
            sampler,
            format,
            volume: None,
            labels: label_volume(
                device,
                &labels_layout,
                [1, 1, 1],
                wgpu::TextureFormat::R8Uint,
            )
            .1,
            labels_layout,
            labels_for: None,
            labels_ok: false,
            uploaded: None,
            offscreen: None,
            drawn: None,
            srgb_target: target.is_srgb(),
        }
    }

    /// Put the volume on the GPU as one 3D texture, a slice at a time.
    /// Fails, rather than crashing, when the GPU runs out of memory.
    fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        volume: &Arc<Volume>,
    ) -> Result<(), String> {
        // Free the previous volume before allocating the next.
        self.volume = None;
        self.drawn = None;
        self.uploaded = Some(volume.clone());
        let voxels = volume.take_voxels();
        let (w, h) = (volume.width, volume.height);
        let layer_len = (w * h) as usize;
        if voxels.len() < layer_len * volume.depth as usize {
            return Err("the volume has no data".into());
        }

        let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dicom volume"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: volume.depth,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (z, layer) in voxels.chunks_exact(layer_len).enumerate() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: z as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(layer),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 2),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volume rays"),
            layout: &self.ray_layout,
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
        // Error scopes pop in the reverse order they were pushed.
        let invalid = pollster::block_on(validation.pop());
        let out_of_memory = pollster::block_on(memory.pop());
        if out_of_memory.is_some() {
            return Err(format!(
                "the GPU does not have {} MB free for this volume",
                volume.bytes() >> 20
            ));
        }
        if let Some(e) = invalid {
            // wgpu's message spans several lines; the last says what was wrong.
            let text = e.to_string();
            let cause = text.lines().map(str::trim).rfind(|l| !l.is_empty());
            return Err(format!(
                "the GPU refused the volume ({})",
                cause.unwrap_or("validation error")
            ));
        }
        self.volume = Some(bind_group);
        Ok(())
    }

    fn ensure_offscreen(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        if self.offscreen.as_ref().is_some_and(|o| o.size == size) {
            return;
        }
        self.drawn = None;
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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.offscreen = Some(Offscreen {
            view,
            bind_group,
            size,
        });
    }

    /// Put segmentation labels on the GPU as a 3D texture, as bytes when
    /// every label fits in one. Fails, rather than crashing, when the GPU
    /// has no room for them.
    fn upload_labels(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        seg: &Segmentation,
    ) -> Result<(), String> {
        self.labels_for = Some(seg.id);
        self.labels_ok = false;
        self.drawn = None;
        let wide = seg.max_label() > 255;
        let format = if wide {
            wgpu::TextureFormat::R16Uint
        } else {
            wgpu::TextureFormat::R8Uint
        };
        let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let (texture, group) = label_volume(
            device,
            &self.labels_layout,
            [seg.width, seg.height, seg.depth],
            format,
        );
        for k in 0..seg.depth as usize {
            let slice = seg.slice(k);
            let bytes: Vec<u8> = if wide {
                bytemuck::cast_slice(slice).to_vec()
            } else {
                slice.iter().map(|&v| v as u8).collect()
            };
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: k as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(seg.width * if wide { 2 } else { 1 }),
                    rows_per_image: Some(seg.height),
                },
                wgpu::Extent3d {
                    width: seg.width,
                    height: seg.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        let invalid = pollster::block_on(validation.pop());
        let out_of_memory = pollster::block_on(memory.pop());
        if out_of_memory.is_some() || invalid.is_some() {
            self.labels = label_volume(device, &self.labels_layout, [1, 1, 1], format).1;
            return Err("the GPU has no room for the segmentation in 3D".into());
        }
        self.labels = group;
        self.labels_ok = true;
        Ok(())
    }

    /// Upload the volume and labels if they are new, then cast the rays if
    /// the view has changed. An upload that fails is reported once and not
    /// retried; a volume that fails is an error, labels that fail a notice.
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        volume: &Arc<Volume>,
        uniforms: VolumeUniforms,
        size: (u32, u32),
        labels: Option<&Segmentation>,
    ) -> (Result<(), String>, Option<String>) {
        let mut result = Ok(());
        let mut notice = None;
        if self
            .uploaded
            .as_ref()
            .is_none_or(|v| !Arc::ptr_eq(v, volume))
        {
            result = self.upload(device, queue, volume);
        }
        if let Some(seg) = labels
            && self.labels_for != Some(seg.id)
            && let Err(e) = self.upload_labels(device, queue, seg)
        {
            notice = Some(e);
        }
        self.ensure_offscreen(device, size);
        let mut u = uniforms;
        u.look[2] = if self.srgb_target { 1.0 } else { 0.0 };
        if labels.is_some() && self.labels_ok {
            u.labels = [1.0, volume.stride as f32, 0.0, 0.0];
        }
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));

        let (Some(group), Some(offscreen)) = (&self.volume, &self.offscreen) else {
            return (result, notice);
        };
        if self.drawn == Some(u) {
            return (result, notice);
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("volume rays"),
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
        pass.set_pipeline(&self.ray);
        pass.set_bind_group(0, group, &[]);
        pass.set_bind_group(1, &self.labels, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        self.drawn = Some(u);
        (result, notice)
    }

    fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        let Some(offscreen) = &self.offscreen else {
            return;
        };
        if self.volume.is_none() {
            return;
        }
        pass.set_pipeline(&self.composite);
        pass.set_bind_group(0, &offscreen.bind_group, &[]);
        pass.draw(0..4, 0..1);
    }
}

/// A 3D texture of labels, R8Uint or R16Uint, with its bind group. A new
/// texture is all zeros.
fn label_volume(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    size: [u32; 3],
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::BindGroup) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("volume labels"),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: size[2],
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("volume labels"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    });
    (texture, group)
}

/// One frame of the 3D view.
pub struct VolumeCallback {
    pub volume: Arc<Volume>,
    /// Segmentation labels to colour the volume with.
    pub labels: Option<Arc<Segmentation>>,
    /// Where to report labels the GPU could not take.
    pub notice: Arc<Mutex<Option<String>>>,
    pub uniforms: VolumeUniforms,
    /// Size of the offscreen image in pixels: the view's, or half that
    /// while it moves.
    pub size_px: (u32, u32),
    /// Where to report a volume the GPU could not take.
    pub error: Arc<Mutex<Option<String>>>,
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
        let (result, notice) = r.prepare(
            device,
            queue,
            encoder,
            &self.volume,
            self.uniforms,
            self.size_px,
            self.labels.as_deref(),
        );
        if let Err(e) = result {
            *self.error.lock().unwrap() = Some(e);
        }
        if let Some(n) = notice {
            *self.notice.lock().unwrap() = Some(n);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: eframe::egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(r) = resources.get::<VolumeRenderer>() {
            r.draw(pass);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing;

    #[test]
    fn rotation_is_identity_at_rest() {
        assert_eq!(
            rotation(0.0, 0.0),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );
    }

    /// A 32-voxel cube of zeros with a bright 12-voxel cube in the middle.
    fn cube() -> Arc<Volume> {
        let n = 32usize;
        let inside = |i: usize| (10..22).contains(&i);
        let values: Vec<f32> = (0..n * n * n)
            .map(|i| {
                let (x, y, z) = (i % n, i / n % n, i / (n * n));
                if inside(x) && inside(y) && inside(z) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        Arc::new(Volume::from_values(32, 32, 32, &values))
    }

    /// The cube in a 64-pixel view, 1.5 pixels per voxel, all the way
    /// into 3D.
    fn scene() -> Scene {
        Scene {
            view: (64.0, 64.0),
            centre: (32.0, 32.0),
            scale: 1.5,
            yaw: 0.0,
            pitch: 0.0,
            gap: 1.0,
            current: 16.0,
            t: 1.0,
            window: (0.5, 1.0),
            invert: false,
            colormap: 0.0,
            density: 8.0,
            maximum: false,
            shading: false,
            fine: true,
        }
    }

    /// Casts the rays and returns the grey level of each screen pixel.
    fn render(
        renderer: &mut VolumeRenderer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        volume: &Arc<Volume>,
        scene: &Scene,
    ) -> Vec<u8> {
        render_rgba(renderer, device, queue, volume, scene, None)
            .iter()
            .map(|p| p[1])
            .collect()
    }

    /// Casts the rays, with labels if given, and returns the screen.
    fn render_rgba(
        renderer: &mut VolumeRenderer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        volume: &Arc<Volume>,
        scene: &Scene,
        labels: Option<&Segmentation>,
    ) -> Vec<[u8; 4]> {
        let uniforms = VolumeUniforms::new(scene, 32.0, 32.0, 32.0);
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        renderer
            .prepare(
                device,
                queue,
                &mut encoder,
                volume,
                uniforms,
                (64, 64),
                labels,
            )
            .0
            .expect("the volume uploads");
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
        let rgba = testing::read(device, queue, encoder, &target);
        rgba.as_chunks::<4>().0.to_vec()
    }

    /// The ray caster sees the cube from any side, with its edges where
    /// the geometry puts them. Skipped when there is no GPU.
    #[test]
    fn rays_find_the_cube() {
        let Some((adapter, device, queue)) = testing::gpu() else {
            eprintln!("no GPU: skipped");
            return;
        };
        let mut renderer = VolumeRenderer::new(&device, &adapter, wgpu::TextureFormat::Rgba8Unorm);
        let volume = cube();
        let at = |image: &[u8], x: usize, y: usize| image[y * 64 + x];
        // The volume spans pixels 8 to 56; the bright cube 23 to 41.
        let looks_right = |image: &[u8], what: &str| {
            assert!(
                at(image, 32, 32) > 200,
                "{what}: centre {}",
                at(image, 32, 32)
            );
            assert!(at(image, 25, 38) > 200, "{what}: inside the cube");
            assert!(at(image, 20, 32) < 10, "{what}: beside the cube");
            assert!(at(image, 2, 2) < 10, "{what}: outside the volume");
        };

        looks_right(
            &render(&mut renderer, &device, &queue, &volume, &scene()),
            "face on",
        );

        // Edge on, slices would vanish; the volume does not.
        let side = Scene {
            yaw: std::f32::consts::FRAC_PI_2,
            ..scene()
        };
        looks_right(
            &render(&mut renderer, &device, &queue, &volume, &side),
            "edge on",
        );

        let tilted = Scene {
            yaw: 0.5,
            pitch: -0.4,
            shading: true,
            ..scene()
        };
        let image = render(&mut renderer, &device, &queue, &volume, &tilted);
        assert!(
            at(&image, 32, 32) > 100 && at(&image, 4, 4) < 10,
            "tilted and shaded"
        );

        let mip = Scene {
            maximum: true,
            ..scene()
        };
        looks_right(
            &render(&mut renderer, &device, &queue, &volume, &mip),
            "MIP",
        );

        // At the start of the turn the current slice shows on its own: one
        // through the cube shows it, one beside it shows nothing.
        let early = Scene {
            t: 0.001,
            ..scene()
        };
        let image = render(&mut renderer, &device, &queue, &volume, &early);
        assert!(at(&image, 32, 32) > 200, "slice through the cube");
        let beside = Scene {
            current: 4.0,
            ..early
        };
        let image = render(&mut renderer, &device, &queue, &volume, &beside);
        assert!(at(&image, 32, 32) < 20, "slice beside the cube");
    }

    /// Labelled voxels show solid in their label's colour, where the
    /// geometry puts them, even where the scan is dark; the rest of the
    /// scan fades. Skipped when there is no GPU.
    #[test]
    fn labels_colour_the_volume() {
        let Some((adapter, device, queue)) = testing::gpu() else {
            eprintln!("no GPU: skipped");
            return;
        };
        let mut renderer = VolumeRenderer::new(&device, &adapter, wgpu::TextureFormat::Rgba8Unorm);
        let volume = cube();
        // Labels at twice the volume's resolution across, every slice:
        // label 3 fills volume voxels 2..8 across and down, in the dark
        // part of the scan, which is screen pixels 11 to 20.
        let (n, d) = (64usize, 32usize);
        let mut voxels = vec![0u16; n * n * d];
        for z in 0..d {
            for y in 4..16 {
                for x in 4..16 {
                    voxels[(z * n + y) * n + x] = 3;
                }
            }
        }
        let seg = Segmentation::new(String::new(), String::new(), [n, n, d], voxels, Vec::new());
        let c = crate::colormap::label_colour(3).map(|v| v * 255.0);
        let is_label = |p: [u8; 4]| (0..3).all(|k| (p[k] as f32 - c[k]).abs() < 3.0);

        for (scene, what) in [
            (scene(), "face on"),
            (
                Scene {
                    pitch: 0.3,
                    shading: true,
                    ..scene()
                },
                "tilted and shaded",
            ),
        ] {
            let image = render_rgba(&mut renderer, &device, &queue, &volume, &scene, Some(&seg));
            let at = |x: usize, y: usize| image[y * 64 + x];
            if !scene.shading {
                assert!(
                    is_label(at(15, 15)),
                    "{what}: label {:?}, want {c:?}",
                    at(15, 15)
                );
            } else {
                assert!(at(15, 15)[0] > 40, "{what}: label {:?}", at(15, 15));
            }
            assert!(
                at(24, 15)[1] < 10,
                "{what}: beside the label {:?}",
                at(24, 15)
            );
            // The bright cube, unlabelled, is only a faint presence.
            let faint = at(32, 32)[1];
            assert!(faint > 5 && faint < 70, "{what}: unlabelled cube {faint}");
        }

        // Without labels the dark corner stays dark.
        let image = render_rgba(&mut renderer, &device, &queue, &volume, &scene(), None);
        assert!(image[15 * 64 + 15][1] < 10, "labels off");
    }
}

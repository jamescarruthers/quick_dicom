//! GPU drawing of the 3D view through an egui paint callback.
//!
//! `prepare` casts a ray through the volume for every pixel of an
//! offscreen image (16-bit float where the GPU allows), but only when the
//! view has changed; `paint` copies that image onto the screen, applying
//! the colour map for maximum intensity. While the view moves, the
//! offscreen image has half the resolution, so turning stays smooth.

use std::sync::{Arc, Mutex};

use eframe::egui_wgpu::{self, wgpu};

use crate::monai::{LabelMask, Segmentation};
use crate::render::{MaskTexture, mask_entry};
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
    /// Segmentation labels shown (0 or 1), every how many of the labels'
    /// slices the volume took one, which labels are shown (a version
    /// number, so a change redraws), and flags: 1 for smooth surfaces, 2
    /// for labels stored in 16 bits. The renderer sets all but the first
    /// flag.
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
    /// Segmentation labels with smooth surfaces rather than as voxels.
    pub smooth: bool,
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
            labels: [0.0, 0.0, 0.0, scene.smooth as u8 as f32],
        }
    }
}

/// The 3D view's geometry on the CPU, as the ray caster uses it: where a
/// point of the volume shows, and what the line of sight through a point
/// of the view meets first. Positions in the view are in the scene's
/// units, from its top left.
pub struct Projection {
    right: [f32; 3],
    down: [f32; 3],
    into: [f32; 3],
    centre: (f32, f32),
    scale: f32,
    /// The volume's width, height and layers.
    dims: [f32; 3],
    /// Distance between layers, and the layer at the centre of turning.
    gap: f32,
    pivot: f32,
    /// The volume took every `stride`th of the labels' slices.
    stride: f32,
    smooth: bool,
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl Projection {
    pub fn new(scene: &Scene, u: &VolumeUniforms, stride: usize) -> Projection {
        let v3 = |v: [f32; 4]| [v[0], v[1], v[2]];
        Projection {
            right: v3(u.right),
            down: v3(u.down),
            into: v3(u.into),
            centre: scene.centre,
            scale: scene.scale,
            dims: [u.dims[0], u.dims[1], u.dims[2]],
            gap: u.slices[0].max(1e-4),
            pivot: u.slices[1],
            stride: stride.max(1) as f32,
            smooth: u.labels[3] as u32 & 1 != 0,
        }
    }

    /// The model point at the middle of label voxel (x, y, slice).
    fn model_point(&self, seg: &Segmentation, v: [u32; 3]) -> [f32; 3] {
        [
            ((v[0] as f32 + 0.5) / seg.width as f32 - 0.5) * self.dims[0],
            ((v[1] as f32 + 0.5) / seg.height as f32 - 0.5) * self.dims[1],
            (v[2] as f32 / self.stride - self.pivot) * self.gap,
        ]
    }

    fn to_view(&self, p: [f32; 3]) -> (f32, f32) {
        (
            self.centre.0 + self.scale * dot(p, self.right),
            self.centre.1 + self.scale * dot(p, self.down),
        )
    }

    /// Where label voxel `v` shows in the view.
    pub fn voxel_in_view(&self, seg: &Segmentation, v: [u32; 3]) -> (f32, f32) {
        self.to_view(self.model_point(seg, v))
    }

    /// The shown label at a model point, as the shader finds it.
    fn label_at(&self, seg: &Segmentation, mask: &LabelMask, p: [f32; 3]) -> u16 {
        let g = [
            (p[0] / self.dims[0] + 0.5) * seg.width as f32 - 0.5,
            (p[1] / self.dims[1] + 0.5) * seg.height as f32 - 0.5,
            (p[2] / self.gap + self.pivot) * self.stride,
        ];
        let id = if self.smooth {
            crate::surface::smooth_label(seg, g).0
        } else {
            let [x, y, k] = g.map(|c| (c + 0.5).floor());
            if x < 0.0 || y < 0.0 || k < 0.0 {
                return 0;
            }
            seg.at(x as u32, y as u32, k as u32)
        };
        if mask.shows(id) { id } else { 0 }
    }

    /// Where a line from `origin` along `dir` enters and leaves the
    /// volume's box, if it meets it.
    fn span(&self, origin: [f32; 3], dir: [f32; 3]) -> Option<(f32, f32)> {
        let [w, h, d] = self.dims;
        let lo = [-0.5 * w, -0.5 * h, (-0.5 - self.pivot) * self.gap];
        let hi = [0.5 * w, 0.5 * h, (d - 0.5 - self.pivot) * self.gap];
        let (mut near, mut far) = (f32::MIN, f32::MAX);
        for i in 0..3 {
            if dir[i].abs() < 1e-12 {
                if origin[i] < lo[i] || origin[i] > hi[i] {
                    return None;
                }
                continue;
            }
            let (a, b) = ((lo[i] - origin[i]) / dir[i], (hi[i] - origin[i]) / dir[i]);
            near = near.max(a.min(b));
            far = far.min(a.max(b));
        }
        (near <= far).then_some((near, far))
    }

    /// Steps small enough to visit every label voxel.
    fn step(&self, seg: &Segmentation) -> f32 {
        let across = (self.dims[0] / seg.width as f32).min(self.dims[1] / seg.height as f32);
        0.5 * across.min(self.gap / self.stride)
    }

    /// The first shown label the line of sight through view point `at`
    /// meets: what the pointer is over.
    pub fn label_under(&self, seg: &Segmentation, mask: &LabelMask, at: (f32, f32)) -> Option<u16> {
        let sx = (at.0 - self.centre.0) / self.scale;
        let sy = (at.1 - self.centre.1) / self.scale;
        let origin: [f32; 3] = std::array::from_fn(|i| sx * self.right[i] + sy * self.down[i]);
        let (near, far) = self.span(origin, self.into)?;
        let step = self.step(seg);
        let mut s = near;
        while s <= far {
            let p = std::array::from_fn(|i| origin[i] + self.into[i] * s);
            let id = self.label_at(seg, mask, p);
            if id != 0 {
                return Some(id);
            }
            s += step;
        }
        None
    }

    /// Whether another shown label lies between voxel `v` of `label` and
    /// the viewer.
    pub fn behind_another(
        &self,
        seg: &Segmentation,
        mask: &LabelMask,
        label: u16,
        v: [u32; 3],
    ) -> bool {
        let origin = self.model_point(seg, v);
        let back = self.into.map(|c| -c);
        let Some((_, far)) = self.span(origin, back) else {
            return false;
        };
        let step = self.step(seg);
        let mut s = step;
        while s <= far {
            let p = std::array::from_fn(|i| origin[i] + back[i] * s);
            let id = self.label_at(seg, mask, p);
            if id != 0 && id != label {
                return true;
            }
            s += step;
        }
        false
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
    mask: MaskTexture,
    /// Which segmentation `labels` holds, or failed to hold, and whether
    /// smoothed.
    labels_for: Option<(u64, bool)>,
    labels_ok: bool,
    /// Labels stored in 16 bits rather than 8, so 64 bits a voxel.
    labels_wide: bool,
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
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                mask_entry(1),
            ],
        });
        let mask = MaskTexture::new(device);
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
                wgpu::TextureFormat::R32Uint,
                &mask.view,
            )
            .1,
            labels_layout,
            mask,
            labels_for: None,
            labels_ok: false,
            labels_wide: false,
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

    /// Put segmentation labels on the GPU as a 3D texture, smoothed or as
    /// the model gave them, each with its three distances: in 32 bits when
    /// every label fits in a byte, else 64. Fails, rather than crashing,
    /// when the GPU has no room for them.
    fn upload_labels(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        seg: &Segmentation,
        smooth: bool,
    ) -> Result<(), String> {
        self.labels_for = Some((seg.id, smooth));
        let labels = if smooth {
            seg.surface_labels()
        } else {
            &seg.voxels
        };
        self.labels_ok = false;
        self.drawn = None;
        let wide = seg.max_label() > 255;
        self.labels_wide = wide;
        let format = if wide {
            wgpu::TextureFormat::Rg32Uint
        } else {
            wgpu::TextureFormat::R32Uint
        };
        let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let (texture, group) = label_volume(
            device,
            &self.labels_layout,
            [seg.width, seg.height, seg.depth],
            format,
            &self.mask.view,
        );
        let plane = (seg.width * seg.height) as usize;
        for k in 0..seg.depth as usize {
            let slice = &labels[k * plane..(k + 1) * plane];
            let distances = &seg.distances[k * plane..(k + 1) * plane];
            let voxels = slice.iter().zip(distances);
            let bytes: Vec<u8> = if wide {
                voxels
                    .flat_map(|(&v, &[own, up, down])| {
                        let label = v as u32 | (own as u32) << 16;
                        let across = up as u32 | (down as u32) << 8;
                        [label.to_le_bytes(), across.to_le_bytes()]
                    })
                    .flatten()
                    .collect()
            } else {
                voxels
                    .flat_map(|(&v, &d)| {
                        u32::from_le_bytes([v as u8, d[0], d[1], d[2]]).to_le_bytes()
                    })
                    .collect()
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
                    bytes_per_row: Some(seg.width * if wide { 8 } else { 4 }),
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
            self.labels = label_volume(
                device,
                &self.labels_layout,
                [1, 1, 1],
                format,
                &self.mask.view,
            )
            .1;
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
        labels: Option<(&Segmentation, &LabelMask)>,
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
        if let Some((seg, mask)) = labels {
            self.mask.update(queue, mask);
            let smooth = uniforms.labels[3] as u32 & 1 != 0;
            if self.labels_for != Some((seg.id, smooth))
                && let Err(e) = self.upload_labels(device, queue, seg, smooth)
            {
                notice = Some(e);
            }
        }
        self.ensure_offscreen(device, size);
        let mut u = uniforms;
        u.look[2] = if self.srgb_target { 1.0 } else { 0.0 };
        if let Some((_, mask)) = labels
            && self.labels_ok
        {
            let version = (mask.generation % (1 << 24)) as f32;
            let flags = (u.labels[3] as u32 & 1) | if self.labels_wide { 2 } else { 0 };
            u.labels = [1.0, volume.stride as f32, version, flags as f32];
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

/// A 3D texture of labels, R32Uint or Rg32Uint, with its bind group. A new
/// texture is all zeros.
fn label_volume(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    size: [u32; 3],
    format: wgpu::TextureFormat,
    mask: &wgpu::TextureView,
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
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(mask),
            },
        ],
    });
    (texture, group)
}

/// One frame of the 3D view.
pub struct VolumeCallback {
    pub volume: Arc<Volume>,
    /// Segmentation labels to colour the volume with, and which are shown.
    pub labels: Option<(Arc<Segmentation>, Arc<LabelMask>)>,
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
            self.labels.as_ref().map(|(s, m)| (&**s, &**m)),
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

    /// Labels at twice the volume's resolution across: label 5 a block
    /// in front (low slices are nearer the viewer face on), label 6 one
    /// straight behind it, label 7 off to the side.
    fn blocks() -> Segmentation {
        let (n, d) = (64usize, 32usize);
        let mut voxels = vec![0u16; n * n * d];
        let mut fill = |x: std::ops::Range<usize>,
                        y: std::ops::Range<usize>,
                        z: std::ops::Range<usize>,
                        l: u16| {
            for k in z {
                for j in y.clone() {
                    for i in x.clone() {
                        voxels[(k * n + j) * n + i] = l;
                    }
                }
            }
        };
        fill(20..30, 20..30, 4..8, 5);
        fill(20..30, 20..30, 20..24, 6);
        fill(44..54, 40..50, 10..14, 7);
        Segmentation::new(String::new(), String::new(), [n, n, d], voxels, Vec::new())
    }

    /// Names go where the ray caster draws: the label under a voxel's
    /// place in the view is that voxel's, from any angle, and one structure
    /// in front of another hides it; with smooth surfaces or without.
    #[test]
    fn projection_agrees_with_the_line_of_sight() {
        let seg = blocks();
        let all = LabelMask::hiding(&Default::default());
        let anchor = |l: u16| seg.structures.iter().find(|s| s.label == l).unwrap().anchor;
        for smooth in [false, true] {
            for (yaw, pitch) in [(0.0, 0.0), (0.6, -0.3), (2.0, 0.4)] {
                let scene = Scene {
                    yaw,
                    pitch,
                    smooth,
                    ..scene()
                };
                let u = VolumeUniforms::new(&scene, 32.0, 32.0, 32.0);
                let p = Projection::new(&scene, &u, 1);
                let at = p.voxel_in_view(&seg, anchor(7));
                assert_eq!(p.label_under(&seg, &all, at), Some(7), "yaw {yaw}");
            }
            let scene = Scene { smooth, ..scene() };
            let u = VolumeUniforms::new(&scene, 32.0, 32.0, 32.0);
            let p = Projection::new(&scene, &u, 1);
            // Face on, the front block covers the one behind.
            assert_eq!(
                p.label_under(&seg, &all, p.voxel_in_view(&seg, anchor(6))),
                Some(5)
            );
            assert!(p.behind_another(&seg, &all, 6, anchor(6)));
            assert!(!p.behind_another(&seg, &all, 5, anchor(5)));
            assert!(!p.behind_another(&seg, &all, 7, anchor(7)));
            // Hiding the front block uncovers the other.
            let hidden = LabelMask::hiding(&[5].into_iter().collect());
            assert_eq!(
                p.label_under(&seg, &hidden, p.voxel_in_view(&seg, anchor(6))),
                Some(6)
            );
            assert!(!p.behind_another(&seg, &hidden, 6, anchor(6)));
            // Off the volume, nothing.
            assert_eq!(p.label_under(&seg, &all, (1.0, 1.0)), None);
        }
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
            smooth: false,
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
        labels: Option<(&Segmentation, &LabelMask)>,
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
        let all = LabelMask::hiding(&Default::default());
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
            (
                Scene {
                    smooth: true,
                    ..scene()
                },
                "smooth",
            ),
        ] {
            let image = render_rgba(
                &mut renderer,
                &device,
                &queue,
                &volume,
                &scene,
                Some((&seg, &all)),
            );
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
        // So it does with label 3 hidden, and the view redraws for it.
        let hidden = LabelMask::hiding(&[3].into_iter().collect());
        let image = render_rgba(
            &mut renderer,
            &device,
            &queue,
            &volume,
            &scene(),
            Some((&seg, &all)),
        );
        assert!(is_label(image[15 * 64 + 15]), "shown again");
        let image = render_rgba(
            &mut renderer,
            &device,
            &queue,
            &volume,
            &scene(),
            Some((&seg, &hidden)),
        );
        assert!(
            image[15 * 64 + 15][1] < 10,
            "hidden: {:?}",
            image[15 * 64 + 15]
        );
    }
    /// The shader's smooth surfaces are where the CPU's copy of its sums
    /// puts them: from an angle, each pixel shows the label the CPU finds
    /// under it. On the edges the rays, which step a voxel at a time, may
    /// miss a sliver that the CPU, stepping half that, finds. Skipped when
    /// there is no GPU.
    #[test]
    fn smooth_surfaces_match_the_cpu() {
        let Some((adapter, device, queue)) = testing::gpu() else {
            eprintln!("no GPU: skipped");
            return;
        };
        let mut renderer = VolumeRenderer::new(&device, &adapter, wgpu::TextureFormat::Rgba8Unorm);
        // A dark scan, so the labels show in their own colours.
        let volume = Arc::new(Volume::from_values(32, 32, 32, &vec![0.0; 32 * 32 * 32]));
        let seg = blocks();
        let all = LabelMask::hiding(&Default::default());
        let colours: Vec<(u16, [f32; 3])> = [5, 6, 7]
            .into_iter()
            .map(|l| (l, crate::colormap::label_colour(l).map(|v| v * 255.0)))
            .collect();
        for (yaw, pitch) in [(0.6, -0.3), (2.0, 0.4)] {
            let scene = Scene {
                yaw,
                pitch,
                smooth: true,
                ..scene()
            };
            let image = render_rgba(
                &mut renderer,
                &device,
                &queue,
                &volume,
                &scene,
                Some((&seg, &all)),
            );
            let u = VolumeUniforms::new(&scene, 32.0, 32.0, 32.0);
            let p = Projection::new(&scene, &u, 1);
            let want: Vec<Option<u16>> = (0..64 * 64)
                .map(|i| p.label_under(&seg, &all, ((i % 64) as f32 + 0.5, (i / 64) as f32 + 0.5)))
                .collect();
            let (mut inner, mut edges, mut missed) = (0, 0, 0);
            for y in 1..63 {
                for x in 1..63 {
                    let px = image[y * 64 + x];
                    // What the pixel shows: a label's colour, or black.
                    let near = colours
                        .iter()
                        .map(|(l, c)| {
                            let d: f32 = (0..3).map(|k| (px[k] as f32 - c[k]).powi(2)).sum();
                            (d, *l)
                        })
                        .min_by(|a, b| a.0.total_cmp(&b.0))
                        .unwrap();
                    let shown = if near.0 < 30.0 * 30.0 {
                        Some(near.1)
                    } else if px[..3].iter().all(|&c| c < 10) {
                        None
                    } else {
                        continue;
                    };
                    let w = want[y * 64 + x];
                    let edge = [(0, 1), (2, 1), (1, 0), (1, 2)]
                        .iter()
                        .any(|&(i, j)| want[(y + j - 1) * 64 + x + i - 1] != w);
                    if edge {
                        edges += 1;
                        missed += (shown != w) as usize;
                    } else {
                        inner += w.is_some() as usize;
                        assert_eq!(shown, w, "yaw {yaw}: pixel ({x}, {y}) {px:?}");
                    }
                }
            }
            assert!(inner > 80, "yaw {yaw}: {inner} labelled pixels inside");
            assert!(
                missed * 3 < edges,
                "yaw {yaw}: {missed} of {edges} edge pixels disagree"
            );
        }
    }
}

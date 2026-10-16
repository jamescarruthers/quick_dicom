//! Runs a network on the GPU with wgpu compute shaders, on its own device
//! so long runs never hold up drawing.

use std::collections::HashMap;

use eframe::egui_wgpu::wgpu;
use wgpu::util::DeviceExt;

use super::net::{GROUP_NORM_EPS, Id, Net, Op};

const SOURCE: &str = include_str!("gpu.wgsl");

/// The shared header of gpu.wgsl, then the section for `name`.
fn source(name: &str) -> String {
    section(SOURCE, name)
}

/// The header of `text`, then its section `name`. Lines may end in CRLF,
/// as git checks files out on Windows.
fn section(text: &str, name: &str) -> String {
    let mut parts = text.split("//@ ");
    let header = parts.next().unwrap_or_default();
    let section = parts
        .find(|p| p.starts_with(name) && p[name.len()..].starts_with(['\n', '\r']))
        .unwrap_or_else(|| panic!("gpu.wgsl has no section {name}"));
    // The section's name line stays, as a comment.
    format!("{header}// {section}")
}

struct Kernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

/// A kernel whose bindings are a uniform, then storage buffers, each read
/// only or not.
fn kernel(device: &wgpu::Device, name: &str, storage: &[bool]) -> Kernel {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(source(name).into()),
    });
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];
    for (i, &read_only) in storage.iter().enumerate() {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: i as u32 + 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(name),
        entries: &entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(name),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(name),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    Kernel { pipeline, layout }
}

struct Kernels {
    conv: Kernel,
    gn_partial: Kernel,
    gn_final: Kernel,
    gn_apply: Kernel,
    add: Kernel,
    upsample: Kernel,
}

/// One step of a run, in order.
enum Command {
    Dispatch {
        kernel: usize,
        group: wgpu::BindGroup,
        workgroups: [u32; 3],
    },
    Copy {
        from: usize,
        to: usize,
        offset: u64,
        size: u64,
    },
}

/// Everything a run needs for one input size: buffers, bound in advance.
struct Plan {
    side: usize,
    slots: Vec<wgpu::Buffer>,
    input: usize,
    commands: Vec<Command>,
    /// Pieces of the output: the slot, the first class, the classes, and
    /// a buffer to read them through.
    outputs: Vec<(usize, usize, usize, wgpu::Buffer)>,
}

pub struct GpuNet {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub adapter: String,
    kernels: Kernels,
    net: Net,
    plan: Option<Plan>,
}

/// Voxels per convolution dispatch. A slow GPU must finish each dispatch
/// well within the two seconds after which Windows resets it.
const SLICE: usize = 32 * 1024;

/// Workgroups for `count`, folded into two dimensions when there are too
/// many for one.
fn fold(count: usize) -> [u32; 2] {
    let across = count.clamp(1, 65_535);
    [across as u32, count.div_ceil(across) as u32]
}

impl GpuNet {
    /// The network on the best GPU. A software GPU only if `software`.
    pub fn new(net: &Net, software: bool) -> Result<GpuNet, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|_| "no GPU that can run compute shaders".to_string())?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu && !software {
            return Err(format!("{} is a software GPU", info.name));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("AI"),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|e| format!("the GPU refused: {e}"))?;
        let kernels = Kernels {
            conv: kernel(&device, "conv", &[true, true, true, false]),
            gn_partial: kernel(&device, "gn_partial", &[true, false]),
            gn_final: kernel(&device, "gn_final", &[true, false]),
            gn_apply: kernel(&device, "gn_apply", &[true, true, true, false]),
            add: kernel(&device, "add", &[true, true, false]),
            upsample: kernel(&device, "upsample", &[true, false]),
        };
        Ok(GpuNet {
            device,
            queue,
            adapter: info.name,
            kernels,
            net: net.clone(),
            plan: None,
        })
    }

    fn kernel(&self, k: usize) -> &Kernel {
        let k_ = &self.kernels;
        [
            &k_.conv,
            &k_.gn_partial,
            &k_.gn_final,
            &k_.gn_apply,
            &k_.add,
            &k_.upsample,
        ][k]
    }

    fn uniform(&self, p: [u32; 16]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&p),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    fn storage(&self, data: &[f32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    fn bind(&self, k: usize, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.kernel(k).layout,
            entries: &entries,
        })
    }

    /// Lays out the buffers and commands for cubes of `side`.
    fn plan(&self, side: usize) -> Result<Plan, String> {
        let layers = &self.net.layers;
        let n = layers.len();
        let limits = self.device.limits();
        let max_binding = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);

        // A PReLU straight after a convolution it alone reads is done by
        // the convolution.
        let mut readers = vec![0usize; n];
        for l in layers {
            for x in super::net::inputs(&l.op) {
                readers[x] += 1;
            }
        }
        let mut alias: Vec<Id> = (0..n).collect();
        let mut activation: HashMap<Id, f32> = HashMap::new();
        for (i, l) in layers.iter().enumerate() {
            if let Op::PRelu { x, slope } = l.op
                && matches!(layers[x].op, Op::Conv { .. } | Op::ConvTranspose { .. })
                && readers[x] == 1
            {
                activation.insert(x, slope);
                alias[i] = x;
            }
        }
        let value = |id: Id| alias[id];
        let mut last_use = vec![0usize; n];
        for (i, l) in layers.iter().enumerate() {
            for x in super::net::inputs(&l.op) {
                last_use[value(x)] = i;
            }
        }
        last_use[value(self.net.output)] = usize::MAX;

        let sizes: Vec<[usize; 3]> = layers.iter().map(|l| [side >> l.level; 3]).collect();
        if sizes
            .iter()
            .zip(layers)
            .any(|(s, l)| s[0] << l.level != side)
        {
            return Err(format!("cubes of {side} do not suit this network"));
        }
        let voxels = |id: Id| sizes[id].iter().product::<usize>();

        // Value buffers, reused once their values are no longer needed.
        let mut slots: Vec<wgpu::Buffer> = Vec::new();
        let mut free: Vec<(u64, usize)> = Vec::new();
        let mut slot_of: Vec<usize> = vec![usize::MAX; n];
        let device = &self.device;
        let take = |bytes: u64,
                    slots: &mut Vec<wgpu::Buffer>,
                    free: &mut Vec<(u64, usize)>|
         -> Result<usize, String> {
            if bytes > max_binding {
                return Err(format!(
                    "the GPU cannot hold {} MB in one buffer",
                    bytes >> 20
                ));
            }
            if let Some(i) = free.iter().position(|&(b, _)| b == bytes) {
                return Ok(free.swap_remove(i).1);
            }
            slots.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            Ok(slots.len() - 1)
        };

        let mut commands = Vec::new();
        let mut outputs = Vec::new();
        let mut input = 0;
        for (i, layer) in layers.iter().enumerate() {
            if alias[i] != i {
                continue;
            }
            let bytes = (layer.channels * voxels(i) * 4) as u64;
            let s = sizes[i].map(|v| v as u32);
            let out_is_split = i == self.net.output && bytes > max_binding;
            if !out_is_split {
                slot_of[i] = take(bytes, &mut slots, &mut free)?;
            }
            let y = slot_of[i];
            // Layer sizes above assume a transposed convolution doubles.
            if let Op::ConvTranspose {
                k,
                stride,
                pad,
                out_pad,
                ..
            } = layer.op
                && (stride != 2 || out_pad != 1 || k != 2 * pad + 1)
            {
                return Err(
                    "the GPU code handles only transposed convolutions that double the size".into(),
                );
            }
            match &layer.op {
                Op::Input => input = y,
                Op::Conv {
                    x,
                    weight,
                    bias,
                    k,
                    stride,
                    pad,
                }
                | Op::ConvTranspose {
                    x,
                    weight,
                    bias,
                    k,
                    stride,
                    pad,
                    ..
                } => {
                    let transposed = matches!(layer.op, Op::ConvTranspose { .. });
                    let x = value(*x);
                    let cin = layers[x].channels;
                    let cout = layer.channels;
                    let taps = k * k * k;
                    // Weights by blocks of eight outputs: [block][in][tap][8].
                    let at = |o: usize, ci: usize, t: usize| {
                        if transposed {
                            weight[(ci * cout + o) * taps + t]
                        } else {
                            weight[(o * cin + ci) * taps + t]
                        }
                    };
                    let (act, slope) = match activation.get(&i) {
                        Some(&s) => (1u32, s),
                        None => (0, 0.0),
                    };
                    let flags = transposed as u32 | (act << 1);
                    // Pieces of output channels, each small enough for one
                    // buffer; one piece unless this is a large output.
                    let per_class = (voxels(i) * 4) as u64;
                    let piece = if out_is_split {
                        ((max_binding / per_class) as usize / 8 * 8).max(8)
                    } else {
                        cout
                    };
                    for first in (0..cout).step_by(piece) {
                        let count = piece.min(cout - first);
                        let blocks = count.div_ceil(8);
                        let mut w = vec![0f32; blocks * cin * taps * 8];
                        let mut b = vec![0f32; blocks * 8];
                        for o in 0..count {
                            b[o] = bias[first + o];
                            for ci in 0..cin {
                                for t in 0..taps {
                                    w[((o / 8 * cin + ci) * taps + t) * 8 + o % 8] =
                                        at(first + o, ci, t);
                                }
                            }
                        }
                        let out = if out_is_split {
                            take(count as u64 * per_class, &mut slots, &mut free)?
                        } else {
                            y
                        };
                        let xs = sizes[x].map(|v| v as u32);
                        let (wb, bb) = (self.storage(&w), self.storage(&b));
                        // Slices of voxels, so no one dispatch runs long.
                        for first_voxel in (0..voxels(i)).step_by(SLICE) {
                            let wg = fold((voxels(i) - first_voxel).min(SLICE).div_ceil(64));
                            let params = [
                                xs[0],
                                xs[1],
                                xs[2],
                                cin as u32,
                                s[0],
                                s[1],
                                s[2],
                                count as u32,
                                *k as u32,
                                *stride as u32,
                                *pad as u32,
                                flags,
                                slope.to_bits(),
                                wg[0],
                                first_voxel as u32,
                                0,
                            ];
                            let ub = self.uniform(params);
                            let group =
                                self.bind(0, &[&ub, &slots[slot_of[x]], &wb, &bb, &slots[out]]);
                            commands.push(Command::Dispatch {
                                kernel: 0,
                                group,
                                workgroups: [wg[0], wg[1], blocks as u32],
                            });
                        }
                        if out_is_split {
                            outputs.push((
                                out,
                                first,
                                count,
                                self.staging(count as u64 * per_class),
                            ));
                        }
                    }
                }
                Op::PRelu { .. } => return Err("a PReLU the GPU code cannot fold".into()),
                Op::GroupNorm {
                    x,
                    groups,
                    gamma,
                    beta,
                    relu,
                } => {
                    let x = value(*x);
                    let per_group = layer.channels / groups * voxels(i);
                    let stretch = 256 * 64;
                    let stretches = per_group.div_ceil(stretch);
                    let partial = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: (groups * stretches * 16) as u64,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    });
                    let stats = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: (groups * 8) as u64,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    });
                    let p1 = self.uniform([
                        per_group as u32,
                        stretch as u32,
                        stretches as u32,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]);
                    let group = self.bind(1, &[&p1, &slots[slot_of[x]], &partial]);
                    commands.push(Command::Dispatch {
                        kernel: 1,
                        group,
                        workgroups: [stretches as u32, *groups as u32, 1],
                    });
                    let p2 = self.uniform([
                        stretches as u32,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        GROUP_NORM_EPS.to_bits(),
                        0,
                        0,
                        0,
                    ]);
                    let group = self.bind(2, &[&p2, &partial, &stats]);
                    commands.push(Command::Dispatch {
                        kernel: 2,
                        group,
                        workgroups: [*groups as u32, 1, 1],
                    });
                    let total = layer.channels * voxels(i);
                    let wg = fold(total.div_ceil(256));
                    let affine: Vec<f32> =
                        gamma.iter().zip(beta).flat_map(|(g, b)| [*g, *b]).collect();
                    let ab = self.storage(&affine);
                    let cpg = layer.channels / groups;
                    let p3 = self.uniform([
                        total as u32,
                        voxels(i) as u32,
                        cpg as u32,
                        wg[0],
                        *relu as u32,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]);
                    let group = self.bind(3, &[&p3, &slots[slot_of[x]], &stats, &ab, &slots[y]]);
                    commands.push(Command::Dispatch {
                        kernel: 3,
                        group,
                        workgroups: [wg[0], wg[1], 1],
                    });
                }
                Op::Add { a, b } => {
                    let (a, b) = (value(*a), value(*b));
                    let total = layer.channels * voxels(i);
                    let wg = fold(total.div_ceil(256));
                    let p = self.uniform([
                        total as u32,
                        wg[0],
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]);
                    let group =
                        self.bind(4, &[&p, &slots[slot_of[a]], &slots[slot_of[b]], &slots[y]]);
                    commands.push(Command::Dispatch {
                        kernel: 4,
                        group,
                        workgroups: [wg[0], wg[1], 1],
                    });
                }
                Op::Concat { a, b } => {
                    let (a, b) = (value(*a), value(*b));
                    let first = (layers[a].channels * voxels(a) * 4) as u64;
                    let second = (layers[b].channels * voxels(b) * 4) as u64;
                    commands.push(Command::Copy {
                        from: slot_of[a],
                        to: y,
                        offset: 0,
                        size: first,
                    });
                    commands.push(Command::Copy {
                        from: slot_of[b],
                        to: y,
                        offset: first,
                        size: second,
                    });
                }
                Op::Upsample { x } => {
                    let x = value(*x);
                    let xs = sizes[x].map(|v| v as u32);
                    let total = layer.channels * voxels(i);
                    let wg = fold(total.div_ceil(256));
                    let p = self.uniform([
                        xs[0],
                        xs[1],
                        xs[2],
                        layer.channels as u32,
                        wg[0],
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]);
                    let group = self.bind(5, &[&p, &slots[slot_of[x]], &slots[y]]);
                    commands.push(Command::Dispatch {
                        kernel: 5,
                        group,
                        workgroups: [wg[0], wg[1], 1],
                    });
                }
            }
            if i == self.net.output && !out_is_split {
                outputs.push((y, 0, layer.channels, self.staging(bytes)));
            }
            // Inputs read for the last time free their buffers.
            for x in super::net::inputs(&layer.op) {
                let x = value(x);
                if last_use[x] == i
                    && slot_of[x] != usize::MAX
                    && !free.iter().any(|&(_, s)| s == slot_of[x])
                {
                    free.push((slots[slot_of[x]].size(), slot_of[x]));
                }
            }
        }
        Ok(Plan {
            side,
            slots,
            input,
            commands,
            outputs,
        })
    }

    fn staging(&self, bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Scores for each class, class by class, for a cube of `side`.
    pub fn logits(&mut self, input: &[f32], side: usize) -> Result<Vec<f32>, String> {
        if self.plan.as_ref().is_none_or(|p| p.side != side) {
            self.plan = None;
            let memory = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            let invalid = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let plan = self.plan(side);
            // Error scopes pop in the reverse order they were pushed.
            let validation = pollster::block_on(invalid.pop());
            let memory = pollster::block_on(memory.pop());
            if let Some(e) = memory {
                return Err(format!("the GPU ran out of memory: {e}"));
            }
            if let Some(e) = validation {
                return Err(format!("the GPU refused the model: {e}"));
            }
            self.plan = Some(plan?);
        }
        let plan = self.plan.as_ref().unwrap();
        self.queue
            .write_buffer(&plan.slots[plan.input], 0, bytemuck::cast_slice(input));
        // Commands go in batches, so no one submission runs long enough
        // for the system to think the GPU has hung.
        for batch in plan.commands.chunks(6) {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            let mut i = 0;
            while i < batch.len() {
                if let Command::Copy {
                    from,
                    to,
                    offset,
                    size,
                } = &batch[i]
                {
                    encoder.copy_buffer_to_buffer(
                        &plan.slots[*from],
                        0,
                        &plan.slots[*to],
                        *offset,
                        *size,
                    );
                    i += 1;
                    continue;
                }
                let mut pass = encoder.begin_compute_pass(&Default::default());
                while let Some(Command::Dispatch {
                    kernel,
                    group,
                    workgroups,
                }) = batch.get(i)
                {
                    pass.set_pipeline(&self.kernel(*kernel).pipeline);
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
                    i += 1;
                }
            }
            self.queue.submit([encoder.finish()]);
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for (slot, _, _, staging) in &plan.outputs {
            encoder.copy_buffer_to_buffer(&plan.slots[*slot], 0, staging, 0, staging.size());
        }
        self.queue.submit([encoder.finish()]);
        for (_, _, _, staging) in &plan.outputs {
            staging.map_async(wgpu::MapMode::Read, .., |_| {});
        }
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("the GPU stopped: {e}"))?;
        let per_class = side * side * side;
        let classes = self.net.classes();
        let mut out = vec![0f32; classes * per_class];
        for (_, first, count, staging) in &plan.outputs {
            {
                let view = staging.get_mapped_range(..).map_err(|e| e.to_string())?;
                out[first * per_class..(first + count) * per_class]
                    .copy_from_slice(bytemuck::cast_slice(&view));
            }
            staging.unmap();
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::net::{run_cpu, tests::*};

    fn worst(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn every_kernel_has_a_section() {
        // As git checks the file out on Windows.
        let crlf = SOURCE.replace("\r\n", "\n").replace('\n', "\r\n");
        for name in [
            "conv",
            "gn_partial",
            "gn_final",
            "gn_apply",
            "add",
            "upsample",
        ] {
            for text in [SOURCE, crlf.as_str()] {
                let s = section(text, name);
                assert!(s.contains("struct P"), "{name}");
                assert_eq!(s.matches("fn main").count(), 1, "{name}");
            }
        }
    }

    /// Every layer type on the GPU against the plain CPU code, on small
    /// networks with made-up weights. Runs on a software GPU too; skipped
    /// when there is none.
    #[test]
    fn matches_the_plain_cpu_code() {
        // 36 voxels a side runs the outer convolutions in two slices.
        for (net, side) in [
            (small_unet(), 8),
            (small_segresnet(), 12),
            (small_unet(), 36),
        ] {
            let mut gpu = match GpuNet::new(&net, true) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("no GPU: skipped ({e})");
                    return;
                }
            };
            let input = random_input(side * side * side, 5);
            let want = run_cpu(&net, &input, [side; 3]);
            let got = gpu.logits(&input, side).unwrap();
            let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            assert!(
                worst(&got, &want) < 1e-4 * scale.max(1.0),
                "{} on {}",
                worst(&got, &want),
                gpu.adapter
            );
            // Again, reusing the plan.
            let input = random_input(side * side * side, 6);
            let want = run_cpu(&net, &input, [side; 3]);
            let got = gpu.logits(&input, side).unwrap();
            assert!(worst(&got, &want) < 1e-4 * scale.max(1.0));
        }
    }

    /// The real models on the GPU against PyTorch, and their speed on a
    /// 96-voxel block. Run with
    /// `AI_REF=<folder> cargo test --release real_weights_on_the_gpu -- --ignored`.
    #[test]
    #[ignore]
    fn real_weights_on_the_gpu() {
        let dir = std::path::PathBuf::from(std::env::var("AI_REF").unwrap());
        let floats = |name: &str| -> Vec<f32> {
            let b = std::fs::read(dir.join(name)).unwrap();
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };
        let input = floats("input32.f32");
        for (name, file) in [("spleen", "spleen.pt"), ("wholebody", "wholebody.pt")] {
            let mut weights =
                crate::ai::torch::read(&std::fs::read(dir.join(file)).unwrap()).unwrap();
            let net = if name == "spleen" {
                crate::ai::net::unet(&mut weights, 1, 2, &[16, 32, 64, 128, 256]).unwrap()
            } else {
                crate::ai::net::segresnet(&mut weights, 1, 105, 32, &[1, 2, 2, 4], &[1, 1, 1])
                    .unwrap()
            };
            let mut gpu = GpuNet::new(&net, true).unwrap();
            let want = floats(&format!("{name}_logits32.f32"));
            let got = gpu.logits(&input, 32).unwrap();
            println!(
                "{name} on {}: worst difference {}",
                gpu.adapter,
                worst(&got, &want)
            );
            assert!(worst(&got, &want) < 1e-3);
            let block = random_input(96 * 96 * 96, 9);
            gpu.logits(&block, 96).unwrap();
            let start = std::time::Instant::now();
            gpu.logits(&block, 96).unwrap();
            println!("{name}: {:.2} s for 96^3", start.elapsed().as_secs_f32());
        }
    }
}

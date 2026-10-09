//! Helpers shared by tests.

use eframe::egui_wgpu::wgpu;

use crate::decode::{Frame, Pixels};

pub fn gray(width: u32, height: u32, values: Vec<f32>) -> Frame {
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

pub fn colour(width: u32, height: u32, rgba: Vec<u8>) -> Frame {
    Frame {
        pixels: Pixels::Rgba(rgba),
        ..gray(width, height, Vec::new())
    }
}

/// Repeatable Gaussian noise: xorshift and Box-Muller.
pub fn gaussian(n: usize, sd: f32, seed: u64) -> Vec<f32> {
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

/// A GPU, or `None` when there is none, as on most CI runners.
pub fn gpu() -> Option<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let options = wgpu::RequestAdapterOptions::default();
    let adapter = pollster::block_on(instance.request_adapter(&options)).ok()?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("the adapter gives a device");
    Some((adapter, device, queue))
}

/// A frame as the viewer uploads it: R32Float for greyscale, Rgba8Unorm for
/// colour. Rows must come to a multiple of 256 bytes, so 64 pixels across.
pub fn upload(device: &wgpu::Device, queue: &wgpu::Queue, frame: &Frame) -> wgpu::TextureView {
    let (format, bytes): (_, &[u8]) = match &frame.pixels {
        Pixels::Gray(v) => (wgpu::TextureFormat::R32Float, bytemuck::cast_slice(v)),
        Pixels::Rgba(v) => (wgpu::TextureFormat::Rgba8Unorm, v),
    };
    let size = extent(frame.width, frame.height);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        bytes,
        layout(frame.width, frame.height),
        size,
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Submit `encoder` with a copy of `texture` (4 bytes a pixel) appended,
/// and return the copied bytes.
pub fn read(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    texture: &wgpu::Texture,
) -> Vec<u8> {
    let (w, h) = (texture.width(), texture.height());
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (w * h * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: layout(w, h),
        },
        extent(w, h),
    );
    queue.submit([encoder.finish()]);
    buffer.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the GPU finishes");
    buffer.get_mapped_range(..).unwrap().to_vec()
}

fn extent(width: u32, height: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    }
}

fn layout(width: u32, height: u32) -> wgpu::TexelCopyBufferLayout {
    wgpu::TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(width * 4),
        rows_per_image: Some(height),
    }
}

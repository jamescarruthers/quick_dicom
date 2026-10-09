//! Builds a 3D volume from a stack in the background, for the 3D view.
//!
//! Slices keep their raw values (as 16-bit floats), so window, level,
//! inversion and colour maps still apply on the GPU. Large stacks are
//! reduced to at most 512 pixels across and 256 slices.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crossbeam_channel::Receiver;
use rayon::prelude::*;

use crate::decode::{Frame, Pixels, decode_frame};
use crate::scan::{FrameRef, Stack, open_dicom};

/// Largest side of a slice in the volume.
const MAX_SIDE: u32 = 512;

pub struct Volume {
    /// Which stack this came from.
    pub key: String,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// Half-precision floats, slice after slice, row after row.
    pub voxels: Vec<u16>,
    /// Distance between slices in units of in-plane pixels; `None` when
    /// the files do not say (a cine loop, say).
    pub slice_gap: Option<f32>,
    /// Every how many stack frames one slice of the volume was taken.
    pub stride: usize,
    pub window: (f32, f32),
    pub inverted: bool,
    /// Frames that could not be decoded, left empty.
    pub failed: usize,
}

/// A volume being built on a background thread. Dropping it cancels.
pub struct VolumeJob {
    pub key: String,
    pub total: usize,
    done: Arc<AtomicUsize>,
    cancel: Arc<AtomicBool>,
    result: Receiver<Result<Volume, String>>,
}

impl Drop for VolumeJob {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl VolumeJob {
    pub fn start(stack: &Stack, max_layers: u32, ctx: eframe::egui::Context) -> Self {
        let stride = stack
            .frames
            .len()
            .div_ceil(max_layers.max(2) as usize)
            .max(1);
        let frames: Vec<FrameRef> = stack.frames.iter().step_by(stride).cloned().collect();
        let total = frames.len();
        let done = Arc::new(AtomicUsize::new(0));
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, result) = crossbeam_channel::bounded(1);
        let key = stack.key.clone();
        let (d, c, k) = (done.clone(), cancel.clone(), key.clone());
        std::thread::Builder::new()
            .name("volume".into())
            .spawn(move || {
                let progress = || ctx.request_repaint();
                let out = build(k, &frames, stride, &d, &c, &progress);
                let _ = tx.send(out);
                ctx.request_repaint();
            })
            .expect("failed to spawn volume thread");
        VolumeJob {
            key,
            total,
            done,
            cancel,
            result,
        }
    }

    pub fn done(&self) -> usize {
        self.done.load(Ordering::Relaxed)
    }

    pub fn poll(&self) -> Option<Result<Volume, String>> {
        self.result.try_recv().ok()
    }
}

fn build(
    key: String,
    frames: &[FrameRef],
    stride: usize,
    done: &AtomicUsize,
    cancel: &AtomicBool,
    progress: &(dyn Fn() + Sync),
) -> Result<Volume, String> {
    let first = frames.first().ok_or("there are no frames")?;
    let shared = if first.multi {
        Some(Arc::new(open_dicom(&first.path, false)?))
    } else {
        None
    };
    let load = |f: &FrameRef| match &shared {
        Some(obj) if f.path == first.path => decode_frame(obj, f.frame, u32::MAX),
        _ => decode_frame(&open_dicom(&f.path, false)?, f.frame, u32::MAX),
    };

    let head = load(first)?;
    let factor = head.width.max(head.height).div_ceil(MAX_SIDE).max(1);
    let (width, height) = (head.width / factor, head.height / factor);
    let layer_len = (width * height) as usize;
    let mut voxels = vec![0u16; layer_len * frames.len()];
    let mut positions = vec![None; frames.len()];
    let mut failed = 0;

    let batch = rayon::current_num_threads().max(1) * 2;
    for (c, chunk) in frames.chunks(batch).enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let layers: Vec<Option<(Vec<u16>, Option<f32>)>> = chunk
            .par_iter()
            .map(|f| {
                let frame = load(f).ok()?;
                if (frame.width, frame.height) != (head.width, head.height) {
                    return None;
                }
                let layer = shrink(&frame, factor)
                    .iter()
                    .map(|&v| f16_bits(v))
                    .collect();
                Some((layer, frame.position))
            })
            .collect();
        for (k, layer) in layers.into_iter().enumerate() {
            let i = c * batch + k;
            match layer {
                Some((data, position)) => {
                    voxels[i * layer_len..(i + 1) * layer_len].copy_from_slice(&data);
                    positions[i] = position;
                }
                None => failed += 1,
            }
            done.fetch_add(1, Ordering::Relaxed);
        }
        progress();
    }

    // Slice spacing from the first and last known positions, in units of
    // the volume's in-plane pixels.
    let known: Vec<(usize, f32)> = positions
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.map(|p| (i, p)))
        .collect();
    let slice_gap = match (known.first(), known.last(), head.spacing) {
        (Some(&(i0, p0)), Some(&(i1, p1)), Some(sp)) if i1 > i0 => {
            let gap_mm = (p1 - p0).abs() / (i1 - i0) as f32;
            let pixel_mm = sp.col * factor as f32;
            (gap_mm > 0.0 && pixel_mm > 0.0).then_some(gap_mm / pixel_mm)
        }
        _ => None,
    };

    Ok(Volume {
        key,
        width,
        height,
        depth: frames.len() as u32,
        voxels,
        slice_gap,
        stride,
        window: head.window,
        inverted: head.inverted,
        failed,
    })
}

/// Greyscale values of a frame, averaged down by `factor` in each direction.
/// Colour frames become their luminance.
fn shrink(frame: &Frame, factor: u32) -> Vec<f32> {
    let (sw, f) = (frame.width as usize, factor as usize);
    let (w, h) = (sw / f, frame.height as usize / f);
    let value = |i: usize| match &frame.pixels {
        Pixels::Gray(v) => v[i],
        Pixels::Rgba(v) => {
            0.2126 * v[i * 4] as f32 + 0.7152 * v[i * 4 + 1] as f32 + 0.0722 * v[i * 4 + 2] as f32
        }
    };
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for dy in 0..f {
                for dx in 0..f {
                    acc += value((y * f + dy) * sw + x * f + dx);
                }
            }
            out.push(acc / (f * f) as f32);
        }
    }
    out
}

/// IEEE half-precision bits for `v`, rounded to nearest. Values beyond the
/// half range clamp to its largest finite value.
pub fn f16_bits(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7E00;
    }
    let sign = if v.is_sign_negative() { 0x8000 } else { 0 };
    let a = v.abs();
    if a >= 65504.0 {
        return sign | 0x7BFF;
    }
    if a < 6.103_515_6e-5 {
        // Subnormal: a multiple of 2^-24.
        return sign | (a * 16_777_216.0).round() as u16;
    }
    let bits = a.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mantissa = bits & 0x7F_FFFF;
    let mut half = ((exponent as u32) << 10) | (mantissa >> 13);
    // Round half up on the dropped bits; a carry correctly bumps the exponent.
    if mantissa & 0x1000 != 0 {
        half += 1;
    }
    sign | (half.min(0x7BFF) as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_floats() {
        assert_eq!(f16_bits(0.0), 0x0000);
        assert_eq!(f16_bits(1.0), 0x3C00);
        assert_eq!(f16_bits(-2.0), 0xC000);
        assert_eq!(f16_bits(0.5), 0x3800);
        assert_eq!(f16_bits(-1024.0), 0xE400);
        // Between 2048 and 4096 halves step by 2; 3071 rounds up to 3072.
        assert_eq!(f16_bits(3071.0), 0x6A00);
        assert_eq!(f16_bits(65504.0), 0x7BFF);
        assert_eq!(f16_bits(1e9), 0x7BFF);
        assert_eq!(f16_bits(5.960_464_5e-8), 0x0001);
    }

    #[test]
    fn shrink_averages_blocks_and_takes_luminance() {
        let frame = Frame {
            id: 0,
            width: 2,
            height: 2,
            pixels: Pixels::Gray(vec![0.0, 2.0, 4.0, 6.0]),
            min: 0.0,
            max: 6.0,
            window: (3.0, 6.0),
            inverted: false,
            spacing: None,
            position: None,
        };
        assert_eq!(shrink(&frame, 2), vec![3.0]);
        let colour = Frame {
            width: 1,
            height: 1,
            pixels: Pixels::Rgba(vec![255, 255, 255, 255]),
            ..frame
        };
        assert!((shrink(&colour, 1)[0] - 255.0).abs() < 1e-3);
    }
}

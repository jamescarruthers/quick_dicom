//! Saves a stack as an H.264 MP4 video, drawn with the viewer's current
//! window, level, inversion and smoothing. Overlay text is left out, so no
//! patient details end up in the video.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crossbeam_channel::Receiver;
use openh264::OpenH264API;
use openh264::encoder::{
    BitRate, Complexity, Encoder, EncoderConfig, FrameRate, IntraFramePeriod, QpRange,
    RateControlMode, VuiConfig,
};
use openh264::formats::YUVBuffer;
use rayon::prelude::*;

use crate::colormap::Colormap;
use crate::decode::{Frame, Pixels, decode_frame};
use crate::mp4;
use crate::scan::{FrameRef, open_dicom};

/// Ticks per second in the MP4 timeline; divides evenly by common frame rates.
const TIMESCALE: u32 = 90_000;
/// Small images are scaled up so the longer side reaches this many pixels.
const MIN_SIDE: f32 = 512.0;
/// Large images are scaled down so the longer side fits this many pixels.
const MAX_SIDE: f32 = 2048.0;

pub struct Settings {
    pub frames: Vec<FrameRef>,
    pub fps: f32,
    /// Window centre and width; `None` takes the first frame's own.
    pub window: Option<(f32, f32)>,
    pub invert: bool,
    pub smooth: bool,
    pub colormap: Colormap,
}

/// How frames are drawn: the viewer's display settings.
#[derive(Clone, Copy)]
struct Look {
    window: (f32, f32),
    invert: bool,
    smooth: bool,
    colormap: Colormap,
}

pub struct Summary {
    pub frames: usize,
    pub bytes: usize,
    /// Frames that could not be decoded and were written black.
    pub failed: usize,
}

/// A video being written on a background thread. Dropping it cancels.
pub struct Export {
    pub path: PathBuf,
    pub total: usize,
    done: Arc<AtomicUsize>,
    cancel: Arc<AtomicBool>,
    result: Receiver<Result<Summary, String>>,
}

impl Drop for Export {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl Export {
    pub fn start(settings: Settings, path: PathBuf, ctx: eframe::egui::Context) -> Self {
        let total = settings.frames.len();
        let done = Arc::new(AtomicUsize::new(0));
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, result) = crossbeam_channel::bounded(1);
        let (d, c, p) = (done.clone(), cancel.clone(), path.clone());
        std::thread::Builder::new()
            .name("export".into())
            .spawn(move || {
                let outcome = write_video(&settings, &d, &c, &|| ctx.request_repaint()).and_then(
                    |(bytes, summary)| {
                        std::fs::write(&p, bytes).map_err(|e| e.to_string())?;
                        Ok(summary)
                    },
                );
                let _ = tx.send(outcome);
                ctx.request_repaint();
            })
            .expect("failed to spawn export thread");
        Export {
            path,
            total,
            done,
            cancel,
            result,
        }
    }

    pub fn done(&self) -> usize {
        self.done.load(Ordering::Relaxed)
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// The outcome, once the thread has finished.
    pub fn poll(&self) -> Option<Result<Summary, String>> {
        self.result.try_recv().ok()
    }
}

/// Decode, draw and encode every frame, returning the MP4 file's bytes.
pub fn write_video(
    settings: &Settings,
    done: &AtomicUsize,
    cancel: &AtomicBool,
    progress: &(dyn Fn() + Sync),
) -> Result<(Vec<u8>, Summary), String> {
    let frames = &settings.frames;
    let first = frames.first().ok_or("there are no frames to save")?;
    // A multi-frame stack is one file: read it once, not once per frame.
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
    let (width, height) = output_size(head.width, head.height);
    let look = Look {
        window: settings.window.unwrap_or(head.window),
        invert: settings.invert,
        smooth: settings.smooth,
        colormap: settings.colormap,
    };
    drop(head);

    let fps = settings.fps.clamp(0.5, 240.0);
    let mut encoder = Encoder::with_api_config(
        OpenH264API::from_source(),
        encoder_config(width, height, fps),
    )
    .map_err(|e| format!("could not start the video encoder: {e}"))?;

    let mut track = mp4::Track {
        width: width as u16,
        height: height as u16,
        timescale: TIMESCALE,
        frame_duration: ((TIMESCALE as f32 / fps).round() as u32).max(1),
        sps: Vec::new(),
        pps: Vec::new(),
        samples: Vec::with_capacity(frames.len()),
        keyframes: Vec::with_capacity(frames.len()),
    };
    let mut failed = 0;

    // Decode and draw a batch in parallel, then encode it in order.
    let batch = rayon::current_num_threads().max(1) * 2;
    for chunk in frames.chunks(batch) {
        let images: Vec<Option<Vec<u8>>> = chunk
            .par_iter()
            .map(|f| {
                let frame = load(f).ok()?;
                Some(to_yuv(&frame, width, height, &look))
            })
            .collect();
        for image in images {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            let yuv = image.unwrap_or_else(|| {
                failed += 1;
                black_yuv(width, height)
            });
            let encoded = encoder
                .encode(&YUVBuffer::from_vec(yuv, width, height))
                .map_err(|e| format!("encoding failed: {e}"))?
                .to_vec();

            let mut sample = Vec::with_capacity(encoded.len());
            let mut keyframe = false;
            for nal in mp4::split_annex_b(&encoded) {
                match nal[0] & 0x1F {
                    7 => track.sps = nal.to_vec(),
                    8 => track.pps = nal.to_vec(),
                    9 => {} // access unit delimiters carry nothing MP4 needs
                    kind => {
                        keyframe |= kind == 5;
                        sample.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                        sample.extend_from_slice(nal);
                    }
                }
            }
            if sample.is_empty() {
                return Err("the encoder dropped a frame".into());
            }
            track.samples.push(sample);
            track.keyframes.push(keyframe);
            done.fetch_add(1, Ordering::Relaxed);
            progress();
        }
    }
    if track.sps.is_empty() || track.pps.is_empty() {
        return Err("the encoder produced no parameter sets".into());
    }

    let bytes = mp4::write(&track)?;
    let summary = Summary {
        frames: track.samples.len(),
        bytes: bytes.len(),
        failed,
    };
    Ok((bytes, summary))
}

/// High quality, never dropping frames, signalled as BT.709 limited range.
fn encoder_config(width: usize, height: usize, fps: f32) -> EncoderConfig {
    let bitrate = (width as f32 * height as f32 * fps * 0.3).clamp(500_000.0, 60_000_000.0);
    EncoderConfig::new()
        .rate_control_mode(RateControlMode::Quality)
        .bitrate(BitRate::from_bps(bitrate as u32))
        .max_frame_rate(FrameRate::from_hz(fps))
        .qp(QpRange::new(8, 30))
        .skip_frames(false)
        .complexity(Complexity::High)
        // A keyframe every two seconds keeps seeking quick in players.
        .intra_frame_period(IntraFramePeriod::from_num_frames((fps * 2.0).round() as u32))
        .vui(VuiConfig::bt709())
}

/// Video size for a frame: even sides, small images scaled up by a whole
/// number, and large ones scaled down to fit.
fn output_size(w: u32, h: u32) -> (usize, usize) {
    let long = w.max(h).max(1) as f32;
    let scale = if long > MAX_SIDE {
        MAX_SIDE / long
    } else if long < MIN_SIDE {
        (MIN_SIDE / long).ceil()
    } else {
        1.0
    };
    let even = |v: u32| (((v as f32 * scale).round() as usize) & !1).max(2);
    (even(w), even(h))
}

fn black_yuv(width: usize, height: usize) -> Vec<u8> {
    let mut yuv = vec![16u8; width * height * 3 / 2];
    yuv[width * height..].fill(128);
    yuv
}

/// Draw a frame into an I420 buffer (limited range, BT.709) the way the
/// viewer's shader does: fit and centre it, sample, apply the window, then
/// the colour map.
fn to_yuv(frame: &Frame, width: usize, height: usize, look: &Look) -> Vec<u8> {
    let (centre, window_width) = look.window;
    let smooth = look.smooth;
    let (fw, fh) = (frame.width as f32, frame.height as f32);
    let scale = (width as f32 / fw).min(height as f32 / fh);
    let left = (width as f32 - fw * scale) / 2.0;
    let top = (height as f32 - fh * scale) / 2.0;
    let invert = frame.inverted != look.invert;
    let colour = frame.is_color();
    let ww = window_width.max(1e-6);
    let low = centre - 0.5 * ww;
    // Several samples per output pixel when shrinking, as in the shader.
    let taps = if smooth && scale < 1.0 {
        (1.0 / scale).ceil().min(4.0) as usize
    } else {
        1
    };

    let fetch = |x: i64, y: i64| -> [f32; 3] {
        let x = x.clamp(0, frame.width as i64 - 1) as usize;
        let y = y.clamp(0, frame.height as i64 - 1) as usize;
        let i = y * frame.width as usize + x;
        match &frame.pixels {
            Pixels::Gray(v) => [v[i]; 3],
            Pixels::Rgba(v) => [v[i * 4] as f32, v[i * 4 + 1] as f32, v[i * 4 + 2] as f32],
        }
    };
    let bilinear = |tx: f32, ty: f32| -> [f32; 3] {
        let (fx, fy) = (tx - 0.5, ty - 0.5);
        let (x0, y0) = (fx.floor(), fy.floor());
        let (wx, wy) = (fx - x0, fy - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let (a, b) = (fetch(x0, y0), fetch(x0 + 1, y0));
        let (c, d) = (fetch(x0, y0 + 1), fetch(x0 + 1, y0 + 1));
        std::array::from_fn(|k| {
            let top = a[k] + (b[k] - a[k]) * wx;
            let bottom = c[k] + (d[k] - c[k]) * wx;
            top + (bottom - top) * wy
        })
    };
    // The displayed brightness (0 to 1) of output pixel (x, y), per channel.
    let shade = |x: usize, y: usize| -> [f32; 3] {
        let tx = (x as f32 + 0.5 - left) / scale;
        let ty = (y as f32 + 0.5 - top) / scale;
        if tx < 0.0 || ty < 0.0 || tx >= fw || ty >= fh {
            return [0.0; 3];
        }
        let raw = if !smooth {
            fetch(tx as i64, ty as i64)
        } else if taps == 1 {
            bilinear(tx, ty)
        } else {
            let mut acc = [0.0; 3];
            for j in 0..taps {
                for k in 0..taps {
                    let ox = ((k as f32 + 0.5) / taps as f32 - 0.5) / scale;
                    let oy = ((j as f32 + 0.5) / taps as f32 - 0.5) / scale;
                    let s = bilinear(tx + ox, ty + oy);
                    (0..3).for_each(|c| acc[c] += s[c]);
                }
            }
            acc.map(|v| v / (taps * taps) as f32)
        };
        let shaded = raw.map(|v| {
            let g = ((v - low) / ww).clamp(0.0, 1.0);
            if invert { 1.0 - g } else { g }
        });
        if colour {
            shaded
        } else {
            look.colormap.apply(shaded[0])
        }
    };

    let mut yuv = vec![0u8; width * height * 3 / 2];
    let (luma, chroma) = yuv.split_at_mut(width * height);
    let (cb_plane, cr_plane) = chroma.split_at_mut(width * height / 4);
    for by in 0..height / 2 {
        for bx in 0..width / 2 {
            let mut sum = [0.0f32; 3];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let (x, y) = (bx * 2 + dx, by * 2 + dy);
                let [r, g, b] = shade(x, y);
                let l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                luma[y * width + x] = (16.0 + 219.0 * l).round() as u8;
                sum = [sum[0] + r, sum[1] + g, sum[2] + b];
            }
            let [r, g, b] = sum.map(|v| v / 4.0);
            let cb = -0.114_57 * r - 0.385_43 * g + 0.5 * b;
            let cr = 0.5 * r - 0.454_15 * g - 0.045_85 * b;
            let i = by * (width / 2) + bx;
            cb_plane[i] = (128.0 + 224.0 * cb).round().clamp(16.0, 240.0) as u8;
            cr_plane[i] = (128.0 + 224.0 * cr).round().clamp(16.0, 240.0) as u8;
        }
    }
    yuv
}

#[cfg(test)]
mod tests {
    use super::*;

    fn look(window: (f32, f32), invert: bool, smooth: bool) -> Look {
        Look {
            window,
            invert,
            smooth,
            colormap: Colormap::Grey,
        }
    }

    fn gray(width: u32, height: u32, values: Vec<f32>) -> Frame {
        Frame {
            id: 0,
            width,
            height,
            pixels: Pixels::Gray(values),
            min: 0.0,
            max: 255.0,
            window: (127.5, 255.0),
            inverted: false,
            spacing: None,
            position: None,
        }
    }

    #[test]
    fn sizes_are_even_and_scaled_into_range() {
        assert_eq!(output_size(512, 512), (512, 512));
        assert_eq!(output_size(641, 481), (640, 480));
        assert_eq!(output_size(128, 64), (512, 256));
        assert_eq!(output_size(4096, 3072), (2048, 1536));
    }

    #[test]
    fn applies_the_window_and_inversion() {
        // Values 0, 50, 100, 200 through a window from 0 to 200.
        let frame = gray(2, 2, vec![0.0, 50.0, 100.0, 200.0]);
        let yuv = to_yuv(&frame, 2, 2, &look((100.0, 200.0), false, false));
        assert_eq!(&yuv[..4], &[16, 71, 126, 235]);
        assert_eq!(&yuv[4..], &[128, 128]);

        let inverted = to_yuv(&frame, 2, 2, &look((100.0, 200.0), true, false));
        assert_eq!(&inverted[..4], &[235, 180, 126, 16]);
    }

    #[test]
    fn colour_uses_bt709_limited_range() {
        let frame = Frame {
            pixels: Pixels::Rgba([255, 0, 0, 255].repeat(4)),
            window: (127.5, 255.0),
            ..gray(2, 2, vec![])
        };
        let yuv = to_yuv(&frame, 2, 2, &look((127.5, 255.0), false, false));
        // Pure red in BT.709 limited range is Y 63, Cb 102, Cr 240.
        assert_eq!(&yuv[..4], &[63; 4]);
        assert_eq!(&yuv[4..], &[102, 240]);
    }

    #[test]
    fn colour_maps_apply_to_greyscale() {
        // Full brightness through hot iron is white; half is orange-red.
        let frame = gray(2, 2, vec![255.0, 255.0, 127.5, 127.5]);
        let hot = Look {
            colormap: Colormap::HotIron,
            ..look((127.5, 255.0), false, false)
        };
        let yuv = to_yuv(&frame, 2, 2, &hot);
        assert_eq!(&yuv[..2], &[235, 235]);
        // Half brightness is (1, 0.5, 0): Y = 16 + 219 * (0.2126 + 0.3576).
        assert_eq!(&yuv[2..4], &[141, 141]);
        assert!(yuv[5] > 128, "red pushes Cr above neutral");
    }

    #[test]
    fn small_images_are_centred_with_black_bars() {
        // A 2x1 image drawn into 4x4: rows 0 and 3 are bars, 1 and 2 the image.
        let frame = gray(2, 1, vec![255.0, 255.0]);
        let yuv = to_yuv(&frame, 4, 4, &look((127.5, 255.0), false, false));
        assert_eq!(&yuv[0..4], &[16; 4]);
        assert_eq!(&yuv[4..12], &[235; 8]);
        assert_eq!(&yuv[12..16], &[16; 4]);
    }

    #[test]
    fn encodes_a_playable_stream() {
        // Encode synthetic frames straight through the encoder and muxer.
        let frames: Vec<Vec<u8>> = (0..5)
            .map(|k| {
                let values = (0..64 * 64).map(|i| ((i + k * 7) % 256) as f32).collect();
                to_yuv(
                    &gray(64, 64, values),
                    64,
                    64,
                    &look((127.5, 255.0), false, true),
                )
            })
            .collect();
        let config = encoder_config(64, 64, 15.0);
        let mut encoder = Encoder::with_api_config(OpenH264API::from_source(), config).unwrap();
        let mut sps = false;
        let mut first_is_key = false;
        for (k, yuv) in frames.into_iter().enumerate() {
            let out = encoder
                .encode(&YUVBuffer::from_vec(yuv, 64, 64))
                .unwrap()
                .to_vec();
            let nals = mp4::split_annex_b(&out);
            assert!(!nals.is_empty());
            sps |= nals.iter().any(|n| n[0] & 0x1F == 7);
            if k == 0 {
                first_is_key = nals.iter().any(|n| n[0] & 0x1F == 5);
            }
        }
        assert!(sps && first_is_key);
    }
}

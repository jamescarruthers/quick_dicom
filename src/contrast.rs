//! Local contrast: contrast-limited adaptive histogram equalisation
//! (CLAHE). The image is cut into tiles, up to 8 by 8. Each tile gets a
//! curve that spreads its own range of brightness over the whole range, so
//! detail in dark and bright parts shows at once. A clip limit keeps flat
//! areas and noise from being blown up. Each pixel blends the curves of its
//! four nearest tiles, so tile edges never show.
//!
//! The curves work on the windowed image, so they are made again whenever
//! the frame or the window changes. They are made on the CPU from the
//! unfiltered frame, which is quick as only a sample of pixels is counted;
//! the display shader (`src/image.wgsl`) looks them up, and `Curves::map`
//! is its CPU twin, used for videos.
//!
//! In colour images only the grey parts change and count: colour such as
//! Doppler flow keeps its brightness, which carries meaning.

use crate::decode::{Frame, Pixels};
use crate::filter::{LUMA, Level};

/// Steps of brightness in each curve.
pub const BINS: usize = 256;
/// Most tiles along each side.
pub const MAX_TILES: usize = 8;
/// Fewest pixels along a tile's side.
const MIN_TILE: u32 = 32;
/// Most pixels counted along each side of a tile.
const SAMPLES: usize = 64;

/// How far a curve may stretch contrast, as a limit on each histogram bin
/// in multiples of the mean. 1 would leave the image as it is.
pub fn clip_limit(level: Level) -> Option<f32> {
    match level {
        Level::Off => None,
        Level::Low => Some(1.5),
        Level::Medium => Some(2.5),
        Level::High => Some(4.0),
    }
}

/// How grey a windowed colour is: 1 for grey, falling to 0 once the
/// channels differ by a tenth of the range, as in Doppler colour.
pub fn greyness(g: [f32; 3]) -> f32 {
    let spread = g[0].max(g[1]).max(g[2]) - g[0].min(g[1]).min(g[2]);
    (1.0 - 10.0 * spread).clamp(0.0, 1.0)
}

/// One curve per tile, row after row.
pub struct Curves {
    /// Tiles across and down.
    pub nx: usize,
    pub ny: usize,
    width: f32,
    height: f32,
    /// `BINS` values per tile, each the new brightness (0 to 1) for the
    /// middle of its bin.
    pub table: Vec<f32>,
}

impl Curves {
    /// Curves for `frame` seen through `window` (centre, width), in the
    /// frame's own units: raw values for greyscale, 0 to 255 for colour.
    pub fn new(frame: &Frame, window: (f32, f32), clip: f32) -> Self {
        let (w, h) = (frame.width as usize, frame.height as usize);
        let nx = (frame.width / MIN_TILE).clamp(1, MAX_TILES as u32) as usize;
        let ny = (frame.height / MIN_TILE).clamp(1, MAX_TILES as u32) as usize;
        let (centre, width) = (window.0, window.1.max(1e-6));
        let low = centre - 0.5 * width;
        let shade = |v: f32| ((v - low) / width).clamp(0.0, 1.0);
        // Brightness as the shader sees it, the window applied to each
        // channel before they are mixed, and how much the pixel counts.
        let brightness = |i: usize| match &frame.pixels {
            Pixels::Gray(v) => (shade(v[i]), 1.0),
            Pixels::Rgba(v) => {
                let g = [0, 1, 2].map(|c| shade(v[i * 4 + c] as f32));
                ((0..3).map(|c| LUMA[c] * g[c]).sum(), greyness(g))
            }
        };

        let mut table = Vec::with_capacity(nx * ny * BINS);
        for ty in 0..ny {
            for tx in 0..nx {
                let (x0, x1) = (tx * w / nx, (tx + 1) * w / nx);
                let (y0, y1) = (ty * h / ny, (ty + 1) * h / ny);
                let mut hist = [0.0f32; BINS];
                let mut count = 0.0;
                for y in (y0..y1).step_by((y1 - y0).div_ceil(SAMPLES).max(1)) {
                    for x in (x0..x1).step_by((x1 - x0).div_ceil(SAMPLES).max(1)) {
                        let (g, weight) = brightness(y * w + x);
                        let b = (g * BINS as f32) as usize;
                        hist[b.min(BINS - 1)] += weight;
                        count += weight;
                    }
                }
                if count < 1.0 {
                    // Nothing grey here: leave brightness as it is.
                    table.extend((0..BINS).map(|i| (i as f32 + 0.5) / BINS as f32));
                    continue;
                }
                // Cut every bin down to the limit and share what was cut
                // among all bins.
                let limit = (clip * count / BINS as f32).max(1.0);
                let excess: f32 = hist.iter().map(|&n| (n - limit).max(0.0)).sum();
                let share = excess / BINS as f32;
                let mut below = 0.0;
                for n in hist {
                    let n = n.min(limit) + share;
                    table.push((below + 0.5 * n) / count);
                    below += n;
                }
            }
        }
        Curves {
            nx,
            ny,
            width: frame.width as f32,
            height: frame.height as f32,
            table,
        }
    }

    /// The new brightness for `g` (0 to 1) at image position (`x`, `y`) in
    /// pixels, where pixel centres sit at half-pixel positions. Blends the
    /// four nearest tiles' curves, each read between its two nearest bins.
    pub fn map(&self, g: f32, x: f32, y: f32) -> f32 {
        let (nx, ny) = (self.nx as f32, self.ny as f32);
        let px = (x / self.width * nx - 0.5).clamp(0.0, nx - 1.0);
        let py = (y / self.height * ny - 0.5).clamp(0.0, ny - 1.0);
        let (ix, iy) = (px.floor() as usize, py.floor() as usize);
        let (fx, fy) = (px - ix as f32, py - iy as f32);
        let (jx, jy) = ((ix + 1).min(self.nx - 1), (iy + 1).min(self.ny - 1));
        let b = (g * BINS as f32 - 0.5).clamp(0.0, (BINS - 1) as f32);
        let (i, f) = (b.floor() as usize, b.fract());
        let j = (i + 1).min(BINS - 1);
        let curve = |tx: usize, ty: usize| {
            let row = &self.table[(ty * self.nx + tx) * BINS..];
            row[i] + (row[j] - row[i]) * f
        };
        let top = curve(ix, iy) + (curve(jx, iy) - curve(ix, iy)) * fx;
        let bottom = curve(ix, jy) + (curve(jx, jy) - curve(ix, jy)) * fx;
        top + (bottom - top) * fy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{colour, gaussian, gray};

    #[test]
    fn a_clip_of_one_leaves_brightness_alone() {
        let values: Vec<f32> = (0..128 * 128).map(|i| ((i * 37) % 256) as f32).collect();
        let curves = Curves::new(&gray(128, 128, values), (128.0, 256.0), 1.0);
        for g in [0.1, 0.25, 0.5, 0.9] {
            assert!((curves.map(g, 40.0, 70.0) - g).abs() < 0.01, "{g}");
        }
    }

    #[test]
    fn stretches_a_narrow_range_in_each_part_of_the_image() {
        // Left half dark (values 20 to 40), right half bright (200 to 220),
        // seen through a 0 to 255 window.
        let (w, h) = (256usize, 128usize);
        let noise = gaussian(w * h, 5.0, 9);
        let values: Vec<f32> = (0..w * h)
            .map(|i| if i % w < w / 2 { 30.0 } else { 210.0 } + noise[i])
            .collect();
        let curves = Curves::new(&gray(w as u32, h as u32, values), (127.5, 255.0), 4.0);
        assert_eq!((curves.nx, curves.ny), (8, 4));
        let spread = |x: f32, lo: f32, hi: f32| {
            curves.map(hi / 255.0, x, 64.0) - curves.map(lo / 255.0, x, 64.0)
        };
        // In each half, a step of 20 (8% of the window) becomes much larger.
        assert!(spread(40.0, 20.0, 40.0) > 0.25);
        assert!(spread(216.0, 200.0, 220.0) > 0.25);
        // Order is kept.
        let mut last = 0.0;
        for k in 0..=20 {
            let v = curves.map(k as f32 / 20.0, 40.0, 64.0);
            assert!(v >= last - 1e-6);
            last = v;
        }
    }

    #[test]
    fn black_stays_dark_and_white_stays_bright() {
        // Most of the image is black background, as outside a CT body.
        let (w, h) = (128usize, 128usize);
        let values: Vec<f32> = (0..w * h)
            .map(|i| {
                if i % w > 96 {
                    100.0 + (i % 7) as f32
                } else {
                    0.0
                }
            })
            .collect();
        let curves = Curves::new(&gray(w as u32, h as u32, values), (127.5, 255.0), 2.5);
        assert!(curves.map(0.0, 10.0, 10.0) < 0.02);
        assert!(curves.map(1.0, 10.0, 10.0) > 0.98);
    }

    #[test]
    fn colour_does_not_count() {
        // Grey speckle with a Doppler patch covering one tile: the patch
        // leaves the grey's curve as it was, and an all-colour tile keeps
        // brightness as it is.
        let (w, h) = (64usize, 64usize);
        let noise = gaussian(w * h, 20.0, 4);
        let rgba = |doppler: bool| -> Vec<u8> {
            (0..w * h)
                .flat_map(|i| {
                    if doppler && i % w < 32 && i / w < 32 {
                        [220, 40, 20, 255]
                    } else {
                        let g = (100.0 + noise[i]).clamp(0.0, 255.0) as u8;
                        [g, g, g, 255]
                    }
                })
                .collect()
        };
        let window = (127.5, 255.0);
        let plain = Curves::new(&colour(w as u32, h as u32, rgba(false)), window, 2.5);
        let mixed = Curves::new(&colour(w as u32, h as u32, rgba(true)), window, 2.5);
        assert_eq!((mixed.nx, mixed.ny), (2, 2));
        // The grey tile at the far corner is unchanged.
        assert_eq!(plain.table[3 * BINS..], mixed.table[3 * BINS..]);
        // The colour tile's curve leaves brightness alone.
        for (i, v) in mixed.table[..BINS].iter().enumerate() {
            assert!((v - (i as f32 + 0.5) / BINS as f32).abs() < 1e-6);
        }
        assert_eq!(greyness([0.5, 0.5, 0.5]), 1.0);
        assert_eq!(greyness([0.86, 0.16, 0.08]), 0.0);
    }

    #[test]
    fn small_images_get_one_tile() {
        let curves = Curves::new(&gray(20, 10, vec![1.0; 200]), (0.5, 1.0), 2.0);
        assert_eq!((curves.nx, curves.ny), (1, 1));
        assert_eq!(curves.table.len(), BINS);
    }
}

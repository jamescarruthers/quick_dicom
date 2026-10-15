//! Everything around a model, as MONAI's bundles do it: resample the scan
//! onto the model's grid, scale its values, run the model on overlapping
//! blocks, and carry the labels back onto the scan's own voxels.

use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

use crate::monai::Scan;

/// How a model wants its input.
#[derive(Clone, Copy, Debug)]
pub struct Recipe {
    /// Voxel size in millimetres along the patient's right, front and
    /// head axes (MONAI's RAS orientation).
    pub spacing: [f64; 3],
    pub intensity: Intensity,
    /// Side of the cubes the model runs on.
    pub block: usize,
    /// How much neighbouring cubes overlap, as a fraction of a side.
    pub overlap: f64,
    /// What fills a scan smaller than a cube.
    pub pad: Pad,
    /// How overlapping cubes' scores are weighted.
    pub blend: Blend,
    /// For two-class models: carry the probability back to the scan and
    /// threshold it there, for smoother edges than the grid's voxels.
    pub smooth: bool,
}

#[derive(Clone, Copy, Debug)]
pub enum Blend {
    /// All voxels of a cube count the same.
    Constant,
    /// Voxels count less towards a cube's edges.
    Gaussian,
}

#[derive(Clone, Copy, Debug)]
pub enum Intensity {
    /// Values from `lo` to `hi` map to 0 to 1, clipped.
    Window { lo: f32, hi: f32 },
    /// Mean 0 and deviation 1 over the non-zero voxels, which zeros keep,
    /// then the whole range onto -1 to 1.
    NormalizeThenRange,
}

#[derive(Clone, Copy, Debug)]
pub enum Pad {
    Constant(f32),
    Edge,
}

/// The model's grid: axis-aligned in the patient, at the model's spacing,
/// covering the scan.
#[derive(Clone, Debug)]
pub struct Grid {
    pub size: [usize; 3],
    pub origin: [f64; 3],
    pub spacing: [f64; 3],
}

impl Grid {
    pub fn covering(scan: &Scan, spacing: [f64; 3]) -> Grid {
        let [w, h, d] = scan.dims;
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for c in 0..8 {
            let v = [
                ((c & 1) * (w - 1)) as f64,
                (((c >> 1) & 1) * (h - 1)) as f64,
                (((c >> 2) & 1) * (d - 1)) as f64,
            ];
            let p = apply(&scan.affine, v);
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        // As MONAI's Spacing transform sizes it, so the cubes fall in the
        // same places.
        let size = std::array::from_fn(|a| {
            ((hi[a] - lo[a]) / spacing[a] + 1.0).round_ties_even() as usize
        });
        Grid {
            size,
            origin: lo,
            spacing,
        }
    }

    fn len(&self) -> usize {
        self.size.iter().product()
    }
}

fn apply(m: &crate::nifti::Affine, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|r| m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2] + m[r][3])
}

/// The inverse of an affine map: patient position to voxel position.
fn invert(m: &crate::nifti::Affine) -> crate::nifti::Affine {
    let a = |r: usize, c: usize| m[r][c];
    let det = a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1))
        - a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0))
        + a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
    let cof = |r: usize, c: usize| {
        let (r0, r1) = ([1, 0, 0][r], [2, 2, 1][r]);
        let (c0, c1) = ([1, 0, 0][c], [2, 2, 1][c]);
        let minor = a(r0, c0) * a(r1, c1) - a(r0, c1) * a(r1, c0);
        if (r + c).is_multiple_of(2) {
            minor
        } else {
            -minor
        }
    };
    std::array::from_fn(|r| {
        let row: [f64; 3] = std::array::from_fn(|c| cof(c, r) / det);
        [
            row[0],
            row[1],
            row[2],
            -(row[0] * m[0][3] + row[1] * m[1][3] + row[2] * m[2][3]),
        ]
    })
}

/// The scan resampled onto `grid`, trilinear, holding the edge values
/// beyond the scan. Laid out with the model's third axis fastest.
pub fn resample(scan: &Scan, grid: &Grid) -> Vec<f32> {
    let [w, h, d] = scan.dims;
    let to_voxel = invert(&scan.affine);
    let [_, ny, nz] = grid.size;
    let mut out = vec![0f32; grid.len()];
    out.par_chunks_mut(ny * nz)
        .enumerate()
        .for_each(|(x, slab)| {
            for y in 0..ny {
                for z in 0..nz {
                    let p = [
                        grid.origin[0] + x as f64 * grid.spacing[0],
                        grid.origin[1] + y as f64 * grid.spacing[1],
                        grid.origin[2] + z as f64 * grid.spacing[2],
                    ];
                    let v = apply(&to_voxel, p);
                    let i = v[0].clamp(0.0, (w - 1) as f64);
                    let j = v[1].clamp(0.0, (h - 1) as f64);
                    let k = v[2].clamp(0.0, (d - 1) as f64);
                    let (i0, j0, k0) = (i.floor() as usize, j.floor() as usize, k.floor() as usize);
                    let (i1, j1, k1) = (
                        (i0 + 1).min(w - 1),
                        (j0 + 1).min(h - 1),
                        (k0 + 1).min(d - 1),
                    );
                    let (fi, fj, fk) = (
                        (i - i0 as f64) as f32,
                        (j - j0 as f64) as f32,
                        (k - k0 as f64) as f32,
                    );
                    let at = |i: usize, j: usize, k: usize| scan.values[i + w * (j + h * k)];
                    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
                    let c00 = lerp(at(i0, j0, k0), at(i1, j0, k0), fi);
                    let c10 = lerp(at(i0, j1, k0), at(i1, j1, k0), fi);
                    let c01 = lerp(at(i0, j0, k1), at(i1, j0, k1), fi);
                    let c11 = lerp(at(i0, j1, k1), at(i1, j1, k1), fi);
                    slab[y * nz + z] = lerp(lerp(c00, c10, fj), lerp(c01, c11, fj), fk);
                }
            }
        });
    out
}

pub fn scale(values: &mut [f32], intensity: Intensity) {
    match intensity {
        Intensity::Window { lo, hi } => {
            values
                .par_iter_mut()
                .for_each(|v| *v = (v.clamp(lo, hi) - lo) / (hi - lo));
        }
        Intensity::NormalizeThenRange => {
            let (n, sum) = values
                .par_iter()
                .filter(|v| **v != 0.0)
                .map(|&v| (1usize, v as f64))
                .reduce(|| (0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
            if n == 0 {
                return;
            }
            let mean = sum / n as f64;
            let var = values
                .par_iter()
                .filter(|v| **v != 0.0)
                .map(|&v| (v as f64 - mean).powi(2))
                .sum::<f64>()
                / n as f64;
            let sd = if var > 0.0 { var.sqrt() } else { 1.0 };
            values.par_iter_mut().for_each(|v| {
                if *v != 0.0 {
                    *v = ((*v as f64 - mean) / sd) as f32;
                }
            });
            let (lo, hi) = values
                .par_iter()
                .fold(
                    || (f32::MAX, f32::MIN),
                    |(lo, hi), &v| (lo.min(v), hi.max(v)),
                )
                .reduce(|| (f32::MAX, f32::MIN), |a, b| (a.0.min(b.0), a.1.max(b.1)));
            if hi > lo {
                values
                    .par_iter_mut()
                    .for_each(|v| *v = (*v - lo) / (hi - lo) * 2.0 - 1.0);
            }
        }
    }
}

/// Where the cubes start along one axis of `size` voxels, as MONAI's
/// sliding window places them.
pub fn starts(size: usize, block: usize, overlap: f64) -> Vec<usize> {
    if size <= block {
        return vec![0];
    }
    let step = ((block as f64 * (1.0 - overlap)) as usize).max(1);
    let count = (size - block).div_ceil(step) + 1;
    (0..count).map(|i| (i * step).min(size - block)).collect()
}

/// A model run on one cube, returning each class's scores in turn.
pub type CubeModel<'a> = dyn FnMut(&[f32]) -> Result<Vec<f32>, String> + 'a;

/// What the cubes add up to, on the model's grid.
pub enum Output {
    Labels(Vec<u16>),
    /// The probability of label 1, for two-class models that carry it back
    /// to the scan smoothly.
    Foreground(Vec<f32>),
}

/// Scores of every class over the whole grid above this many bytes are
/// not kept; only the best few at each voxel are.
pub const AVERAGE_BUDGET: usize = 1 << 30;

/// Runs `model` on cubes covering the grid. Where cubes overlap, their
/// scores are averaged, weighted as the recipe says, as MONAI does. When
/// the scores of all classes would take more than `budget` bytes, only
/// the best few classes at each voxel are kept. `progress(done, total)`
/// follows the cubes.
#[allow(clippy::too_many_arguments)]
pub fn run_blocks(
    values: &[f32],
    grid: &Grid,
    recipe: &Recipe,
    classes: usize,
    model: &mut CubeModel,
    budget: usize,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Output, String> {
    let b = recipe.block;
    let cube_len = b * b * b;
    // A scan smaller than a cube is padded evenly on both sides.
    let padded: [usize; 3] = grid.size.map(|n| n.max(b));
    let before: [usize; 3] = std::array::from_fn(|a| (padded[a] - grid.size[a]) / 2);
    let [nx, ny, nz] = grid.size;
    let len = grid.len();
    let value_at = |x: isize, y: isize, z: isize| -> f32 {
        let inside = |v: isize, n: usize| v >= 0 && (v as usize) < n;
        if inside(x, nx) && inside(y, ny) && inside(z, nz) {
            return values[(x as usize * ny + y as usize) * nz + z as usize];
        }
        match recipe.pad {
            Pad::Constant(c) => c,
            Pad::Edge => {
                let c = |v: isize, n: usize| v.clamp(0, n as isize - 1) as usize;
                values[(c(x, nx) * ny + c(y, ny)) * nz + c(z, nz)]
            }
        }
    };
    // MONAI's Gaussian: centred on the cube's middle, sigma an eighth of
    // the side, and never below a thousandth.
    let sigma = 0.125 * b as f32;
    let centre = (b as f32 - 1.0) / 2.0;
    let gaussian: Vec<f32> = (0..b)
        .map(|i| (-((i as f32 - centre).powi(2)) / (2.0 * sigma * sigma)).exp())
        .collect();
    let weight = |i: usize, j: usize, k: usize| match recipe.blend {
        Blend::Constant => 1.0,
        Blend::Gaussian => (gaussian[i] * gaussian[j] * gaussian[k]).max(1e-3),
    };
    let mut tally = if classes <= SLOTS || classes * len * 4 <= budget {
        Tally::Full {
            sums: vec![0.0; classes * len],
            weight: vec![0.0; len],
        }
    } else {
        Tally::Top {
            kept: vec![EMPTY; SLOTS * len],
            sums: vec![0.0; SLOTS * len],
            floor: vec![0.0; len],
        }
    };

    let (sx, sy, sz) = (
        starts(padded[0], b, recipe.overlap),
        starts(padded[1], b, recipe.overlap),
        starts(padded[2], b, recipe.overlap),
    );
    let total = sx.len() * sy.len() * sz.len();
    let mut done = 0;
    progress(0, total);
    let mut cube = vec![0f32; cube_len];
    for &x0 in &sx {
        for &y0 in &sy {
            for &z0 in &sz {
                if cancel.load(Ordering::Relaxed) {
                    return Err("cancelled".into());
                }
                cube.par_chunks_mut(b * b)
                    .enumerate()
                    .for_each(|(i, slab)| {
                        let x = (x0 + i) as isize - before[0] as isize;
                        for j in 0..b {
                            let y = (y0 + j) as isize - before[1] as isize;
                            for k in 0..b {
                                let z = (z0 + k) as isize - before[2] as isize;
                                slab[j * b + k] = value_at(x, y, z);
                            }
                        }
                    });
                let scores = model(&cube)?;
                if scores.len() != classes * cube_len {
                    return Err("the model returned the wrong number of scores".into());
                }
                // Each cube voxel that lies in the grid: its index in the
                // cube and in the grid, and its weight.
                let mut targets = Vec::with_capacity(cube_len);
                for i in 0..b {
                    let x = (x0 + i) as isize - before[0] as isize;
                    for j in 0..b {
                        let y = (y0 + j) as isize - before[1] as isize;
                        for k in 0..b {
                            let z = (z0 + k) as isize - before[2] as isize;
                            let inside = |v: isize, n: usize| v >= 0 && (v as usize) < n;
                            if inside(x, nx) && inside(y, ny) && inside(z, nz) {
                                let v = (x as usize * ny + y as usize) * nz + z as usize;
                                targets.push(((i * b + j) * b + k, v, weight(i, j, k)));
                            }
                        }
                    }
                }
                tally.add(&scores, classes, cube_len, &targets);
                done += 1;
                progress(done, total);
            }
        }
    }
    Ok(tally.finish(classes, len, recipe.smooth))
}

/// Classes kept at each voxel when there is no room to keep them all.
const SLOTS: usize = 4;
const EMPTY: u16 = u16::MAX;

/// How the cubes' scores add up on the grid.
enum Tally {
    /// Every class's weighted scores, class by class, and the weights.
    Full { sums: Vec<f32>, weight: Vec<f32> },
    /// The few best classes so far at each voxel with their weighted
    /// scores, and a bound on what any other class could have scored: a
    /// class not among a cube's two best scored no more than its second.
    Top {
        kept: Vec<u16>,
        sums: Vec<f32>,
        floor: Vec<f32>,
    },
}

impl Tally {
    fn add(
        &mut self,
        scores: &[f32],
        classes: usize,
        cube_len: usize,
        targets: &[(usize, usize, f32)],
    ) {
        match self {
            Tally::Full { sums, weight } => {
                let len = weight.len();
                sums.par_chunks_mut(len).enumerate().for_each(|(c, sum)| {
                    let s = &scores[c * cube_len..(c + 1) * cube_len];
                    for &(u, v, w) in targets {
                        sum[v] += w * s[u];
                    }
                });
                for &(_, v, w) in targets {
                    weight[v] += w;
                }
            }
            Tally::Top { kept, sums, floor } => {
                // Each cube voxel's two best classes, class by class so the
                // scores are read in order.
                let mut first = vec![(0u16, f32::MIN); cube_len];
                let mut second = vec![(0u16, f32::MIN); cube_len];
                const CHUNK: usize = 4096;
                first
                    .par_chunks_mut(CHUNK)
                    .zip(second.par_chunks_mut(CHUNK))
                    .enumerate()
                    .for_each(|(n, (f, s))| {
                        for c in 0..classes {
                            let row = &scores[c * cube_len + n * CHUNK..];
                            for u in 0..f.len() {
                                let score = row[u];
                                if score > f[u].1 {
                                    s[u] = f[u];
                                    f[u] = (c as u16, score);
                                } else if score > s[u].1 {
                                    s[u] = (c as u16, score);
                                }
                            }
                        }
                    });
                for &(u, v, w) in targets {
                    let slots = v * SLOTS..(v + 1) * SLOTS;
                    let (kept, sums) = (&mut kept[slots.clone()], &mut sums[slots]);
                    for k in 0..SLOTS {
                        if kept[k] != EMPTY {
                            sums[k] += w * scores[kept[k] as usize * cube_len + u];
                        }
                    }
                    for (c, score) in [first[u], second[u]] {
                        if kept.contains(&c) {
                            continue;
                        }
                        // At most what it could have had from earlier cubes.
                        let estimate = floor[v] + w * score;
                        let low = (0..SLOTS)
                            .min_by(|&a, &b| {
                                let key =
                                    |k: usize| if kept[k] == EMPTY { f32::MIN } else { sums[k] };
                                key(a).total_cmp(&key(b))
                            })
                            .unwrap();
                        if kept[low] == EMPTY || sums[low] < estimate {
                            kept[low] = c;
                            sums[low] = estimate;
                        }
                    }
                    floor[v] += w * second[u].1;
                }
            }
        }
    }

    fn finish(self, classes: usize, len: usize, smooth: bool) -> Output {
        match self {
            Tally::Full { sums, weight } => {
                if smooth && classes == 2 {
                    // Softmax of the averaged scores, for label 1.
                    let p = (0..len)
                        .into_par_iter()
                        .map(|v| {
                            let d = (sums[len + v] - sums[v]) / weight[v].max(1e-12);
                            1.0 / (1.0 + (-d).exp())
                        })
                        .collect();
                    return Output::Foreground(p);
                }
                Output::Labels(
                    (0..len)
                        .into_par_iter()
                        .map(|v| argmax(&sums, classes, len, v))
                        .collect(),
                )
            }
            Tally::Top { kept, sums, .. } => Output::Labels(
                (0..len)
                    .into_par_iter()
                    .map(|v| {
                        let mut best = (EMPTY, f32::MIN);
                        for k in v * SLOTS..(v + 1) * SLOTS {
                            let better =
                                sums[k] > best.1 || (sums[k] == best.1 && kept[k] < best.0);
                            if kept[k] != EMPTY && better {
                                best = (kept[k], sums[k]);
                            }
                        }
                        if best.0 == EMPTY { 0 } else { best.0 }
                    })
                    .collect(),
            ),
        }
    }
}

/// The class with the highest score at `v`, where class `c`'s scores start
/// at `c * stride`. Ties go to the lower class.
fn argmax(scores: &[f32], classes: usize, stride: usize, v: usize) -> u16 {
    let mut best = 0;
    for c in 1..classes {
        if scores[c * stride + v] > scores[best * stride + v] {
            best = c;
        }
    }
    best as u16
}

/// The labels at each of the scan's voxels, in the scan's order (first
/// index fastest): the nearest grid voxel's label, or label 1 where the
/// foreground probability, interpolated, is over a half.
pub fn to_scan(output: &Output, grid: &Grid, scan: &Scan) -> Vec<u16> {
    let [w, h, d] = scan.dims;
    let [nx, ny, nz] = grid.size;
    let n = [nx, ny, nz];
    let mut out = vec![0u16; w * h * d];
    out.par_chunks_mut(w * h)
        .enumerate()
        .for_each(|(k, slice)| {
            for j in 0..h {
                for i in 0..w {
                    // The grid covers the scan, give or take half a voxel.
                    let p = apply(&scan.affine, [i as f64, j as f64, k as f64]);
                    let g: [f64; 3] = std::array::from_fn(|a| {
                        ((p[a] - grid.origin[a]) / grid.spacing[a]).clamp(0.0, (n[a] - 1) as f64)
                    });
                    slice[j * w + i] = match output {
                        Output::Labels(labels) => {
                            let [x, y, z] = g.map(|v| v.round() as usize);
                            labels[(x * ny + y) * nz + z]
                        }
                        Output::Foreground(prob) => {
                            let [x0, y0, z0] = g.map(|v| v.floor() as usize);
                            let [x1, y1, z1] = [
                                (x0 + 1).min(nx - 1),
                                (y0 + 1).min(ny - 1),
                                (z0 + 1).min(nz - 1),
                            ];
                            let [fx, fy, fz] =
                                [g[0] - x0 as f64, g[1] - y0 as f64, g[2] - z0 as f64]
                                    .map(|f| f as f32);
                            let at = |x: usize, y: usize, z: usize| prob[(x * ny + y) * nz + z];
                            let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
                            let c00 = lerp(at(x0, y0, z0), at(x0, y0, z1), fz);
                            let c01 = lerp(at(x0, y1, z0), at(x0, y1, z1), fz);
                            let c10 = lerp(at(x1, y0, z0), at(x1, y0, z1), fz);
                            let c11 = lerp(at(x1, y1, z0), at(x1, y1, z1), fz);
                            (lerp(lerp(c00, c01, fy), lerp(c10, c11, fy), fx) > 0.5) as u16
                        }
                    };
                }
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An LPS axial scan as DICOM gives it: x to the patient's left, y to
    /// the back, z up, 0.75 mm pixels and 2 mm slices.
    fn scan(dims: [usize; 3], value: impl Fn(f64, f64, f64) -> f32) -> Scan {
        let affine = [
            [-0.75, 0.0, 0.0, 100.0],
            [0.0, -0.75, 0.0, 80.0],
            [0.0, 0.0, 2.0, -30.0],
        ];
        let [w, h, d] = dims;
        let mut values = Vec::with_capacity(w * h * d);
        for k in 0..d {
            for j in 0..h {
                for i in 0..w {
                    let p = apply(&affine, [i as f64, j as f64, k as f64]);
                    values.push(value(p[0], p[1], p[2]));
                }
            }
        }
        Scan {
            dims,
            affine,
            values,
        }
    }

    #[test]
    fn affines_invert() {
        let m = [
            [-0.7, 0.1, 0.0, 12.0],
            [0.05, -0.8, 0.2, -3.0],
            [0.0, 0.1, 2.5, 40.0],
        ];
        let inv = invert(&m);
        let v = [3.0, -7.0, 11.0];
        let back = apply(&inv, apply(&m, v));
        assert!(
            v.iter().zip(back).all(|(a, b)| (a - b).abs() < 1e-9),
            "{back:?}"
        );
    }

    #[test]
    fn the_grid_runs_right_front_and_up() {
        // A value that rises to the patient's right (RAS +x).
        let s = scan([40, 30, 10], |x, _, _| x as f32);
        let grid = Grid::covering(&s, [1.5, 1.5, 3.0]);
        // 39 * 0.75 = 29.25 mm across, 29 * 0.75 down, 9 * 2 tall: 19.5,
        // 14.5 and 6 voxels, plus one, rounded half to even.
        assert_eq!(grid.size, [20, 16, 7]);
        let v = resample(&s, &grid);
        let (ny, nz) = (grid.size[1], grid.size[2]);
        let at = |x: usize, y: usize, z: usize| v[(x * ny + y) * nz + z];
        assert!((at(0, 3, 2) - grid.origin[0] as f32).abs() < 1e-3);
        assert!((at(10, 3, 2) - (grid.origin[0] + 15.0) as f32).abs() < 1e-3);
    }

    #[test]
    fn labels_come_back_to_the_right_voxels() {
        let s = scan([40, 30, 10], |_, _, _| 0.0);
        let grid = Grid::covering(&s, [1.5, 1.5, 3.0]);
        // Label the grid by position: 1 on the patient's right half.
        let mid = grid.origin[0] + grid.size[0] as f64 * 1.5 / 2.0;
        let mut labels = vec![0u16; grid.len()];
        for x in 0..grid.size[0] {
            if grid.origin[0] + x as f64 * 1.5 > mid {
                for v in &mut labels
                    [x * grid.size[1] * grid.size[2]..(x + 1) * grid.size[1] * grid.size[2]]
                {
                    *v = 1;
                }
            }
        }
        let back = to_scan(&Output::Labels(labels), &grid, &s);
        // RAS x falls as the column rises (DICOM x runs to the left), so
        // the first columns are the patient's right.
        assert_eq!(back[0], 1);
        assert_eq!(back[39], 0);
    }

    #[test]
    fn cubes_cover_the_volume_like_monai() {
        assert_eq!(starts(96, 96, 0.25), [0]);
        assert_eq!(starts(50, 96, 0.25), [0]);
        assert_eq!(starts(187, 96, 0.25), [0, 72, 91]);
        assert_eq!(starts(187, 96, 0.5), [0, 48, 91]);
        assert_eq!(starts(200, 96, 0.5), [0, 48, 96, 104]);
    }

    #[test]
    fn scaling_follows_the_recipes() {
        let mut v = vec![-1000.0, -57.0, 53.5, 164.0, 900.0];
        scale(
            &mut v,
            Intensity::Window {
                lo: -57.0,
                hi: 164.0,
            },
        );
        assert_eq!(v, [0.0, 0.0, 0.5, 1.0, 1.0]);
        let mut v = vec![0.0, 10.0, 20.0, 30.0];
        scale(&mut v, Intensity::NormalizeThenRange);
        // Non-zero values become -1.22, 0, 1.22; zero stays between.
        assert_eq!(v[1], -1.0);
        assert_eq!(v[3], 1.0);
        assert!((v[0] - 0.0).abs() < 1e-6 && (v[2] - 0.0).abs() < 1e-6);
    }

    fn recipe(blend: Blend, smooth: bool) -> Recipe {
        Recipe {
            spacing: [1.0; 3],
            intensity: Intensity::Window { lo: 0.0, hi: 1.0 },
            block: 8,
            overlap: 0.5,
            pad: Pad::Edge,
            blend,
            smooth,
        }
    }

    /// A grid 20 long in x: cubes of 8 start at 0, 4, 8 and 12, centred
    /// at 3.5, 7.5, 11.5 and 15.5.
    fn long_grid() -> Grid {
        Grid {
            size: [20, 8, 8],
            origin: [0.0; 3],
            spacing: [1.0; 3],
        }
    }

    fn run(
        grid: &Grid,
        recipe: &Recipe,
        classes: usize,
        budget: usize,
        score: &dyn Fn(usize) -> Vec<f32>,
    ) -> Output {
        let values = vec![0.0; grid.len()];
        let mut n = 0;
        let mut model = |_: &[f32]| -> Result<Vec<f32>, String> {
            n += 1;
            let per_class = score(n);
            Ok(per_class
                .iter()
                .flat_map(|&s| std::iter::repeat_n(s, 512))
                .collect())
        };
        run_blocks(
            &values,
            grid,
            recipe,
            classes,
            &mut model,
            budget,
            &AtomicBool::new(false),
            &mut |_, _| {},
        )
        .unwrap()
    }

    #[test]
    fn with_or_without_room_the_most_central_cube_wins() {
        // Cube n scores class n highest; Gaussian weights favour the cube
        // in which a voxel is most central. Five classes are more than the
        // slots kept without room.
        let grid = long_grid();
        let score = |n: usize| (0..5).map(|c| (c == n) as u8 as f32).collect();
        for budget in [AVERAGE_BUDGET, 0] {
            let Output::Labels(labels) =
                run(&grid, &recipe(Blend::Gaussian, false), 5, budget, &score)
            else {
                panic!()
            };
            let at = |x: usize| labels[x * 64 + 3 * 8 + 3];
            // x = 5 is 1.5 from the first centre, x = 6 1.5 from the second.
            assert_eq!(
                [at(0), at(5), at(6), at(7), at(11), at(19)],
                [1, 1, 2, 2, 3, 4],
                "budget {budget}"
            );
        }
    }

    #[test]
    fn overlapping_scores_are_averaged() {
        // Two classes; cube n's score for class 1 is -3, +2, -3, +2.
        let grid = long_grid();
        let score = |n: usize| vec![0.0, if n % 2 == 1 { -3.0 } else { 2.0 }];
        let Output::Labels(labels) = run(
            &grid,
            &recipe(Blend::Constant, false),
            2,
            AVERAGE_BUDGET,
            &score,
        ) else {
            panic!()
        };
        let at = |x: usize| labels[x * 64 + 3 * 8 + 3];
        // x 0-3: cube 1 alone, -3. x 4-7: cubes 1 and 2, -0.5. x 8-11: 2
        // and 3, -0.5. x 16-19: cube 4 alone, +2.
        assert_eq!([at(2), at(5), at(9), at(17)], [0, 0, 0, 1]);
        // Smoothly: the probability of label 1 is the softmax of the mean.
        let Output::Foreground(p) = run(
            &grid,
            &recipe(Blend::Constant, true),
            2,
            AVERAGE_BUDGET,
            &score,
        ) else {
            panic!()
        };
        let sigmoid = |d: f32| 1.0 / (1.0 + (-d).exp());
        assert!((p[5 * 64 + 27] - sigmoid(-0.5)).abs() < 1e-6);
        assert!((p[17 * 64 + 27] - sigmoid(2.0)).abs() < 1e-6);
    }

    #[test]
    fn foreground_comes_back_smoothly() {
        // A grid of 2 mm voxels over a scan of 1 mm ones: probability rises
        // along x from 0 to 1; the label starts half way.
        let s = Scan {
            dims: [9, 1, 1],
            affine: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            values: vec![0.0; 9],
        };
        let grid = Grid {
            size: [5, 1, 1],
            origin: [0.0; 3],
            spacing: [2.0, 1.0, 1.0],
        };
        let p = Output::Foreground(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(to_scan(&p, &grid, &s), [0, 0, 0, 0, 0, 1, 1, 1, 1]);
    }
}

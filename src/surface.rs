//! Smooth surfaces for segmentation labels in 3D.
//!
//! Drawn voxel by voxel, labels look like stacked bricks, worst between
//! slices, which often lie several pixels apart. So each voxel also keeps
//! how far it lies inside its own region, measured across its slice.
//! Between voxel centres the 3D view blends those distances, counting
//! each one for its own label and against every other, and a label holds
//! where its blend is positive. Within a slice that rounds off the pixel
//! steps. Between slices it turns one slice's outline into the next
//! (shape-based interpolation), and where a structure ends it closes in a
//! low dome rather than a flat step. The surface's slope, which lights it,
//! comes from the same blend.
//!
//! Counting a voxel's own distance against every other label is exact
//! for the label next to it, but not across slices: above the last slice
//! of a structure, a background voxel may lie near some other structure,
//! and the first would bulge towards it. So where the voxel on the next
//! slice holds another structure, each voxel also keeps how far it lies
//! from that structure on its own slice.
//!
//! A model that works on a coarser grid than the scan hands back its
//! labels in blocks, each of its voxels covering several of the scan's
//! (the built-in whole-body models work at 3 or 1.5 mm, as MONAI's own
//! recipe does). Those blocks would show as terraces, so first
//! `smooth_labels` blurs each structure at the blocks' scale and gives
//! each voxel to the one that covers most of it. Only the 3D view uses
//! the smoothed labels; the slices show the model's own.
//!
//! `smooth_label` is the CPU's copy of the shader's sums, so names and
//! hovering agree with what the 3D view draws.

use rayon::prelude::*;

use crate::monai::Segmentation;

/// Distances are kept in eighths of a pixel...
pub const STEPS: f32 = 8.0;
/// ...up to 24 pixels; anything further counts as that far. The volume
/// shader holds the same number.
pub const FAR: u8 = 192;

/// What each voxel keeps, all across its own slice in `STEPS` per pixel
/// up to `FAR`: how far it lies inside its own region (a structure, or
/// the background), then how far from the structure in the voxel above
/// it, and below it, where that is another structure (else 0). Beyond the
/// image counts as background. Slices run in parallel.
pub fn distances(voxels: &[u16], dims: [usize; 3]) -> Vec<[u8; 3]> {
    let [w, h, d] = dims;
    let plane = w * h;
    let mut out = vec![[FAR, 0, 0]; voxels.len()];
    if plane == 0 {
        return out;
    }
    let slice = |k: usize| &voxels[k * plane..(k + 1) * plane];
    out.par_chunks_mut(plane).enumerate().for_each(|(k, out)| {
        let mut s = Scratch::default();
        own_distances(slice(k), w, h, out, &mut s);
        if k + 1 < d {
            partner_distances(slice(k), slice(k + 1), w, h, out, 1, &mut s);
        }
        if k > 0 {
            partner_distances(slice(k), slice(k - 1), w, h, out, 2, &mut s);
        }
    });
    out
}

/// Squared distances beyond this are not worth knowing: `FAR` and a
/// little more.
const BIG: f32 = (FAR as f32 / STEPS + 2.0) * (FAR as f32 / STEPS + 2.0);

/// A distance between pixel centres as kept: a pixel of another label is
/// one pixel away, and the edge lies halfway.
fn steps(squared: f32) -> u8 {
    ((squared.sqrt() - 0.5) * STEPS)
        .round()
        .clamp(0.0, FAR as f32) as u8
}

/// Each label's bounding box on a slice, as (x0, y0, x1, y1), among the
/// pixels `counts` picks; `None` for labels not there.
fn boxes(labels: &[u16], w: usize, counts: impl Fn(usize) -> bool) -> Vec<Option<[usize; 4]>> {
    let top = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut boxes: Vec<Option<[usize; 4]>> = vec![None; top + 1];
    for (i, &v) in labels.iter().enumerate() {
        if !counts(i) {
            continue;
        }
        let (x, y) = (i % w, i / w);
        let b = boxes[v as usize].get_or_insert([x, y, x, y]);
        *b = [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)];
    }
    boxes
}

/// The squared distance transform over box (x0, y0, x1, y1), which may
/// reach past the slice: in `grid`, row by row, to the nearest place
/// where `feature` holds. Returns the grid's width.
fn box_transform(
    b: [isize; 4],
    feature: impl Fn(isize, isize) -> bool,
    grid: &mut Vec<f32>,
    s: &mut Scratch,
) -> usize {
    let (gw, gh) = ((b[2] - b[0] + 1) as usize, (b[3] - b[1] + 1) as usize);
    grid.clear();
    grid.extend((0..gw * gh).map(|i| {
        let (x, y) = (b[0] + (i % gw) as isize, b[1] + (i / gw) as isize);
        if feature(x, y) { 0.0 } else { BIG }
    }));
    distance_transform(grid, gw, gh, s);
    gw
}

/// How far each pixel lies inside its own region. For each label, the
/// transform of everything else runs over the label's bounding box and
/// one pixel round it: anything outside the box is another label, and
/// the ring is at least as near as anything beyond it.
fn own_distances(labels: &[u16], w: usize, h: usize, out: &mut [[u8; 3]], s: &mut Scratch) {
    let at = |x: isize, y: isize| {
        if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
            0
        } else {
            labels[y as usize * w + x as usize]
        }
    };
    let mut grid = Vec::new();
    for (label, b) in boxes(labels, w, |_| true).into_iter().enumerate() {
        let Some(b) = b else { continue };
        let label = label as u16;
        let ring = [
            b[0] as isize - 1,
            b[1] as isize - 1,
            b[2] as isize + 1,
            b[3] as isize + 1,
        ];
        let gw = box_transform(ring, |x, y| at(x, y) != label, &mut grid, s);
        for y in b[1]..=b[3] {
            for x in b[0]..=b[2] {
                if labels[y * w + x] == label {
                    out[y * w + x][0] = steps(grid[(y + 1 - b[1]) * gw + x + 1 - b[0]]);
                }
            }
        }
    }
}

/// Where the voxel on the next slice, above or below, holds a structure
/// this one does not, how far this one lies from that structure on its
/// own slice: into `slot` of `out`. A structure further than `FAR` from
/// its box cannot matter.
fn partner_distances(
    this: &[u16],
    next: &[u16],
    w: usize,
    h: usize,
    out: &mut [[u8; 3]],
    slot: usize,
    s: &mut Scratch,
) {
    let wanted = |i: usize| next[i] != 0 && next[i] != this[i];
    let reach = (FAR as f32 / STEPS) as usize + 1;
    let mut grid = Vec::new();
    for (label, b) in boxes(next, w, wanted).into_iter().enumerate() {
        let Some(b) = b else { continue };
        let label = label as u16;
        let wide = [
            b[0].saturating_sub(reach),
            b[1].saturating_sub(reach),
            (b[2] + reach).min(w - 1),
            (b[3] + reach).min(h - 1),
        ];
        let gw = box_transform(
            wide.map(|v| v as isize),
            |x, y| this[y as usize * w + x as usize] == label,
            &mut grid,
            s,
        );
        for y in b[1]..=b[3] {
            for x in b[0]..=b[2] {
                let i = y * w + x;
                if wanted(i) && next[i] == label {
                    out[i][slot] = steps(grid[(y - wide[1]) * gw + x - wide[0]]);
                }
            }
        }
    }
}

/// How much `smooth_labels` blurs, in blocks: enough to take the steps
/// out of a diagonal edge, not so much that a structure one block across
/// disappears.
const SOFTEN: f32 = 0.4;

/// The blur along each axis, in scan voxels, for labels made in blocks of
/// `blocks` scan voxels: none for blocks under one and a half.
pub fn blur_for(blocks: [f32; 3]) -> [f32; 3] {
    blocks.map(|b| if b >= 1.5 { SOFTEN * b } else { 0.0 })
}

/// Gaussian weights out to three sigma, summing to 1, centre in the
/// middle; a lone 1 for no blur.
fn kernel(sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        return vec![1.0];
    }
    let r = (3.0 * sigma).ceil() as isize;
    let k: Vec<f32> = (-r..=r)
        .map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let total: f32 = k.iter().sum();
    k.into_iter().map(|v| v / total).collect()
}

/// A box of voxels: x, y and slice ranges.
#[derive(Clone)]
struct Region([std::ops::Range<usize>; 3]);

impl Region {
    fn len(&self) -> usize {
        self.0.iter().map(|r| r.len()).product()
    }
}

/// The labels blurred by `sigma` voxels along each axis, structure by
/// structure, each voxel going to the structure that covers most of it
/// once blurred, or to the background where they all together cover no
/// more than half. Beyond the volume its edge repeats. Works through the
/// volume eight slices at a time, in parallel.
pub fn smooth_labels(voxels: &[u16], dims: [usize; 3], sigma: [f32; 3]) -> Vec<u16> {
    let [w, h, d] = dims;
    let kernels = sigma.map(kernel);
    let reach = kernels.clone().map(|k| k.len() / 2);
    const CHUNK: usize = 8;
    let mut out = vec![0u16; voxels.len()];
    if w * h == 0 {
        return out;
    }
    out.par_chunks_mut(CHUNK * w * h)
        .enumerate()
        .for_each(|(c, out)| {
            let (z0, z1) = (c * CHUNK, (c * CHUNK + CHUNK).min(d));
            let chunk = Region([0..w, 0..h, z0..z1]);
            let any = blur_region(voxels, dims, &kernels, &chunk, |v| v != 0);
            // Each structure within reach, and the box it fills.
            let slab = z0.saturating_sub(reach[2])..(z1 + reach[2]).min(d);
            let mut boxes: std::collections::BTreeMap<u16, [usize; 6]> = Default::default();
            for z in slab {
                for (i, &v) in voxels[z * w * h..(z + 1) * w * h].iter().enumerate() {
                    if v != 0 {
                        let (x, y) = (i % w, i / w);
                        let b = boxes.entry(v).or_insert([x, y, z, x, y, z]);
                        *b = [
                            b[0].min(x),
                            b[1].min(y),
                            b[2].min(z),
                            b[3].max(x),
                            b[4].max(y),
                            b[5].max(z),
                        ];
                    }
                }
            }
            let mut best = vec![0f32; out.len()];
            for (label, b) in boxes {
                let near = |a: usize, n: usize, lo: usize, hi: usize| {
                    b[a].saturating_sub(reach[a]).max(lo)..(b[a + 3] + 1 + reach[a]).min(n).min(hi)
                };
                let near = Region([near(0, w, 0, w), near(1, h, 0, h), near(2, d, z0, z1)]);
                if near.len() == 0 {
                    continue;
                }
                let share = blur_region(voxels, dims, &kernels, &near, |v| v == label);
                let [xs, ys, zs] = &near.0;
                let mut i = 0;
                for z in zs.clone() {
                    for y in ys.clone() {
                        let row = ((z - z0) * h + y) * w;
                        for x in xs.clone() {
                            if share[i] > best[row + x] {
                                best[row + x] = share[i];
                                out[row + x] = label;
                            }
                            i += 1;
                        }
                    }
                }
            }
            for i in 0..out.len() {
                if best[i] <= 1.0 - any[i] {
                    out[i] = 0;
                }
            }
        });
    out
}

/// A mask of the voxels whose labels `pick` picks, blurred with
/// `kernels` along each axis, over `region`: x, then y, then slices,
/// each pass reading as far past the region as its kernel reaches.
fn blur_region(
    voxels: &[u16],
    dims: [usize; 3],
    kernels: &[Vec<f32>; 3],
    region: &Region,
    pick: impl Fn(u16) -> bool,
) -> Vec<f32> {
    let [w, h, d] = dims;
    let r = kernels.clone().map(|k| (k.len() / 2) as isize);
    let [xs, ys, zs] = &region.0;
    let at = |i: isize, n: usize| i.clamp(0, n as isize - 1) as usize;
    // The rows and slices the later passes read.
    let wide_y: Vec<usize> = (ys.start as isize - r[1]..ys.end as isize + r[1])
        .map(|y| at(y, h))
        .collect();
    let wide_z: Vec<usize> = (zs.start as isize - r[2]..zs.end as isize + r[2])
        .map(|z| at(z, d))
        .collect();
    let nx = xs.len();
    let mut a = Vec::with_capacity(wide_z.len() * wide_y.len() * nx);
    for &z in &wide_z {
        for &y in &wide_y {
            let row = &voxels[(z * h + y) * w..(z * h + y + 1) * w];
            for x in xs.clone() {
                let mut sum = 0.0;
                for (i, k) in kernels[0].iter().enumerate() {
                    if pick(row[at(x as isize + i as isize - r[0], w)]) {
                        sum += k;
                    }
                }
                a.push(sum);
            }
        }
    }
    let ny = ys.len();
    let mut b = Vec::with_capacity(wide_z.len() * ny * nx);
    for z in 0..wide_z.len() {
        for y in 0..ny {
            for x in 0..nx {
                let mut sum = 0.0;
                for (j, k) in kernels[1].iter().enumerate() {
                    sum += k * a[(z * wide_y.len() + y + j) * nx + x];
                }
                b.push(sum);
            }
        }
    }
    let mut c = Vec::with_capacity(region.len());
    for z in 0..zs.len() {
        for y in 0..ny {
            for x in 0..nx {
                let mut sum = 0.0;
                for (l, k) in kernels[2].iter().enumerate() {
                    sum += k * b[((z + l) * ny + y) * nx + x];
                }
                c.push(sum);
            }
        }
    }
    c
}

#[derive(Default)]
struct Scratch {
    line: Vec<f32>,
    out: Vec<f32>,
    v: Vec<usize>,
    z: Vec<f32>,
}

/// Squared Euclidean distance from each cell to the nearest cell holding
/// 0, in place, columns then rows. Cells start at 0 or `BIG`.
fn distance_transform(grid: &mut [f32], w: usize, h: usize, s: &mut Scratch) {
    for x in 0..w {
        s.line.clear();
        s.line.extend((0..h).map(|y| grid[y * w + x]));
        squared_distances(s);
        for y in 0..h {
            grid[y * w + x] = s.out[y];
        }
    }
    for y in 0..h {
        s.line.clear();
        s.line.extend_from_slice(&grid[y * w..(y + 1) * w]);
        squared_distances(s);
        grid[y * w..(y + 1) * w].copy_from_slice(&s.out);
    }
}

/// The one-dimensional step: for each position q, the least of
/// (q - p)² + line[p] over all p, from the lower envelope of those
/// parabolas (Felzenszwalb and Huttenlocher, 2012).
fn squared_distances(s: &mut Scratch) {
    let f = &s.line;
    let n = f.len();
    s.out.resize(n, 0.0);
    s.v.resize(n, 0);
    s.z.resize(n + 1, 0.0);
    let (v, z) = (&mut s.v, &mut s.z);
    let mut k = 0;
    v[0] = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;
    for q in 1..n {
        let fq = f[q] + (q * q) as f32;
        let mut cross;
        loop {
            let p = v[k];
            cross = (fq - (f[p] + (p * p) as f32)) / (2 * (q - p)) as f32;
            if cross > z[k] {
                break;
            }
            // z[0] is minus infinity, so k never passes 0.
            k -= 1;
        }
        k += 1;
        v[k] = q;
        z[k] = cross;
        z[k + 1] = f32::INFINITY;
    }
    k = 0;
    for q in 0..n {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let p = v[k];
        let d = q as f32 - p as f32;
        s.out[q] = d * d + f[p];
    }
}

/// The label whose smooth surface holds point `g` of the label grid (x,
/// y and slice, in voxels, with voxel centres at whole numbers), and how
/// far inside it the point lies, in `STEPS` per pixel; or 0 where no
/// label holds. Each of the eight voxels round the point counts its
/// distance for its own label and against every other, weighted as in
/// trilinear interpolation; the label with the largest sum wins if that
/// sum is positive. `smooth_label` in `volume.wgsl` does the same.
pub fn smooth_label(seg: &Segmentation, g: [f32; 3]) -> (u16, f32) {
    let corners = corners(seg, g);
    let mut best = (0u16, f32::MIN);
    for i in 0..8 {
        let a = corners[i].0;
        if a == 0 || corners[..i].iter().any(|c| c.0 == a) {
            continue;
        }
        let inside = inside_label(&corners, a);
        if inside > best.1 {
            best = (a, inside);
        }
    }
    if best.1 > 0.0 { best } else { (0, best.1) }
}

/// The eight voxels round grid point `g`, lower slice first: label,
/// distances and trilinear weight.
fn corners(seg: &Segmentation, g: [f32; 3]) -> [(u16, [u8; 3], f32); 8] {
    let b = g.map(f32::floor);
    let f = [g[0] - b[0], g[1] - b[1], g[2] - b[2]];
    std::array::from_fn(|i| {
        let o = [i & 1, (i >> 1) & 1, (i >> 2) & 1];
        let q: [i64; 3] = std::array::from_fn(|a| b[a] as i64 + o[a] as i64);
        let (id, d) = voxel(seg, q);
        let w = (0..3)
            .map(|a| if o[a] == 1 { f[a] } else { 1.0 - f[a] })
            .product();
        (id, d, w)
    })
}

/// How far inside label `a` a point lies, from its eight voxels: for
/// their own label, their distance inside it; for the label of the voxel
/// across the slab from them, their distance from it; for any other,
/// their distance inside their own region, against.
fn inside_label(corners: &[(u16, [u8; 3], f32); 8], a: u16) -> f32 {
    let mut inside = 0.0;
    for (j, &(id, d, w)) in corners.iter().enumerate() {
        let across = corners[j ^ 4].0;
        inside += w * if id == a {
            d[0] as f32
        } else if across == a {
            // Corners 0 to 3 are on the lower slice: their partner is above.
            -(d[if j < 4 { 1 } else { 2 }] as f32)
        } else {
            -(d[0] as f32)
        };
    }
    inside
}

/// A voxel's label and distances; beyond the volume, background and far.
fn voxel(seg: &Segmentation, q: [i64; 3]) -> (u16, [u8; 3]) {
    let size = [seg.width, seg.height, seg.depth].map(|s| s as i64);
    if (0..3).any(|a| q[a] < 0 || q[a] >= size[a]) {
        return (0, [FAR, 0, 0]);
    }
    let i = ((q[2] * size[1] + q[1]) * size[0] + q[0]) as usize;
    (seg.surface_labels()[i], seg.distances[i])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distances by brute force: from each pixel to the nearest one of
    /// another label, with the image framed in background, and to the
    /// structure in the pixel above and below where that differs.
    fn brute(voxels: &[u16], w: usize, h: usize, d: usize) -> Vec<[u8; 3]> {
        let at = |x: isize, y: isize, k: usize| {
            if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
                0
            } else {
                voxels[(k * h + y as usize) * w + x as usize]
            }
        };
        // The least squared distance from (x, y) on slice k to a pixel,
        // framing included, where `hit` holds.
        let nearest = |x: isize, y: isize, k: usize, hit: &dyn Fn(u16) -> bool| {
            let mut best = BIG;
            for v in -1..=h as isize {
                for u in -1..=w as isize {
                    if hit(at(u, v, k)) {
                        best = best.min(((u - x).pow(2) + (v - y).pow(2)) as f32);
                    }
                }
            }
            steps(best)
        };
        let mut out = vec![[0; 3]; w * h * d];
        for k in 0..d {
            for y in 0..h as isize {
                for x in 0..w as isize {
                    let own = at(x, y, k);
                    let o = &mut out[(k * h + y as usize) * w + x as usize];
                    o[0] = nearest(x, y, k, &|l| l != own);
                    for (slot, next) in [(1, k + 1), (2, k.wrapping_sub(1))] {
                        if next < d {
                            let a = at(x, y, next);
                            if a != 0 && a != own {
                                o[slot] = nearest(x, y, k, &|l| l == a);
                            }
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn distances_match_brute_force() {
        // Blobs of three labels, odd sizes, touching each other and the
        // edges, on three slices, from a fixed pseudo-random sequence.
        let mut seed = 7u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed >> 8
        };
        for (w, h) in [(37, 23), (60, 61), (5, 70)] {
            let d = 3;
            let mut labels = vec![0u16; w * h * d];
            for k in 0..d {
                for _ in 0..6 {
                    let (cx, cy) = ((next() % w as u32) as f32, (next() % h as u32) as f32);
                    let r = 2.0 + (next() % 12) as f32;
                    let l = 1 + (next() % 3) as u16;
                    for y in 0..h {
                        for x in 0..w {
                            if (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2) < r * r {
                                labels[(k * h + y) * w + x] = l;
                            }
                        }
                    }
                }
            }
            let got = distances(&labels, [w, h, d]);
            assert!(got.iter().any(|v| v[1] > 0) && got.iter().any(|v| v[2] > 0));
            assert_eq!(got, brute(&labels, w, h, d), "{w} x {h}");
        }
        // An empty slice is far from everything.
        assert!(
            distances(&[0; 12], [4, 3, 1])
                .iter()
                .all(|&d| d == [FAR, 0, 0])
        );
    }

    /// Smoothing takes the steps out of labels made in blocks four voxels
    /// across, keeps straight edges and lone blocks, and leaves labels
    /// made at the scan's scale alone.
    #[test]
    fn smoothing_rounds_off_blocks() {
        let (w, h) = (64usize, 64usize);
        let sigma = blur_for([4.0, 4.0, 1.0]);
        assert_eq!(sigma, [1.6, 1.6, 0.0]);
        assert_eq!(blur_for([1.2, 1.0, 1.4]), [0.0; 3]);
        // A diagonal edge in four-pixel blocks: on rows 4b to 4b + 3 the
        // edge is at 35.5 - 4b, and the line through the middle of the
        // steps is x = 37 - y.
        let blocks: Vec<u16> = (0..w * h)
            .map(|i| ((i % w) / 4 + (i / w) / 4 < 9) as u16)
            .collect();
        let smooth = smooth_labels(&blocks, [w, h, 1], sigma);
        let edge = |labels: &[u16], y: usize| {
            (0..w).position(|x| labels[y * w + x] == 0).unwrap() as f32 - 0.5
        };
        let stray = |labels: &[u16]| {
            (8..28)
                .map(|y| (edge(labels, y) - (37.0 - y as f32)).abs())
                .fold(0.0, f32::max)
        };
        assert_eq!(stray(&blocks), 1.5);
        assert!(stray(&smooth) <= 0.5, "smoothed {}", stray(&smooth));
        // A straight edge stays put, and a lone block stays.
        let mut v: Vec<u16> = (0..w * h).map(|i| (i % w < 30) as u16 * 2).collect();
        for y in 40..44 {
            for x in 40..44 {
                v[y * w + x] = 3;
            }
        }
        let smooth = smooth_labels(&v, [w, h, 1], sigma);
        for y in 0..h {
            assert_eq!(edge(&smooth, y), 29.5, "row {y}");
        }
        assert!(smooth.contains(&3));
        // Through slices too, with labels on every other one.
        let (n, d) = (16usize, 12usize);
        let layers: Vec<u16> = (0..n * n * d)
            .map(|i| ((i / (n * n)) / 4 % 2) as u16)
            .collect();
        let smooth = smooth_labels(&layers, [n, n, d], [0.0, 0.0, 1.6]);
        assert_eq!(smooth, layers, "straight edges between slices stay too");
    }

    fn seg(dims: [usize; 3], voxels: Vec<u16>) -> Segmentation {
        Segmentation::new(String::new(), String::new(), dims, voxels, Vec::new())
    }

    /// Inside a structure and well clear of it the answer is plain;
    /// voxel centres keep their own label.
    #[test]
    fn smooth_labels_keep_voxel_centres() {
        let (w, h, d) = (20usize, 20usize, 3usize);
        let mut v = vec![0u16; w * h * d];
        for k in 0..d {
            for y in 5..15 {
                for x in 5..15 {
                    v[(k * h + y) * w + x] = if x < 10 { 2 } else { 4 };
                }
            }
        }
        let s = seg([w, h, d], v.clone());
        for k in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let (id, _) = smooth_label(&s, [x as f32, y as f32, k as f32]);
                    assert_eq!(id, v[(k * h + y) * w + x], "({x}, {y}, {k})");
                }
            }
        }
        // Edges lie halfway between voxels, also between two labels.
        assert_eq!(smooth_label(&s, [4.6, 10.0, 1.0]).0, 2);
        assert_eq!(smooth_label(&s, [4.4, 10.0, 1.0]).0, 0);
        assert_eq!(smooth_label(&s, [9.4, 10.0, 1.0]).0, 2);
        assert_eq!(smooth_label(&s, [9.6, 10.0, 1.0]).0, 4);
    }

    /// Between slices an edge moves steadily from one slice's place to the
    /// next's, rather than jumping halfway.
    #[test]
    fn edges_move_smoothly_between_slices() {
        // Label 1 left of x = 10 on slice 0, left of x = 14 on slice 1.
        let (w, h) = (32usize, 32usize);
        let mut v = vec![0u16; w * h * 2];
        for y in 0..h {
            for x in 0..w {
                v[y * w + x] = (x < 10) as u16;
                v[(h + y) * w + x] = (x < 14) as u16;
            }
        }
        let s = seg([w, h, 2], v);
        // The edge is at 9.5 on slice 0 and 13.5 on slice 1, so at
        // 9.5 + 4z between them.
        for z in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let edge = 9.5 + 4.0 * z;
            assert_eq!(smooth_label(&s, [edge - 0.1, 16.0, z]).0, 1, "z {z}");
            assert_eq!(smooth_label(&s, [edge + 0.1, 16.0, z]).0, 0, "z {z}");
        }
    }

    /// A structure on one slice closes in a dome: half a slice out at its
    /// middle, less towards its edges, nothing beyond them.
    #[test]
    fn ends_close_in_a_dome() {
        let (w, h) = (61usize, 61usize);
        let mut v = vec![0u16; w * h * 3];
        for y in 10..51 {
            for x in 10..51 {
                v[(h + y) * w + x] = 1;
            }
        }
        let s = seg([w, h, 3], v);
        let inside = |x: usize, y: usize| s.distances[(h + y) * w + x][0] as f32;
        // The middle lies about 20 pixels in; above it the empty slice is
        // as far as can be, 24, so the surface is about 20 / 44 of the
        // way up.
        let c = inside(30, 30);
        assert!(c > 19.0 * STEPS && c < 20.6 * STEPS, "{c}");
        let top = 1.0 + c / (c + FAR as f32);
        assert_eq!(smooth_label(&s, [30.0, 30.0, top - 0.02]).0, 1);
        assert_eq!(smooth_label(&s, [30.0, 30.0, top + 0.02]).0, 0);
        assert_eq!(smooth_label(&s, [30.0, 30.0, 2.0 - top - 0.02]).0, 0);
        assert_eq!(smooth_label(&s, [30.0, 30.0, 2.0 - top + 0.02]).0, 1);
        // 1.5 pixels in from the edge, 1.5 / 25.5 of the way.
        assert_eq!(inside(11, 30), 1.5 * STEPS);
        assert_eq!(smooth_label(&s, [11.0, 30.0, 1.04]).0, 1);
        assert_eq!(smooth_label(&s, [11.0, 30.0, 1.08]).0, 0);
    }
}

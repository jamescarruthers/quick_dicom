//! NIfTI, the volume format MONAI Label reads and writes: a fixed header,
//! then the voxels. Only what the MONAI Label client needs: writing a scan
//! with its place in the patient, and reading back a label volume.

use std::io::Read;

use crate::decode::Plane;

/// A volume's place in the patient: the position, in millimetres, of voxel
/// (i, j, k) is `affine · [i, j, k, 1]`. NIfTI's axes are RAS+: x grows to
/// the patient's right, y to the front, z to the head. DICOM's are LPS+, so
/// its x and y change sign on the way.
pub type Affine = [[f64; 4]; 3];

/// The affine of a stack of slices, from its first and last frames. The
/// slice direction comes from their positions, so slices may run either
/// way; without a second position it is the planes' normal, one slice
/// thickness `gap` apart.
pub fn affine(first: &Plane, last: Option<&Plane>, count: usize, gap: f64) -> Affine {
    let (dy, dx) = first.spacing;
    let normal = cross(first.row, first.column);
    let step = match last {
        Some(l) if count > 1 => {
            let d = sub(l.position, first.position).map(|v| v / (count - 1) as f64);
            if norm(d) > 1e-6 {
                d
            } else {
                normal.map(|v| v * gap)
            }
        }
        _ => normal.map(|v| v * gap),
    };
    let i = first.row.map(|v| v * dx);
    let j = first.column.map(|v| v * dy);
    // DICOM is LPS+, NIfTI RAS+: x and y change sign.
    let flip = [-1.0, -1.0, 1.0];
    std::array::from_fn(|r| {
        [
            flip[r] * i[r],
            flip[r] * j[r],
            flip[r] * step[r],
            flip[r] * first.position[r],
        ]
    })
}

/// How the voxels are stored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    I16,
    F32,
}

impl Kind {
    fn code(self) -> (i16, i16) {
        match self {
            Kind::I16 => (4, 16),
            Kind::F32 => (16, 32),
        }
    }
}

/// Bytes of a NIfTI-1 header (with its four-byte extension flag) for a
/// volume of `dims` voxels, x fastest. The voxels follow it, little-endian.
pub fn header(dims: [usize; 3], affine: &Affine, kind: Kind) -> Vec<u8> {
    let mut h = vec![0u8; 352];
    let put_i16 =
        |h: &mut Vec<u8>, at: usize, v: i16| h[at..at + 2].copy_from_slice(&v.to_le_bytes());
    let put_f32 =
        |h: &mut Vec<u8>, at: usize, v: f32| h[at..at + 4].copy_from_slice(&v.to_le_bytes());
    h[0..4].copy_from_slice(&348i32.to_le_bytes());
    let dim = [
        3,
        dims[0] as i16,
        dims[1] as i16,
        dims[2] as i16,
        1,
        1,
        1,
        1,
    ];
    for (k, d) in dim.into_iter().enumerate() {
        put_i16(&mut h, 40 + 2 * k, d);
    }
    let (datatype, bitpix) = kind.code();
    put_i16(&mut h, 70, datatype);
    put_i16(&mut h, 72, bitpix);

    // Split the affine into voxel sizes, a rotation and an offset, for the
    // quaternion form; the matrix form below holds it exactly.
    let column = |c: usize| [affine[0][c], affine[1][c], affine[2][c]];
    let sizes = [norm(column(0)), norm(column(1)), norm(column(2))];
    let mut r: [[f64; 3]; 3] =
        std::array::from_fn(|row| std::array::from_fn(|c| affine[row][c] / sizes[c].max(1e-12)));
    let qfac = if det(&r) < 0.0 {
        (0..3).for_each(|row| r[row][2] = -r[row][2]);
        -1.0
    } else {
        1.0
    };
    let [_, b, c, d] = quaternion(&r);
    put_f32(&mut h, 76, qfac);
    for (k, s) in sizes.into_iter().enumerate() {
        put_f32(&mut h, 80 + 4 * k, s as f32);
    }
    put_f32(&mut h, 108, 352.0); // vox_offset
    put_f32(&mut h, 112, 1.0); // scl_slope
    h[123] = 2; // millimetres
    put_i16(&mut h, 252, 1); // qform: scanner coordinates
    put_i16(&mut h, 254, 1); // sform: scanner coordinates
    put_f32(&mut h, 256, b as f32);
    put_f32(&mut h, 260, c as f32);
    put_f32(&mut h, 264, d as f32);
    for (k, row) in affine.iter().enumerate() {
        put_f32(&mut h, 268 + 4 * k, row[3] as f32);
        for (c, v) in row.iter().enumerate() {
            put_f32(&mut h, 280 + 16 * k + 4 * c, *v as f32);
        }
    }
    h[344..348].copy_from_slice(b"n+1\0");
    h
}

/// A label volume read back from NIfTI.
#[derive(Debug, PartialEq)]
pub struct Labels {
    pub dims: [usize; 3],
    /// One label per voxel, x fastest.
    pub voxels: Vec<u16>,
}

/// Reads a NIfTI-1 or NIfTI-2 label volume, gzipped or not, in either
/// byte order. Values become whole labels from 0 to 65535.
pub fn read_labels(bytes: &[u8]) -> Result<Labels, String> {
    let unzipped;
    let bytes = if bytes.starts_with(&[0x1f, 0x8b]) {
        let mut out = Vec::new();
        flate2::read::MultiGzDecoder::new(bytes)
            .read_to_end(&mut out)
            .map_err(|e| format!("the labels are not valid gzip: {e}"))?;
        unzipped = out;
        &unzipped[..]
    } else {
        bytes
    };
    if bytes.len() < 348 {
        return Err("the labels are not a NIfTI file".into());
    }
    let size_le = i32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let size_be = i32::from_be_bytes(bytes[0..4].try_into().unwrap());
    let (version, le) = match (size_le, size_be) {
        (348, _) => (1, true),
        (_, 348) => (1, false),
        (540, _) => (2, true),
        (_, 540) => (2, false),
        _ => return Err("the labels are not a NIfTI file".into()),
    };
    let field = |at: usize, n: usize| -> &[u8] { &bytes[at..at + n] };
    let i16_at = |at| {
        let b: [u8; 2] = field(at, 2).try_into().unwrap();
        if le {
            i16::from_le_bytes(b)
        } else {
            i16::from_be_bytes(b)
        }
    };
    let i64_at = |at| {
        let b: [u8; 8] = field(at, 8).try_into().unwrap();
        if le {
            i64::from_le_bytes(b)
        } else {
            i64::from_be_bytes(b)
        }
    };
    let f32_at = |at| {
        let b: [u8; 4] = field(at, 4).try_into().unwrap();
        if le {
            f32::from_le_bytes(b)
        } else {
            f32::from_be_bytes(b)
        }
    };
    let f64_at = |at| {
        let b: [u8; 8] = field(at, 8).try_into().unwrap();
        if le {
            f64::from_le_bytes(b)
        } else {
            f64::from_be_bytes(b)
        }
    };
    let (dim, datatype, offset, slope, intercept): ([i64; 8], i16, usize, f64, f64) =
        if version == 1 {
            (
                std::array::from_fn(|k| i16_at(40 + 2 * k) as i64),
                i16_at(70),
                f32_at(108) as usize,
                f32_at(112) as f64,
                f32_at(116) as f64,
            )
        } else {
            if bytes.len() < 540 {
                return Err("the labels' NIfTI header is cut short".into());
            }
            (
                std::array::from_fn(|k| i64_at(16 + 8 * k)),
                i16_at(12),
                i64_at(168) as usize,
                f64_at(176),
                f64_at(184),
            )
        };
    // Up to three dimensions, plus any of size one beyond them.
    let rank = dim[0].clamp(1, 7) as usize;
    if dim[1..=rank].iter().any(|&d| d < 1) || dim[4..=rank.max(3)].iter().any(|&d| d > 1) {
        return Err(format!(
            "the labels have an unexpected shape {:?}",
            &dim[1..=rank]
        ));
    }
    let dims = [1, 2, 3].map(|k| if k <= rank { dim[k] as usize } else { 1 });
    let count = dims[0] * dims[1] * dims[2];
    let width = match datatype {
        2 | 256 => 1,
        4 | 512 => 2,
        8 | 16 | 768 => 4,
        64 | 1024 | 1280 => 8,
        _ => {
            return Err(format!(
                "labels stored as NIfTI type {datatype} are not supported"
            ));
        }
    };
    let data = bytes
        .get(offset..offset + count * width)
        .ok_or("the labels file is cut short")?;
    let scale = |v: f64| {
        if slope != 0.0 && slope.is_finite() {
            v * slope + intercept
        } else {
            v
        }
    };
    let word = |c: &[u8]| -> [u8; 8] {
        let mut b = [0u8; 8];
        b[..c.len()].copy_from_slice(c);
        if !le {
            b[..c.len()].reverse();
        }
        b
    };
    let voxels = data
        .chunks_exact(width)
        .map(|c| {
            let b = word(c);
            let v = match datatype {
                2 => b[0] as f64,
                256 => b[0] as i8 as f64,
                4 => i16::from_le_bytes([b[0], b[1]]) as f64,
                512 => u16::from_le_bytes([b[0], b[1]]) as f64,
                8 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
                768 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
                16 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
                64 => f64::from_le_bytes(b),
                1024 => i64::from_le_bytes(b) as f64,
                _ => u64::from_le_bytes(b) as f64,
            };
            scale(v).round().clamp(0.0, 65535.0) as u16
        })
        .collect();
    Ok(Labels { dims, voxels })
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn det(r: &[[f64; 3]; 3]) -> f64 {
    r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
}

/// Unit quaternion (a, b, c, d), a ≥ 0, of a rotation matrix, as
/// `nifti_mat44_to_quatern` computes it.
fn quaternion(r: &[[f64; 3]; 3]) -> [f64; 4] {
    let trace = r[0][0] + r[1][1] + r[2][2] + 1.0;
    let q = if trace > 0.5 {
        let a = 0.5 * trace.sqrt();
        [
            a,
            0.25 * (r[2][1] - r[1][2]) / a,
            0.25 * (r[0][2] - r[2][0]) / a,
            0.25 * (r[1][0] - r[0][1]) / a,
        ]
    } else {
        let xd = 1.0 + r[0][0] - (r[1][1] + r[2][2]);
        let yd = 1.0 + r[1][1] - (r[0][0] + r[2][2]);
        let zd = 1.0 + r[2][2] - (r[0][0] + r[1][1]);
        if xd > 1.0 {
            let b = 0.5 * xd.sqrt();
            [
                0.25 * (r[2][1] - r[1][2]) / b,
                b,
                0.25 * (r[0][1] + r[1][0]) / b,
                0.25 * (r[0][2] + r[2][0]) / b,
            ]
        } else if yd > 1.0 {
            let c = 0.5 * yd.sqrt();
            [
                0.25 * (r[0][2] - r[2][0]) / c,
                0.25 * (r[0][1] + r[1][0]) / c,
                c,
                0.25 * (r[1][2] + r[2][1]) / c,
            ]
        } else {
            let d = 0.5 * zd.sqrt();
            [
                0.25 * (r[1][0] - r[0][1]) / d,
                0.25 * (r[0][2] + r[2][0]) / d,
                0.25 * (r[1][2] + r[2][1]) / d,
                d,
            ]
        }
    };
    if q[0] < 0.0 { q.map(|v| -v) } else { q }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axial(z: f64) -> Plane {
        Plane {
            position: [-100.0, -120.0, z],
            row: [1.0, 0.0, 0.0],
            column: [0.0, 1.0, 0.0],
            spacing: (0.8, 0.7),
        }
    }

    #[test]
    fn affine_turns_dicom_into_ras() {
        // Axial slices from z = 50 up to z = 59, 1 mm apart.
        let a = affine(&axial(50.0), Some(&axial(59.0)), 10, 1.0);
        assert_eq!(a[0], [-0.7, 0.0, 0.0, 100.0]);
        assert_eq!(a[1], [0.0, -0.8, 0.0, 120.0]);
        assert_eq!(a[2], [0.0, 0.0, 1.0, 50.0]);
        // Slices running downwards keep their order and a negative step.
        let down = affine(&axial(59.0), Some(&axial(50.0)), 10, 1.0);
        assert_eq!(down[2], [0.0, 0.0, -1.0, 59.0]);
        // One slice: the normal, one thickness long.
        let one = affine(&axial(50.0), None, 1, 2.5);
        assert_eq!(one[2], [0.0, 0.0, 2.5, 50.0]);
    }

    #[test]
    fn header_round_trips_through_the_reader() {
        let a = affine(&axial(50.0), Some(&axial(52.0)), 3, 1.0);
        let mut file = header([4, 3, 2], &a, Kind::I16);
        assert_eq!(file.len(), 352);
        let values: Vec<i16> = (0..24).collect();
        values
            .iter()
            .for_each(|v| file.extend_from_slice(&v.to_le_bytes()));
        let labels = read_labels(&file).unwrap();
        assert_eq!(labels.dims, [4, 3, 2]);
        assert_eq!(labels.voxels, (0..24).collect::<Vec<u16>>());
        // Turning LPS into RAS is a half turn about z: qfac 1 and the
        // quaternion (0, 0, 0, 1).
        let f = |at: usize| f32::from_le_bytes(file[at..at + 4].try_into().unwrap());
        assert_eq!(f(76), 1.0);
        assert_eq!((f(256), f(260), f(264)), (0.0, 0.0, 1.0));
        assert_eq!((f(268), f(272), f(276)), (100.0, 120.0, 50.0));
    }

    #[test]
    fn reads_gzip_big_endian_floats_and_nifti2() {
        use std::io::Write;
        // NIfTI-1, big-endian float32 labels with a slope, gzipped.
        let mut h = vec![0u8; 352];
        h[0..4].copy_from_slice(&348i32.to_be_bytes());
        for (k, d) in [3i16, 2, 2, 1, 1, 1, 1, 1].iter().enumerate() {
            h[40 + 2 * k..42 + 2 * k].copy_from_slice(&d.to_be_bytes());
        }
        h[70..72].copy_from_slice(&16i16.to_be_bytes());
        h[108..112].copy_from_slice(&352f32.to_be_bytes());
        h[112..116].copy_from_slice(&2f32.to_be_bytes());
        for v in [0.0f32, 1.0, 2.4, 3.0] {
            h.extend_from_slice(&v.to_be_bytes());
        }
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        gz.write_all(&h).unwrap();
        let labels = read_labels(&gz.finish().unwrap()).unwrap();
        assert_eq!(labels.dims, [2, 2, 1]);
        assert_eq!(labels.voxels, vec![0, 2, 5, 6]);

        // NIfTI-2, little-endian uint8.
        let mut h = vec![0u8; 544];
        h[0..4].copy_from_slice(&540i32.to_le_bytes());
        h[4..8].copy_from_slice(b"n+2\0");
        h[12..14].copy_from_slice(&2i16.to_le_bytes());
        for (k, d) in [3i64, 3, 1, 1, 1, 1, 1, 1].iter().enumerate() {
            h[16 + 8 * k..24 + 8 * k].copy_from_slice(&d.to_le_bytes());
        }
        h[168..176].copy_from_slice(&544i64.to_le_bytes());
        h.extend_from_slice(&[7, 0, 9]);
        let labels = read_labels(&h).unwrap();
        assert_eq!(labels.dims, [3, 1, 1]);
        assert_eq!(labels.voxels, vec![7, 0, 9]);

        assert!(read_labels(b"not a nifti file at all").is_err());
    }
}

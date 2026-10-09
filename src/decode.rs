//! Turns one frame of a DICOM object into values ready for a GPU texture.
//!
//! Uncompressed pixel data is read straight from the in-memory element, so
//! a frame of a large multi-frame file costs only that frame. Compressed
//! transfer syntaxes go through `dicom-pixeldata`'s codecs.

use std::sync::atomic::{AtomicU64, Ordering};

use dicom_core::value::{PrimitiveValue, Value};
use dicom_dictionary_std::tags;
use dicom_object::{DefaultDicomObject, InMemDicomObject};
use dicom_pixeldata::PixelDecoder;

use crate::scan::{get_str, get_u32};

pub enum Pixels {
    /// Modality values (rescale slope and intercept applied), one per pixel.
    Gray(Vec<f32>),
    /// 8-bit RGBA.
    Rgba(Vec<u8>),
}

pub struct Frame {
    /// Unique per decode, so the renderer knows when to upload.
    pub id: u64,
    pub width: u32,
    pub height: u32,
    pub pixels: Pixels,
    pub min: f32,
    pub max: f32,
    /// Window centre and width, from the file or from the value range.
    pub window: (f32, f32),
    /// MONOCHROME1: low values are displayed white.
    pub inverted: bool,
}

impl Frame {
    pub fn bytes(&self) -> usize {
        match &self.pixels {
            Pixels::Gray(v) => v.len() * 4,
            Pixels::Rgba(v) => v.len(),
        }
    }

    pub fn is_color(&self) -> bool {
        matches!(self.pixels, Pixels::Rgba(_))
    }

    /// Value at a pixel, as text for the overlay.
    pub fn value_at(&self, x: u32, y: u32) -> Option<String> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let i = (y * self.width + x) as usize;
        Some(match &self.pixels {
            Pixels::Gray(v) => {
                let v = v[i];
                if v.fract() == 0.0 {
                    format!("{v}")
                } else {
                    format!("{v:.2}")
                }
            }
            Pixels::Rgba(v) => format!("RGB {} {} {}", v[i * 4], v[i * 4 + 1], v[i * 4 + 2]),
        })
    }
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

const RLE_LOSSLESS: &str = "1.2.840.10008.1.2.5";

/// Image attributes needed to interpret the raw samples.
#[derive(Clone)]
struct Layout {
    rows: usize,
    cols: usize,
    samples: usize,
    bits_allocated: u32,
    bits_stored: u32,
    high_bit: u32,
    signed: bool,
    photometric: String,
    planar: bool,
}

impl Layout {
    fn pixels(&self) -> usize {
        self.rows * self.cols
    }
}

pub fn decode_frame(obj: &DefaultDicomObject, frame: u32, max_dim: u32) -> Result<Frame, String> {
    let pixel_data = obj.get(tags::PIXEL_DATA).ok_or("no pixel data")?;
    let rows = get_u32(obj, tags::ROWS).ok_or("missing Rows")? as usize;
    let cols = get_u32(obj, tags::COLUMNS).ok_or("missing Columns")? as usize;
    let bits_allocated = get_u32(obj, tags::BITS_ALLOCATED).unwrap_or(8);
    let bits_stored = get_u32(obj, tags::BITS_STORED).unwrap_or(bits_allocated);
    let layout = Layout {
        rows,
        cols,
        samples: get_u32(obj, tags::SAMPLES_PER_PIXEL).unwrap_or(1).max(1) as usize,
        bits_allocated,
        bits_stored,
        high_bit: get_u32(obj, tags::HIGH_BIT).unwrap_or(bits_stored.max(1) - 1),
        signed: get_u32(obj, tags::PIXEL_REPRESENTATION) == Some(1),
        photometric: get_str(obj, tags::PHOTOMETRIC_INTERPRETATION).to_ascii_uppercase(),
        planar: get_u32(obj, tags::PLANAR_CONFIGURATION) == Some(1),
    };
    if rows == 0 || cols == 0 {
        return Err("empty image".into());
    }

    let (slope, intercept) = rescale(obj, frame);
    let mut out = match pixel_data.value() {
        Value::Primitive(p) => native(obj, &layout, p, frame as usize, slope, intercept)?,
        Value::PixelSequence(seq) if obj.meta().transfer_syntax() == RLE_LOSSLESS => {
            let frames = get_u32(obj, tags::NUMBER_OF_FRAMES).unwrap_or(1).max(1);
            let data = rle_frame(seq.fragments(), frames, frame, &layout)?;
            let layout = Layout {
                planar: false,
                ..layout
            };
            convert(obj, &layout, Raw::Bytes(&data), slope, intercept)?
        }
        Value::PixelSequence(_) => {
            let decoded = obj
                .decode_pixel_data_frame(frame)
                .map_err(|e| e.to_string())?;
            let bits_allocated = decoded.bits_allocated() as u32;
            let layout = Layout {
                rows: decoded.rows() as usize,
                cols: decoded.columns() as usize,
                samples: decoded.samples_per_pixel() as usize,
                // 1-bit data is unpacked by the decoder to one byte per pixel.
                bits_allocated: if bits_allocated == 1 {
                    8
                } else {
                    bits_allocated
                },
                bits_stored: if bits_allocated == 1 {
                    8
                } else {
                    decoded.bits_stored() as u32
                },
                high_bit: if bits_allocated == 1 {
                    7
                } else {
                    decoded.high_bit() as u32
                },
                photometric: decoded
                    .photometric_interpretation()
                    .as_str()
                    .to_ascii_uppercase(),
                planar: false,
                ..layout
            };
            let raw = Raw::Bytes(decoded.data());
            convert(obj, &layout, raw, slope, intercept)?
        }
        Value::Sequence(_) => return Err("pixel data is a sequence".into()),
    };

    if let Some(w) = window(obj, frame).filter(|w| w.1 > 0.0)
        && !out.is_color()
    {
        out.window = w;
    }

    Ok(downsample(out, max_dim))
}

/// Rescale slope and intercept, including enhanced multi-frame functional groups.
fn rescale(obj: &DefaultDicomObject, frame: u32) -> (f32, f32) {
    let slope = functional(
        obj,
        frame,
        tags::PIXEL_VALUE_TRANSFORMATION_SEQUENCE,
        tags::RESCALE_SLOPE,
    )
    .filter(|s| *s != 0.0)
    .unwrap_or(1.0);
    let intercept = functional(
        obj,
        frame,
        tags::PIXEL_VALUE_TRANSFORMATION_SEQUENCE,
        tags::RESCALE_INTERCEPT,
    )
    .unwrap_or(0.0);
    (slope as f32, intercept as f32)
}

fn window(obj: &DefaultDicomObject, frame: u32) -> Option<(f32, f32)> {
    let c = functional(obj, frame, tags::FRAME_VOILUT_SEQUENCE, tags::WINDOW_CENTER)?;
    let w = functional(obj, frame, tags::FRAME_VOILUT_SEQUENCE, tags::WINDOW_WIDTH)?;
    Some((c as f32, w as f32))
}

/// Look a value up at the top level, then per-frame, then in the shared
/// functional groups. Multi-valued attributes yield their first value.
fn functional(
    obj: &DefaultDicomObject,
    frame: u32,
    group: dicom_core::Tag,
    tag: dicom_core::Tag,
) -> Option<f64> {
    fn first(o: &InMemDicomObject, tag: dicom_core::Tag) -> Option<f64> {
        o.get(tag)?.to_multi_float64().ok()?.first().copied()
    }
    fn nested(
        o: &InMemDicomObject,
        seq: dicom_core::Tag,
        index: usize,
        group: dicom_core::Tag,
        tag: dicom_core::Tag,
    ) -> Option<f64> {
        let item = o.get(seq)?.items()?.get(index)?;
        first(item.get(group)?.items()?.first()?, tag)
    }
    first(obj, tag)
        .or_else(|| {
            nested(
                obj,
                tags::PER_FRAME_FUNCTIONAL_GROUPS_SEQUENCE,
                frame as usize,
                group,
                tag,
            )
        })
        .or_else(|| nested(obj, tags::SHARED_FUNCTIONAL_GROUPS_SEQUENCE, 0, group, tag))
}

enum Raw<'a> {
    Bytes(&'a [u8]),
    Words(&'a [u16]),
}

fn native(
    obj: &DefaultDicomObject,
    layout: &Layout,
    value: &PrimitiveValue,
    frame: usize,
    slope: f32,
    intercept: f32,
) -> Result<Frame, String> {
    let samples_per_frame = if layout.photometric == "YBR_FULL_422" {
        layout.pixels() * 2
    } else {
        layout.pixels() * layout.samples
    };
    let range = |unit_bits: usize, len: usize| -> Result<std::ops::Range<usize>, String> {
        let per_frame = if layout.bits_allocated == 1 {
            samples_per_frame.div_ceil(8)
        } else {
            samples_per_frame * layout.bits_allocated as usize / unit_bits
        };
        let start = per_frame * frame;
        let end = start + per_frame;
        if end > len {
            Err(format!("frame {frame} is past the end of the pixel data"))
        } else {
            Ok(start..end)
        }
    };

    let owned;
    let raw = match value {
        PrimitiveValue::U8(b) if layout.bits_allocated == 1 => {
            let r = range(8, b.len())?;
            owned = unpack_bits(&b[r], layout.pixels());
            Raw::Bytes(&owned)
        }
        PrimitiveValue::U8(b) => Raw::Bytes(&b[range(8, b.len())?]),
        PrimitiveValue::U16(w) if layout.bits_allocated == 16 => {
            Raw::Words(&w[range(16, w.len())?])
        }
        other => {
            let bytes = other.to_bytes();
            let r = range(8, bytes.len())?;
            owned = bytes[r].to_vec();
            Raw::Bytes(&owned)
        }
    };

    if layout.bits_allocated == 1 {
        let layout = Layout {
            bits_allocated: 8,
            bits_stored: 8,
            high_bit: 7,
            ..layout.clone()
        };
        return convert(obj, &layout, raw, slope, intercept);
    }
    convert(obj, layout, raw, slope, intercept)
}

/// Decode one RLE Lossless frame into little-endian, pixel-interleaved
/// samples. Done here rather than in `dicom-pixeldata`, whose 0.10 decoder
/// shifts 8-bit greyscale data by one byte.
fn rle_frame(
    fragments: &[Vec<u8>],
    frames: u32,
    frame: u32,
    l: &Layout,
) -> Result<Vec<u8>, String> {
    let joined;
    let data: &[u8] = if fragments.len() == frames as usize {
        &fragments[frame as usize]
    } else if frames == 1 {
        joined = fragments.concat();
        &joined
    } else {
        return Err("RLE data does not have one fragment per frame".into());
    };
    if data.len() < 64 {
        return Err("RLE header is truncated".into());
    }
    let header: Vec<usize> = data[..64]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
        .collect();
    let bytes_per_sample = (l.bits_allocated / 8).max(1) as usize;
    let segments = header[0];
    if segments != bytes_per_sample * l.samples || segments > 15 {
        return Err(format!("unexpected number of RLE segments: {segments}"));
    }
    let n = l.pixels();
    let mut out = vec![0u8; n * l.samples * bytes_per_sample];
    for seg in 0..segments {
        let start = header[1 + seg].min(data.len());
        let end = if seg + 1 < segments {
            header[2 + seg]
        } else {
            data.len()
        }
        .clamp(start, data.len());
        let src = &data[start..end];
        // Segments run from the most significant byte of the first sample.
        let sample = seg / bytes_per_sample;
        let byte = bytes_per_sample - 1 - seg % bytes_per_sample;
        let at = |p: usize| (p * l.samples + sample) * bytes_per_sample + byte;
        let (mut i, mut p) = (0, 0);
        while i < src.len() && p < n {
            let h = src[i] as i8;
            i += 1;
            if h >= 0 {
                let count = (h as usize + 1).min(src.len() - i).min(n - p);
                for k in 0..count {
                    out[at(p + k)] = src[i + k];
                }
                i += h as usize + 1;
                p += count;
            } else if h != -128 && i < src.len() {
                let count = (1 - h as isize) as usize;
                let v = src[i];
                i += 1;
                for _ in 0..count.min(n - p) {
                    out[at(p)] = v;
                    p += 1;
                }
            }
        }
    }
    Ok(out)
}

fn unpack_bits(bytes: &[u8], count: usize) -> Vec<u8> {
    bytes
        .iter()
        .flat_map(|&b| (0..8).map(move |i| ((b >> i) & 1) * 255))
        .take(count)
        .collect()
}

/// Read sample `i` as an unsigned integer of `bits_allocated` bits.
#[inline(always)]
fn sample(raw: &Raw, bits_allocated: u32, i: usize) -> u32 {
    match raw {
        Raw::Words(w) => w[i] as u32,
        Raw::Bytes(b) => match bits_allocated {
            8 => b[i] as u32,
            16 => u16::from_le_bytes([b[2 * i], b[2 * i + 1]]) as u32,
            32 => u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]),
            _ => 0,
        },
    }
}

fn raw_len(raw: &Raw, bits_allocated: u32) -> usize {
    match raw {
        Raw::Words(w) => w.len(),
        Raw::Bytes(b) => b.len() * 8 / bits_allocated.max(8) as usize,
    }
}

fn convert(
    obj: &DefaultDicomObject,
    l: &Layout,
    raw: Raw,
    slope: f32,
    intercept: f32,
) -> Result<Frame, String> {
    if !matches!(l.bits_allocated, 8 | 16 | 32) {
        return Err(format!("unsupported Bits Allocated: {}", l.bits_allocated));
    }
    let n = l.pixels();
    let ba = l.bits_allocated;
    let have = raw_len(&raw, ba);

    let (width, height) = (l.cols as u32, l.rows as u32);
    let make = |pixels: Pixels, min: f32, max: f32, inverted: bool| Frame {
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        width,
        height,
        pixels,
        min,
        max,
        window: ((min + max) * 0.5, (max - min).max(1.0)),
        inverted,
    };

    if l.samples == 1 && l.photometric != "PALETTE COLOR" {
        if have < n {
            return Err("pixel data is shorter than expected".into());
        }
        let shift = (l.high_bit + 1).saturating_sub(l.bits_stored);
        let bits = l.bits_stored.clamp(1, 32);
        let mask: u32 = if bits == 32 {
            u32::MAX
        } else {
            (1 << bits) - 1
        };
        let sign_bit: u32 = 1 << (bits - 1);
        let signed = l.signed;
        let mut out = Vec::with_capacity(n);
        let (mut min, mut max) = (f32::MAX, f32::MIN);
        for i in 0..n {
            let v = (sample(&raw, ba, i) >> shift) & mask;
            let v = if signed && v & sign_bit != 0 {
                v as i64 - mask as i64 - 1
            } else {
                v as i64
            };
            let v = v as f32 * slope + intercept;
            min = min.min(v);
            max = max.max(v);
            out.push(v);
        }
        return Ok(make(
            Pixels::Gray(out),
            min,
            max,
            l.photometric == "MONOCHROME1",
        ));
    }

    // Colour: reduce every sample to 8 bits, then build RGBA.
    let shift = l.bits_stored.saturating_sub(8);
    let s8 = |i: usize| -> u8 { (sample(&raw, ba, i) >> shift).min(255) as u8 };
    let mut rgba = vec![255u8; n * 4];

    if l.photometric == "PALETTE COLOR" {
        let lut = Palette::read(obj)?;
        if have < n {
            return Err("pixel data is shorter than expected".into());
        }
        for i in 0..n {
            let rgb = lut.map(sample(&raw, ba, i));
            rgba[i * 4..i * 4 + 3].copy_from_slice(&rgb);
        }
    } else if l.photometric == "YBR_FULL_422" && l.samples == 3 {
        // Uncompressed 4:2:2 stores Y0 Y1 Cb Cr for each pair of pixels.
        if have < n * 2 {
            return Err("pixel data is shorter than expected".into());
        }
        for pair in 0..n / 2 {
            let (y0, y1, cb, cr) = (
                s8(pair * 4),
                s8(pair * 4 + 1),
                s8(pair * 4 + 2),
                s8(pair * 4 + 3),
            );
            rgba[pair * 8..pair * 8 + 3].copy_from_slice(&ybr_to_rgb(y0, cb, cr));
            rgba[pair * 8 + 4..pair * 8 + 7].copy_from_slice(&ybr_to_rgb(y1, cb, cr));
        }
    } else if l.samples >= 3 {
        if have < n * l.samples {
            return Err("pixel data is shorter than expected".into());
        }
        let ybr = l.photometric.starts_with("YBR_FULL");
        if let (Raw::Bytes(b), 8, false, false) = (&raw, ba, l.planar, ybr) {
            // The common case (ultrasound, secondary capture): plain 8-bit RGB.
            for (dst, src) in rgba.chunks_exact_mut(4).zip(b.chunks_exact(l.samples)) {
                dst[..3].copy_from_slice(&src[..3]);
            }
            return Ok(make(Pixels::Rgba(rgba), 0.0, 255.0, false));
        }
        for i in 0..n {
            let (a, b, c) = if l.planar {
                (s8(i), s8(n + i), s8(2 * n + i))
            } else {
                (
                    s8(i * l.samples),
                    s8(i * l.samples + 1),
                    s8(i * l.samples + 2),
                )
            };
            let rgb = if ybr { ybr_to_rgb(a, b, c) } else { [a, b, c] };
            rgba[i * 4..i * 4 + 3].copy_from_slice(&rgb);
        }
    } else {
        return Err(format!(
            "unsupported image: {} samples, {}",
            l.samples, l.photometric
        ));
    }
    Ok(make(Pixels::Rgba(rgba), 0.0, 255.0, false))
}

fn ybr_to_rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let (y, cb, cr) = (y as f32, cb as f32 - 128.0, cr as f32 - 128.0);
    let r = y + 1.402 * cr;
    let g = y - 0.344136 * cb - 0.714136 * cr;
    let b = y + 1.772 * cb;
    [r, g, b].map(|v| v.round().clamp(0.0, 255.0) as u8)
}

struct Palette {
    first: i64,
    tables: [Vec<u8>; 3],
}

impl Palette {
    fn read(obj: &DefaultDicomObject) -> Result<Self, String> {
        let descriptors = [
            tags::RED_PALETTE_COLOR_LOOKUP_TABLE_DESCRIPTOR,
            tags::GREEN_PALETTE_COLOR_LOOKUP_TABLE_DESCRIPTOR,
            tags::BLUE_PALETTE_COLOR_LOOKUP_TABLE_DESCRIPTOR,
        ];
        let data = [
            tags::RED_PALETTE_COLOR_LOOKUP_TABLE_DATA,
            tags::GREEN_PALETTE_COLOR_LOOKUP_TABLE_DATA,
            tags::BLUE_PALETTE_COLOR_LOOKUP_TABLE_DATA,
        ];
        let mut first = 0;
        let mut tables: [Vec<u8>; 3] = Default::default();
        for c in 0..3 {
            let desc = obj
                .get(descriptors[c])
                .and_then(|e| e.to_multi_int::<i64>().ok())
                .filter(|d| d.len() >= 3)
                .ok_or("missing or segmented palette, which is not supported")?;
            first = desc[1];
            let bits = desc[2];
            let elem = obj.get(data[c]).ok_or("missing palette data")?;
            let words: Vec<u16> = match elem.value() {
                Value::Primitive(PrimitiveValue::U16(w)) => w.to_vec(),
                Value::Primitive(p) => {
                    let b = p.to_bytes();
                    if bits == 8 && b.len() as i64 == desc[0].max(1) {
                        b.iter().map(|&x| x as u16).collect()
                    } else {
                        b.chunks_exact(2)
                            .map(|w| u16::from_le_bytes([w[0], w[1]]))
                            .collect()
                    }
                }
                _ => return Err("invalid palette data".into()),
            };
            tables[c] = words
                .iter()
                .map(|&w| if bits > 8 { (w >> 8) as u8 } else { w as u8 })
                .collect();
        }
        Ok(Palette { first, tables })
    }

    fn map(&self, index: u32) -> [u8; 3] {
        let get = |t: &Vec<u8>| {
            let i = (index as i64 - self.first)
                .clamp(0, t.len() as i64 - 1)
                .max(0) as usize;
            t.get(i).copied().unwrap_or(0)
        };
        [
            get(&self.tables[0]),
            get(&self.tables[1]),
            get(&self.tables[2]),
        ]
    }
}

/// Shrink images larger than the GPU's texture limit by an integer factor.
fn downsample(frame: Frame, max_dim: u32) -> Frame {
    let factor = frame.width.max(frame.height).div_ceil(max_dim.max(1));
    if factor <= 1 {
        return frame;
    }
    let (w, h) = (frame.width / factor, frame.height / factor);
    let (sw, f) = (frame.width as usize, factor as usize);
    let pixels = match &frame.pixels {
        Pixels::Gray(src) => {
            let mut out = Vec::with_capacity((w * h) as usize);
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let mut acc = 0.0;
                    for dy in 0..f {
                        let row = (y * f + dy) * sw + x * f;
                        acc += src[row..row + f].iter().sum::<f32>();
                    }
                    out.push(acc / (f * f) as f32);
                }
            }
            Pixels::Gray(out)
        }
        Pixels::Rgba(src) => {
            let mut out = Vec::with_capacity((w * h * 4) as usize);
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let i = ((y * f) * sw + x * f) * 4;
                    out.extend_from_slice(&src[i..i + 4]);
                }
            }
            Pixels::Rgba(out)
        }
    };
    Frame {
        width: w,
        height: h,
        pixels,
        ..frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom_core::value::PixelFragmentSequence;
    use dicom_core::{DataElement, Tag, VR};
    use dicom_object::meta::FileMetaTableBuilder;

    const EXPLICIT_LE: &str = "1.2.840.10008.1.2.1";

    fn object(
        ts: &str,
        attrs: Vec<(Tag, VR, PrimitiveValue)>,
        pixels: DataElement<InMemDicomObject>,
    ) -> DefaultDicomObject {
        let mut obj = InMemDicomObject::new_empty();
        for (tag, vr, v) in attrs {
            obj.put(DataElement::new(tag, vr, v));
        }
        obj.put(pixels);
        obj.with_exact_meta(
            FileMetaTableBuilder::new()
                .transfer_syntax(ts)
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7")
                .media_storage_sop_instance_uid("1.2.3.4")
                .build()
                .unwrap(),
        )
    }

    fn image(
        rows: u16,
        cols: u16,
        samples: u16,
        bits: (u16, u16),
        signed: bool,
        pi: &str,
    ) -> Vec<(Tag, VR, PrimitiveValue)> {
        vec![
            (tags::ROWS, VR::US, PrimitiveValue::from(rows)),
            (tags::COLUMNS, VR::US, PrimitiveValue::from(cols)),
            (
                tags::SAMPLES_PER_PIXEL,
                VR::US,
                PrimitiveValue::from(samples),
            ),
            (tags::BITS_ALLOCATED, VR::US, PrimitiveValue::from(bits.0)),
            (tags::BITS_STORED, VR::US, PrimitiveValue::from(bits.1)),
            (tags::HIGH_BIT, VR::US, PrimitiveValue::from(bits.1 - 1)),
            (
                tags::PIXEL_REPRESENTATION,
                VR::US,
                PrimitiveValue::from(signed as u16),
            ),
            (
                tags::PHOTOMETRIC_INTERPRETATION,
                VR::CS,
                PrimitiveValue::from(pi),
            ),
        ]
    }

    fn words(v: Vec<u16>) -> DataElement<InMemDicomObject> {
        DataElement::new(tags::PIXEL_DATA, VR::OW, PrimitiveValue::U16(v.into()))
    }

    fn bytes(v: Vec<u8>) -> DataElement<InMemDicomObject> {
        DataElement::new(tags::PIXEL_DATA, VR::OB, PrimitiveValue::U8(v.into()))
    }

    fn gray(f: &Frame) -> &[f32] {
        match &f.pixels {
            Pixels::Gray(v) => v,
            Pixels::Rgba(_) => panic!("expected greyscale"),
        }
    }

    fn rgba(f: &Frame) -> &[u8] {
        match &f.pixels {
            Pixels::Rgba(v) => v,
            Pixels::Gray(_) => panic!("expected colour"),
        }
    }

    #[test]
    fn signed_12_bit_with_rescale() {
        let mut attrs = image(1, 5, 1, (16, 12), true, "MONOCHROME2");
        attrs.push((tags::RESCALE_SLOPE, VR::DS, PrimitiveValue::from("2")));
        attrs.push((
            tags::RESCALE_INTERCEPT,
            VR::DS,
            PrimitiveValue::from("-100"),
        ));
        // The top nibble of the last value is junk that must be masked off.
        let obj = object(
            EXPLICIT_LE,
            attrs,
            words(vec![0, 0x0FFF, 0x0800, 0x07FF, 0xF001]),
        );
        let f = decode_frame(&obj, 0, 8192).unwrap();
        assert_eq!(gray(&f), &[-100.0, -102.0, -4196.0, 3994.0, -98.0]);
        assert_eq!((f.min, f.max), (-4196.0, 3994.0));
    }

    #[test]
    fn window_from_file_wins_and_monochrome1_inverts() {
        let mut attrs = image(1, 2, 1, (8, 8), false, "MONOCHROME1");
        attrs.push((
            tags::WINDOW_CENTER,
            VR::DS,
            dicom_core::dicom_value!(Strs, ["50", "60"]),
        ));
        attrs.push((
            tags::WINDOW_WIDTH,
            VR::DS,
            dicom_core::dicom_value!(Strs, ["10", "20"]),
        ));
        let obj = object(EXPLICIT_LE, attrs, bytes(vec![1, 200]));
        let f = decode_frame(&obj, 0, 8192).unwrap();
        assert!(f.inverted);
        assert_eq!(f.window, (50.0, 10.0));
    }

    #[test]
    fn picks_the_right_frame_of_a_multi_frame_file() {
        let mut attrs = image(2, 2, 1, (16, 16), false, "MONOCHROME2");
        attrs.push((tags::NUMBER_OF_FRAMES, VR::IS, PrimitiveValue::from("3")));
        let obj = object(EXPLICIT_LE, attrs, words((0..12).collect()));
        let f = decode_frame(&obj, 2, 8192).unwrap();
        assert_eq!(gray(&f), &[8.0, 9.0, 10.0, 11.0]);
        assert!(decode_frame(&obj, 3, 8192).is_err());
    }

    #[test]
    fn planar_rgb() {
        let mut attrs = image(1, 2, 3, (8, 8), false, "RGB");
        attrs.push((
            tags::PLANAR_CONFIGURATION,
            VR::US,
            PrimitiveValue::from(1u16),
        ));
        let obj = object(EXPLICIT_LE, attrs, bytes(vec![10, 11, 20, 21, 30, 31]));
        let f = decode_frame(&obj, 0, 8192).unwrap();
        assert_eq!(rgba(&f), &[10, 20, 30, 255, 11, 21, 31, 255]);
    }

    #[test]
    fn ybr_full_converts_to_rgb() {
        let attrs = image(1, 1, 3, (8, 8), false, "YBR_FULL");
        let obj = object(EXPLICIT_LE, attrs, bytes(vec![128, 128, 128, 0]));
        let f = decode_frame(&obj, 0, 8192).unwrap();
        assert_eq!(rgba(&f), &[128, 128, 128, 255]);
    }

    fn rle(segments: &[Vec<u8>]) -> Vec<u8> {
        let mut header = [0u32; 16];
        header[0] = segments.len() as u32;
        let mut offset = 64;
        for (i, s) in segments.iter().enumerate() {
            header[i + 1] = offset;
            offset += s.len() as u32;
        }
        let mut out: Vec<u8> = header.iter().flat_map(|h| h.to_le_bytes()).collect();
        for s in segments {
            out.extend_from_slice(s);
        }
        out
    }

    fn encapsulated(fragments: Vec<Vec<u8>>) -> DataElement<InMemDicomObject> {
        DataElement::new(
            tags::PIXEL_DATA,
            VR::OB,
            PixelFragmentSequence::new(Vec::<u32>::new(), fragments),
        )
    }

    #[test]
    fn rle_8_bit_is_not_shifted() {
        let attrs = image(1, 7, 1, (8, 8), false, "MONOCHROME2");
        // A literal run of three, then a replicate run of four.
        let data = rle(&[vec![2, 10, 20, 30, (-3i8) as u8, 99]]);
        let obj = object(RLE_LOSSLESS, attrs, encapsulated(vec![data]));
        let f = decode_frame(&obj, 0, 8192).unwrap();
        assert_eq!(gray(&f), &[10.0, 20.0, 30.0, 99.0, 99.0, 99.0, 99.0]);
    }

    #[test]
    fn rle_16_bit_puts_the_high_byte_first() {
        let attrs = image(1, 2, 1, (16, 16), false, "MONOCHROME2");
        let data = rle(&[vec![1, 0x01, 0x03], vec![1, 0x02, 0x04]]);
        let obj = object(RLE_LOSSLESS, attrs, encapsulated(vec![data]));
        let f = decode_frame(&obj, 0, 8192).unwrap();
        assert_eq!(gray(&f), &[258.0, 772.0]);
    }

    #[test]
    fn rle_multi_frame_uses_one_fragment_per_frame() {
        let mut attrs = image(1, 2, 1, (8, 8), false, "MONOCHROME2");
        attrs.push((tags::NUMBER_OF_FRAMES, VR::IS, PrimitiveValue::from("2")));
        let frames = vec![rle(&[vec![1, 1, 2]]), rle(&[vec![1, 3, 4]])];
        let obj = object(RLE_LOSSLESS, attrs, encapsulated(frames));
        assert_eq!(gray(&decode_frame(&obj, 1, 8192).unwrap()), &[3.0, 4.0]);
    }

    #[test]
    fn large_images_are_downsampled_to_fit_the_gpu() {
        let attrs = image(4, 4, 1, (8, 8), false, "MONOCHROME2");
        let obj = object(EXPLICIT_LE, attrs, bytes((0..16).collect()));
        let f = decode_frame(&obj, 0, 2).unwrap();
        assert_eq!((f.width, f.height), (2, 2));
        assert_eq!(gray(&f), &[2.5, 4.5, 10.5, 12.5]);
    }
}

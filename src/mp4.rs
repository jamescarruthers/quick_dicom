//! A minimal MP4 writer for one H.264 video track.
//!
//! The index (`moov`) goes before the media data (`mdat`), so players and
//! browsers can start playing before the whole file has downloaded.

/// An encoded H.264 track, ready to be written.
pub struct Track {
    pub width: u16,
    pub height: u16,
    /// Ticks per second for sample timing.
    pub timescale: u32,
    /// Duration of each frame, in ticks.
    pub frame_duration: u32,
    pub sps: Vec<u8>,
    pub pps: Vec<u8>,
    /// One entry per frame: NAL units, each prefixed by its 4-byte length.
    pub samples: Vec<Vec<u8>>,
    /// Whether each frame can be decoded on its own (an IDR frame).
    pub keyframes: Vec<bool>,
}

/// Split an Annex B stream (NAL units after 00 00 01 start codes) into
/// NAL units without their start codes.
pub fn split_annex_b(stream: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= stream.len() {
        if stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut nals = Vec::with_capacity(starts.len());
    for (k, &s) in starts.iter().enumerate() {
        let mut end = starts.get(k + 1).map_or(stream.len(), |&n| n - 3);
        // A four-byte start code leaves a zero byte at the end of the previous unit.
        while end > s && stream[end - 1] == 0 {
            end -= 1;
        }
        if end > s {
            nals.push(&stream[s..end]);
        }
    }
    nals
}

/// Serialise the track as a complete MP4 file.
pub fn write(track: &Track) -> Result<Vec<u8>, String> {
    let n = track.samples.len();
    if n == 0 {
        return Err("no frames to write".into());
    }
    let ftyp = boxed(b"ftyp", |b| {
        b.extend_from_slice(b"isom");
        b.extend_from_slice(&512u32.to_be_bytes());
        for brand in [b"isom", b"iso2", b"avc1", b"mp41"] {
            b.extend_from_slice(brand);
        }
    });
    let media_len: usize = track.samples.iter().map(Vec::len).sum();

    // The chunk offset depends on the size of `moov`, which does not depend
    // on the offset's value, so build it once to measure and once for real.
    let probe = moov(track, 0);
    let mdat_header = 8;
    let offset = ftyp.len() + probe.len() + mdat_header;
    let total = offset + media_len;
    if total > u32::MAX as usize {
        return Err("video is larger than 4 GB".into());
    }
    let moov = moov(track, offset as u32);

    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&ftyp);
    out.extend_from_slice(&moov);
    out.extend_from_slice(&((mdat_header + media_len) as u32).to_be_bytes());
    out.extend_from_slice(b"mdat");
    for s in &track.samples {
        out.extend_from_slice(s);
    }
    Ok(out)
}

fn boxed(kind: &[u8; 4], fill: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let mut b = vec![0, 0, 0, 0];
    b.extend_from_slice(kind);
    fill(&mut b);
    let len = b.len() as u32;
    b[..4].copy_from_slice(&len.to_be_bytes());
    b
}

/// A "full box": version and flags precede the payload.
fn full(kind: &[u8; 4], version: u8, flags: u32, fill: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    boxed(kind, |b| {
        b.extend_from_slice(&((version as u32) << 24 | flags).to_be_bytes());
        fill(b);
    })
}

trait Put {
    fn u16(&mut self, v: u16);
    fn u32(&mut self, v: u32);
    fn zeros(&mut self, n: usize);
}

impl Put for Vec<u8> {
    fn u16(&mut self, v: u16) {
        self.extend_from_slice(&v.to_be_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.extend_from_slice(&v.to_be_bytes());
    }
    fn zeros(&mut self, n: usize) {
        self.resize(self.len() + n, 0);
    }
}

const UNITY_MATRIX: [u32; 9] = [0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x4000_0000];

fn moov(t: &Track, chunk_offset: u32) -> Vec<u8> {
    let n = t.samples.len() as u32;
    let duration = n * t.frame_duration;

    let mvhd = full(b"mvhd", 0, 0, |b| {
        b.u32(0); // creation time
        b.u32(0); // modification time
        b.u32(t.timescale);
        b.u32(duration);
        b.u32(0x0001_0000); // rate 1.0
        b.u16(0x0100); // volume 1.0
        b.zeros(10);
        UNITY_MATRIX.iter().for_each(|&m| b.u32(m));
        b.zeros(24);
        b.u32(2); // next track ID
    });

    let tkhd = full(b"tkhd", 0, 0x3, |b| {
        b.u32(0);
        b.u32(0);
        b.u32(1); // track ID
        b.u32(0);
        b.u32(duration);
        b.zeros(8);
        b.u16(0); // layer
        b.u16(0); // alternate group
        b.u16(0); // volume
        b.u16(0);
        UNITY_MATRIX.iter().for_each(|&m| b.u32(m));
        b.u32((t.width as u32) << 16);
        b.u32((t.height as u32) << 16);
    });

    let mdhd = full(b"mdhd", 0, 0, |b| {
        b.u32(0);
        b.u32(0);
        b.u32(t.timescale);
        b.u32(duration);
        b.u16(0x55C4); // language "und"
        b.u16(0);
    });
    let hdlr = full(b"hdlr", 0, 0, |b| {
        b.u32(0);
        b.extend_from_slice(b"vide");
        b.zeros(12);
        b.extend_from_slice(b"VideoHandler\0");
    });

    let avcc = boxed(b"avcC", |b| {
        b.push(1); // configuration version
        b.push(t.sps.get(1).copied().unwrap_or(66)); // profile
        b.push(t.sps.get(2).copied().unwrap_or(0)); // profile compatibility
        b.push(t.sps.get(3).copied().unwrap_or(30)); // level
        b.push(0xFC | 3); // four-byte NAL lengths
        b.push(0xE0 | 1); // one SPS
        b.u16(t.sps.len() as u16);
        b.extend_from_slice(&t.sps);
        b.push(1); // one PPS
        b.u16(t.pps.len() as u16);
        b.extend_from_slice(&t.pps);
    });
    let avc1 = boxed(b"avc1", |b| {
        b.zeros(6);
        b.u16(1); // data reference index
        b.zeros(16);
        b.u16(t.width);
        b.u16(t.height);
        b.u32(0x0048_0000); // 72 dpi
        b.u32(0x0048_0000);
        b.u32(0);
        b.u16(1); // frames per sample
        b.zeros(32); // compressor name
        b.u16(0x0018); // depth
        b.u16(0xFFFF); // pre-defined -1
        b.extend_from_slice(&avcc);
    });
    let stsd = full(b"stsd", 0, 0, |b| {
        b.u32(1);
        b.extend_from_slice(&avc1);
    });
    let stts = full(b"stts", 0, 0, |b| {
        b.u32(1);
        b.u32(n);
        b.u32(t.frame_duration);
    });
    let stss = full(b"stss", 0, 0, |b| {
        let sync: Vec<u32> = (1..=n).filter(|&i| t.keyframes[i as usize - 1]).collect();
        b.u32(sync.len() as u32);
        sync.iter().for_each(|&i| b.u32(i));
    });
    let stsc = full(b"stsc", 0, 0, |b| {
        b.u32(1);
        b.u32(1); // first chunk
        b.u32(n); // all samples in one chunk
        b.u32(1); // sample description index
    });
    let stsz = full(b"stsz", 0, 0, |b| {
        b.u32(0);
        b.u32(n);
        t.samples.iter().for_each(|s| b.u32(s.len() as u32));
    });
    let stco = full(b"stco", 0, 0, |b| {
        b.u32(1);
        b.u32(chunk_offset);
    });
    let stbl = boxed(b"stbl", |b| {
        for part in [&stsd, &stts, &stss, &stsc, &stsz, &stco] {
            b.extend_from_slice(part);
        }
    });

    let vmhd = full(b"vmhd", 0, 1, |b| b.zeros(8));
    let dinf = boxed(b"dinf", |b| {
        let url = full(b"url ", 0, 1, |_| {});
        let dref = full(b"dref", 0, 0, |b| {
            b.u32(1);
            b.extend_from_slice(&url);
        });
        b.extend_from_slice(&dref);
    });
    let minf = boxed(b"minf", |b| {
        b.extend_from_slice(&vmhd);
        b.extend_from_slice(&dinf);
        b.extend_from_slice(&stbl);
    });
    let mdia = boxed(b"mdia", |b| {
        b.extend_from_slice(&mdhd);
        b.extend_from_slice(&hdlr);
        b.extend_from_slice(&minf);
    });
    let trak = boxed(b"trak", |b| {
        b.extend_from_slice(&tkhd);
        b.extend_from_slice(&mdia);
    });
    boxed(b"moov", |b| {
        b.extend_from_slice(&mvhd);
        b.extend_from_slice(&trak);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Walk the boxes in `data`, returning (type, payload) pairs.
    fn boxes(data: &[u8]) -> Vec<(String, &[u8])> {
        let mut out = Vec::new();
        let mut i = 0;
        while i + 8 <= data.len() {
            let len = u32::from_be_bytes(data[i..i + 4].try_into().unwrap()) as usize;
            let kind = String::from_utf8_lossy(&data[i + 4..i + 8]).into_owned();
            out.push((kind, &data[i + 8..i + len]));
            i += len;
        }
        assert_eq!(i, data.len(), "boxes must tile the buffer exactly");
        out
    }

    #[test]
    fn splits_three_and_four_byte_start_codes() {
        let stream = [
            0, 0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3, 0, 0, 0, 1, 0x65, 4, 5,
        ];
        let nals = split_annex_b(&stream);
        assert_eq!(nals, vec![&[0x67, 1, 2][..], &[0x68, 3], &[0x65, 4, 5]]);
    }

    #[test]
    fn writes_a_well_formed_file() {
        let track = Track {
            width: 64,
            height: 32,
            timescale: 90_000,
            frame_duration: 3000,
            sps: vec![0x67, 66, 0xC0, 30],
            pps: vec![0x68, 0xCE],
            samples: vec![
                vec![0, 0, 0, 2, 0x65, 1],
                vec![0, 0, 0, 1, 0x41],
                vec![0, 0, 0, 1, 0x41],
            ],
            keyframes: vec![true, false, false],
        };
        let file = write(&track).unwrap();
        let top = boxes(&file);
        let kinds: Vec<&str> = top.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(kinds, ["ftyp", "moov", "mdat"]);
        assert_eq!(
            top[2].1,
            &[0, 0, 0, 2, 0x65, 1, 0, 0, 0, 1, 0x41, 0, 0, 0, 1, 0x41]
        );

        // The chunk offset must point at the first sample inside mdat.
        let at = file.windows(4).position(|w| w == b"stco").unwrap();
        let offset = u32::from_be_bytes(file[at + 12..at + 16].try_into().unwrap()) as usize;
        assert_eq!(&file[offset..offset + 6], &[0, 0, 0, 2, 0x65, 1]);

        let moov = boxes(top[1].1);
        assert_eq!(moov[0].0, "mvhd");
        assert_eq!(moov[1].0, "trak");
    }
}

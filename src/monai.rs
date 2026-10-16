//! A client for MONAI Label (https://github.com/Project-MONAI/MONAILabel),
//! a server that runs AI segmentation models built with MONAI. The viewer
//! sends a series to the server as a NIfTI volume, with its place in the
//! patient, and shows the labels that come back.
//!
//! The server does the work; it runs wherever its user starts it, on this
//! machine or one with a GPU. Its API: `GET /info/` lists the models, and
//! `POST /infer/{model}?output=image` takes the volume as a multipart
//! upload and answers with the label volume.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crossbeam_channel::Receiver;
use rayon::prelude::*;

use crate::decode::{Frame, Pixels, decode_frame, plane};
use crate::nifti::{self, Kind};
use crate::scan::{FrameRef, Stack, open_dicom};

/// Where a MONAI Label server listens by default.
pub const DEFAULT_SERVER: &str = "http://127.0.0.1:8000";

/// A model the server offers.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub name: String,
    pub kind: String,
    pub description: String,
    /// Label numbers and names, background left out.
    pub labels: Vec<(u16, String)>,
}

/// The labels for one series, one per voxel.
pub struct Segmentation {
    /// Unique, so the GPU knows when to upload.
    pub id: u64,
    /// The stack it belongs to.
    pub key: String,
    pub model: String,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// Slice after slice, row after row; 0 is background.
    pub voxels: Vec<u16>,
    /// Names from the model.
    pub names: Vec<(u16, String)>,
    /// The labels that occur, in order.
    pub found: Vec<u16>,
    /// What each label that occurs covers, in the same order.
    pub structures: Vec<Structure>,
}

/// Where one label lies.
#[derive(Clone, Debug, PartialEq)]
pub struct Structure {
    pub label: u16,
    pub voxels: usize,
    /// The structure's own voxel nearest its centre, as (x, y, slice):
    /// where its name goes. A curved structure's centre may lie outside
    /// it; this voxel never does.
    pub anchor: [u32; 3],
    /// The first and last slices it appears on.
    pub slices: (u32, u32),
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

impl Segmentation {
    pub fn new(
        key: String,
        model: String,
        dims: [usize; 3],
        voxels: Vec<u16>,
        names: Vec<(u16, String)>,
    ) -> Self {
        let structures = structures(&voxels, dims);
        Segmentation {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            key,
            model,
            width: dims[0] as u32,
            height: dims[1] as u32,
            depth: dims[2] as u32,
            voxels,
            names,
            found: structures.iter().map(|s| s.label).collect(),
            structures,
        }
    }

    pub fn slice(&self, k: usize) -> &[u16] {
        let n = (self.width * self.height) as usize;
        &self.voxels[k * n..(k + 1) * n]
    }

    pub fn at(&self, x: u32, y: u32, k: u32) -> u16 {
        if x >= self.width || y >= self.height || k >= self.depth {
            return 0;
        }
        self.voxels[((k * self.height + y) * self.width + x) as usize]
    }

    pub fn name(&self, label: u16) -> String {
        self.names
            .iter()
            .find(|(id, _)| *id == label)
            .map_or_else(|| format!("label {label}"), |(_, n)| n.clone())
    }

    /// The largest label, which decides how the GPU stores them.
    pub fn max_label(&self) -> u16 {
        self.found.last().copied().unwrap_or(0)
    }
}

/// Each label's size, slices and anchor, from two passes over the voxels,
/// slices in parallel.
fn structures(voxels: &[u16], dims: [usize; 3]) -> Vec<Structure> {
    let [w, h, d] = dims;
    let plane = w * h;
    if plane == 0 || d == 0 {
        return Vec::new();
    }
    let top = voxels.par_iter().copied().max().unwrap_or(0) as usize;
    if top == 0 {
        return Vec::new();
    }

    // Counts, sums of positions and slice ranges.
    #[derive(Clone, Copy)]
    struct Sums {
        n: usize,
        x: f64,
        y: f64,
        z: f64,
        first: u32,
        last: u32,
    }
    let empty = Sums {
        n: 0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
        first: u32::MAX,
        last: 0,
    };
    let sums = voxels
        .par_chunks(plane)
        .enumerate()
        .fold(
            || vec![empty; top + 1],
            |mut acc, (k, slice)| {
                for (i, &v) in slice.iter().enumerate() {
                    if v == 0 {
                        continue;
                    }
                    let a = &mut acc[v as usize];
                    a.n += 1;
                    a.x += (i % w) as f64;
                    a.y += (i / w) as f64;
                    a.z += k as f64;
                    a.first = a.first.min(k as u32);
                    a.last = a.last.max(k as u32);
                }
                acc
            },
        )
        .reduce(
            || vec![empty; top + 1],
            |mut a, b| {
                for (p, q) in a.iter_mut().zip(b) {
                    p.n += q.n;
                    p.x += q.x;
                    p.y += q.y;
                    p.z += q.z;
                    p.first = p.first.min(q.first);
                    p.last = p.last.max(q.last);
                }
                a
            },
        );
    let centre: Vec<[f64; 3]> = sums
        .iter()
        .map(|s| {
            let n = s.n.max(1) as f64;
            [s.x / n, s.y / n, s.z / n]
        })
        .collect();

    // The voxel of each label nearest its centre.
    let nearest = voxels
        .par_chunks(plane)
        .enumerate()
        .fold(
            || vec![(f64::MAX, [0u32; 3]); top + 1],
            |mut acc, (k, slice)| {
                for (i, &v) in slice.iter().enumerate() {
                    if v == 0 {
                        continue;
                    }
                    let c = centre[v as usize];
                    let p = [(i % w) as f64, (i / w) as f64, k as f64];
                    let d2 = (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2);
                    if d2 < acc[v as usize].0 {
                        acc[v as usize] = (d2, [p[0] as u32, p[1] as u32, k as u32]);
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![(f64::MAX, [0u32; 3]); top + 1],
            |mut a, b| {
                for (p, q) in a.iter_mut().zip(b) {
                    if q.0 < p.0 {
                        *p = q;
                    }
                }
                a
            },
        );
    (1..=top)
        .filter(|&l| sums[l].n > 0)
        .map(|l| Structure {
            label: l as u16,
            voxels: sums[l].n,
            anchor: nearest[l].1,
            slices: (sums[l].first, sums[l].last),
        })
        .collect()
}

/// Which labels are shown: one byte per label, laid out as the shaders
/// read it, 256 labels to a row.
pub struct LabelMask {
    /// Unique, so the GPU knows when to upload and redraw.
    pub generation: u64,
    shown: Vec<u8>,
}

impl LabelMask {
    /// Every label shown but those in `hidden`.
    pub fn hiding(hidden: &std::collections::BTreeSet<u16>) -> LabelMask {
        let mut shown = vec![1u8; 65536];
        shown[0] = 0;
        for &l in hidden {
            shown[l as usize] = 0;
        }
        LabelMask {
            generation: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            shown,
        }
    }

    pub fn shows(&self, label: u16) -> bool {
        self.shown[label as usize] != 0
    }

    /// 256 × 256 bytes, row after row.
    pub fn bytes(&self) -> &[u8] {
        &self.shown
    }
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(10)))
        // Models can take minutes on a CPU; wait as long as they need.
        .timeout_global(None)
        .http_status_as_error(false)
        .build()
        .new_agent()
}

/// The models on the server that segment a whole volume by themselves.
/// Interactive ones, which need clicks or scribbles, are left out.
pub fn models(server: &str) -> Result<Vec<Model>, String> {
    let url = format!("{}/info/", server.trim_end_matches('/'));
    let mut response = agent()
        .get(&url)
        .call()
        .map_err(|e| format!("could not reach {server}: {e}"))?;
    let status = response.status().as_u16();
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("could not read the server's answer: {e}"))?;
    if status != 200 {
        return Err(format!("{server} answered {status}: {}", detail(&text)));
    }
    let info: serde_json::Value =
        serde_json::from_str(&text).map_err(|_| format!("{server} is not a MONAI Label server"))?;
    Ok(parse_models(&info))
}

fn parse_models(info: &serde_json::Value) -> Vec<Model> {
    let Some(models) = info.get("models").and_then(|m| m.as_object()) else {
        return Vec::new();
    };
    let text = |m: &serde_json::Value, k: &str| {
        m.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string()
    };
    models
        .iter()
        .filter(|(_, m)| {
            let kind = text(m, "type");
            let dimension = m.get("dimension").and_then(|d| d.as_u64()).unwrap_or(3);
            dimension == 3 && (kind == "segmentation" || kind == "deepedit")
        })
        .map(|(name, m)| Model {
            name: name.clone(),
            kind: text(m, "type"),
            description: text(m, "description"),
            labels: parse_labels(m.get("labels")),
        })
        .collect()
}

/// Labels come as `{"name": number}` or as a list of names numbered from 1.
fn parse_labels(labels: Option<&serde_json::Value>) -> Vec<(u16, String)> {
    let mut out: Vec<(u16, String)> = match labels {
        Some(serde_json::Value::Object(map)) => map
            .iter()
            .filter_map(|(name, id)| Some((u16::try_from(id.as_u64()?).ok()?, name.clone())))
            .collect(),
        Some(serde_json::Value::Array(list)) => list
            .iter()
            .filter_map(|v| v.as_str())
            .filter(|name| !name.eq_ignore_ascii_case("background"))
            .enumerate()
            .map(|(i, name)| (i as u16 + 1, name.to_string()))
            .collect(),
        _ => Vec::new(),
    };
    out.retain(|(id, _)| *id != 0);
    out.sort();
    out
}

/// The server's error message, from a FastAPI `{"detail": ...}` body.
fn detail(body: &str) -> String {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| {
            v.get("detail")
                .map(|d| d.as_str().map_or_else(|| d.to_string(), str::to_string))
        })
        .unwrap_or_else(|| body.chars().take(300).collect())
}

/// Characters that may stand unescaped in a URL path segment.
fn escape(segment: &str) -> String {
    segment
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

const BOUNDARY: &str = "quick-dicom-boundary-7d3f61c2a9e84b05";

/// A multipart form with `params` and the NIfTI `file`, built in one
/// buffer so a large volume is not copied twice.
fn multipart(nifti_header: &[u8], voxels: &[u8]) -> Vec<u8> {
    let head = format!(
        "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"params\"\r\n\r\n{{}}\r\n\
         --{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"image.nii\"\r\n\
         Content-Type: application/octet-stream\r\n\r\n"
    );
    let tail = format!("\r\n--{BOUNDARY}--\r\n");
    let mut body = Vec::with_capacity(head.len() + nifti_header.len() + voxels.len() + tail.len());
    body.extend_from_slice(head.as_bytes());
    body.extend_from_slice(nifti_header);
    body.extend_from_slice(voxels);
    body.extend_from_slice(tail.as_bytes());
    body
}

/// Run `model` on the server over a NIfTI volume, given as its header and
/// voxels. Returns the label volume's bytes.
fn infer(server: &str, model: &str, nifti_header: &[u8], voxels: &[u8]) -> Result<Vec<u8>, String> {
    let url = format!(
        "{}/infer/{}?output=image",
        server.trim_end_matches('/'),
        escape(model)
    );
    let mut response = agent()
        .post(&url)
        .header(
            "Content-Type",
            &format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .send(&multipart(nifti_header, voxels)[..])
        .map_err(|e| format!("could not reach {server}: {e}"))?;
    let status = response.status().as_u16();
    let body = response
        .body_mut()
        .with_config()
        .limit(u64::MAX)
        .read_to_vec()
        .map_err(|e| format!("could not read the server's answer: {e}"))?;
    if status != 200 {
        let text = String::from_utf8_lossy(&body);
        return Err(format!("{model} failed ({status}): {}", detail(&text)));
    }
    Ok(body)
}

/// A volume ready to send: voxels x fastest, as the NIfTI file stores them.
pub struct Scan {
    pub dims: [usize; 3],
    pub affine: nifti::Affine,
    pub values: Vec<f32>,
}

impl Scan {
    /// The NIfTI header and voxels: 16-bit integers when every value is a
    /// whole number that fits, as for CT, else 32-bit floats.
    fn encode(&self) -> (Vec<u8>, Vec<u8>) {
        let whole = self
            .values
            .iter()
            .all(|&v| v.fract() == 0.0 && (i16::MIN as f32..=i16::MAX as f32).contains(&v));
        let (kind, voxels) = if whole {
            let data = self
                .values
                .iter()
                .flat_map(|&v| (v as i16).to_le_bytes())
                .collect();
            (Kind::I16, data)
        } else {
            let data = self.values.iter().flat_map(|&v| v.to_le_bytes()).collect();
            (Kind::F32, data)
        };
        (nifti::header(self.dims, &self.affine, kind), voxels)
    }
}

/// Send `scan` to the server and read back its labels, which must match it
/// voxel for voxel.
pub fn segment(server: &str, model: &str, scan: &Scan) -> Result<nifti::Labels, String> {
    let (header, voxels) = scan.encode();
    let bytes = infer(server, model, &header, &voxels)?;
    let labels = nifti::read_labels(&bytes)?;
    if labels.dims != scan.dims {
        return Err(format!(
            "{model} returned labels of {:?} voxels for a scan of {:?}",
            labels.dims, scan.dims
        ));
    }
    Ok(labels)
}

/// Every frame of a stack at full resolution, with the stack's place in the
/// patient. Colour series are refused: segmentation models expect scans.
pub fn read_scan(
    frames: &[FrameRef],
    cancel: &AtomicBool,
    done: &dyn Fn(usize),
) -> Result<Scan, String> {
    let first = frames.first().ok_or("the series has no frames")?;
    let shared = if first.multi {
        Some(Arc::new(open_dicom(&first.path, false)?))
    } else {
        None
    };
    let open = |f: &FrameRef| match &shared {
        Some(obj) if f.path == first.path => Ok(obj.clone()),
        _ => open_dicom(&f.path, false).map(Arc::new),
    };
    let load =
        |f: &FrameRef| -> Result<Frame, String> { decode_frame(&*open(f)?, f.frame, u32::MAX) };

    let head = load(first)?;
    if head.is_color() {
        return Err("colour images cannot be segmented".into());
    }
    let (w, h) = (head.width as usize, head.height as usize);
    let mut values = vec![0f32; w * h * frames.len()];
    let batch = rayon::current_num_threads().max(1) * 2;
    for (c, chunk) in frames.chunks(batch).enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let decoded: Vec<Result<Frame, String>> = chunk.par_iter().map(load).collect();
        for (k, frame) in decoded.into_iter().enumerate() {
            let i = c * batch + k;
            let frame = frame.map_err(|e| format!("slice {} could not be read: {e}", i + 1))?;
            if (frame.width, frame.height) != (head.width, head.height) {
                return Err("the slices are not all the same size".into());
            }
            let Pixels::Gray(v) = frame.pixels else {
                return Err("colour images cannot be segmented".into());
            };
            values[i * w * h..(i + 1) * w * h].copy_from_slice(&v);
        }
        done(((c + 1) * batch).min(frames.len()));
    }

    let geometry = |f: &FrameRef| open(f).ok().and_then(|o| plane(&o, f.frame));
    let first_plane = geometry(first);
    let last_plane = frames.last().and_then(geometry);
    let affine = match first_plane {
        Some(p) => {
            let gap = open(first)
                .ok()
                .and_then(|o| crate::scan::get_f64(&o, dicom_dictionary_std::tags::SLICE_THICKNESS))
                .filter(|&t| t > 0.0)
                .unwrap_or(1.0);
            nifti::affine(&p, last_plane.as_ref(), frames.len(), gap)
        }
        // Without a position in the patient, square voxels of the pixel
        // size, or 1 mm, in a plain axial frame.
        None => {
            let s = head
                .spacing
                .map_or((1.0, 1.0), |s| (s.row as f64, s.col as f64));
            [
                [-s.1, 0.0, 0.0, 0.0],
                [0.0, -s.0, 0.0, 0.0],
                [0.0, 0.0, s.1, 0.0],
            ]
        }
    };
    Ok(Scan {
        dims: [w, h, frames.len()],
        affine,
        values,
    })
}

/// Segmentation of a stack running on a background thread. Dropping it
/// stops waiting for the answer (the server may still finish its work).
pub struct Segmenting {
    stage: Arc<Mutex<String>>,
    cancel: Arc<AtomicBool>,
    result: Receiver<Result<Segmentation, String>>,
}

impl Drop for Segmenting {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl Segmenting {
    pub fn start(stack: &Stack, server: String, model: Model, ctx: eframe::egui::Context) -> Self {
        let frames = stack.frames.clone();
        let key = stack.key.clone();
        let stage = Arc::new(Mutex::new(format!("Reading {} slices…", frames.len())));
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, result) = crossbeam_channel::bounded(1);
        let (s, c) = (stage.clone(), cancel.clone());
        std::thread::Builder::new()
            .name("monai".into())
            .spawn(move || {
                let say = |text: String| {
                    *s.lock().unwrap() = text;
                    ctx.request_repaint();
                };
                let total = frames.len();
                let out = (|| {
                    let scan = read_scan(&frames, &c, &|n| {
                        say(format!("Reading slices… {n} / {total}"))
                    })?;
                    if c.load(Ordering::Relaxed) {
                        return Err("cancelled".to_string());
                    }
                    let mb = (scan.values.len() * 2) >> 20;
                    say(format!(
                        "Sending about {mb} MB to {server}; {} is running…",
                        model.name
                    ));
                    let labels = segment(&server, &model.name, &scan)?;
                    if c.load(Ordering::Relaxed) {
                        return Err("cancelled".to_string());
                    }
                    Ok(Segmentation::new(
                        key,
                        model.name,
                        labels.dims,
                        labels.voxels,
                        model.labels,
                    ))
                })();
                let _ = tx.send(out);
                ctx.request_repaint();
            })
            .expect("failed to spawn the MONAI thread");
        Segmenting {
            stage,
            cancel,
            result,
        }
    }

    pub fn stage(&self) -> String {
        self.stage.lock().unwrap().clone()
    }

    pub fn poll(&self) -> Option<Result<Segmentation, String>> {
        self.result.try_recv().ok()
    }
}

/// Asking a server for its models, on a background thread.
pub struct Connecting {
    pub server: String,
    result: Receiver<Result<Vec<Model>, String>>,
}

impl Connecting {
    pub fn start(server: String, ctx: eframe::egui::Context) -> Self {
        let (tx, result) = crossbeam_channel::bounded(1);
        let s = server.clone();
        std::thread::Builder::new()
            .name("monai info".into())
            .spawn(move || {
                let _ = tx.send(models(&s));
                ctx.request_repaint();
            })
            .expect("failed to spawn the MONAI thread");
        Connecting { server, result }
    }

    pub fn poll(&self) -> Option<Result<Vec<Model>, String>> {
        self.result.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;

    #[test]
    fn reads_models_and_their_labels() {
        let info = serde_json::json!({
            "name": "radiology",
            "models": {
                "segmentation_spleen": {
                    "type": "segmentation", "dimension": 3,
                    "description": "Spleen", "labels": {"spleen": 1, "background": 0}
                },
                "deepedit": {"type": "deepedit", "dimension": 3, "labels": ["background", "liver", "kidney"]},
                "deepgrow_2d": {"type": "deepgrow", "dimension": 2, "labels": []},
                "localization": {"type": "segmentation", "dimension": 3, "labels": {"spine": 3, "c1": 1}}
            }
        });
        let models = parse_models(&info);
        let find = |n: &str| models.iter().find(|m| m.name == n).unwrap();
        assert_eq!(models.len(), 3, "the 2D click model is left out");
        assert_eq!(
            find("segmentation_spleen").labels,
            vec![(1, "spleen".into())]
        );
        assert_eq!(
            find("deepedit").labels,
            vec![(1, "liver".into()), (2, "kidney".into())]
        );
        assert_eq!(
            find("localization").labels,
            vec![(1, "c1".into()), (3, "spine".into())]
        );
        assert!(parse_models(&serde_json::json!({"hello": 1})).is_empty());
    }

    #[test]
    fn escapes_model_names_and_reads_errors() {
        assert_eq!(escape("seg 1/x"), "seg%201%2Fx");
        assert_eq!(detail(r#"{"detail": "no such model"}"#), "no such model");
        assert_eq!(detail("plain text"), "plain text");
    }

    /// A pretend MONAI Label server on a local port, answering one
    /// `/info/` and one `/infer/` request. Its model labels every voxel
    /// above 100 as 1, and every voxel above 1000 as 2.
    fn fake_server() -> (String, std::thread::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let mut seen = Vec::new();
            for stream in listener.incoming().take(2) {
                let mut stream = stream.unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                seen.push(line.trim().to_string());
                let mut length = 0usize;
                let mut content_type = String::new();
                loop {
                    let mut h = String::new();
                    reader.read_line(&mut h).unwrap();
                    if h.trim().is_empty() {
                        break;
                    }
                    let (k, v) = h.split_once(':').unwrap();
                    match k.to_ascii_lowercase().as_str() {
                        "content-length" => length = v.trim().parse().unwrap(),
                        "content-type" => content_type = v.trim().to_string(),
                        _ => {}
                    }
                }
                let mut body = vec![0u8; length];
                reader.read_exact(&mut body).unwrap();
                let reply = if line.starts_with("GET /info/") {
                    serde_json::json!({"models": {"seg": {"type": "segmentation", "dimension": 3,
                        "labels": {"soft": 1, "bone": 2}}}})
                    .to_string()
                    .into_bytes()
                } else {
                    seen.push(content_type);
                    // Find the uploaded NIfTI inside the multipart body.
                    let start = body.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
                    let file_at = body[start + 4..]
                        .windows(4)
                        .position(|w| w == b"\r\n\r\n")
                        .map(|p| start + 4 + p + 4)
                        .unwrap();
                    let nii = &body[file_at..];
                    assert_eq!(&nii[344..348], b"n+1\0");
                    let dims: Vec<i16> = (0..4)
                        .map(|k| i16::from_le_bytes([nii[40 + 2 * k], nii[41 + 2 * k]]))
                        .collect();
                    let n = (dims[1] * dims[2] * dims[3]) as usize;
                    let values: Vec<i16> = (0..n)
                        .map(|i| i16::from_le_bytes([nii[352 + 2 * i], nii[353 + 2 * i]]))
                        .collect();
                    // Labels as uint8, gzipped, as nibabel or ITK might write them.
                    let mut out = nii[..352].to_vec();
                    out[70..72].copy_from_slice(&2i16.to_le_bytes());
                    out[72..74].copy_from_slice(&8i16.to_le_bytes());
                    out.extend(values.iter().map(|&v| (v > 100) as u8 + (v > 1000) as u8));
                    let mut gz =
                        flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
                    gz.write_all(&out).unwrap();
                    gz.finish().unwrap()
                };
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    reply.len()
                )
                .unwrap();
                stream.write_all(&reply).unwrap();
            }
            seen
        });
        (url, handle)
    }

    #[test]
    fn talks_to_a_server() {
        let (url, server) = fake_server();
        let models = models(&url).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(
            models[0].labels,
            vec![(1, "soft".into()), (2, "bone".into())]
        );

        // A 3 x 2 x 2 scan: air, soft tissue and bone.
        let scan = Scan {
            dims: [3, 2, 2],
            affine: [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            values: vec![
                -1000.0, 40.0, 400.0, 1200.0, -50.0, 200.0, 0.0, 0.0, 1500.0, 30.0, 300.0, -900.0,
            ],
        };
        let labels = segment(&url, "seg", &scan).unwrap();
        assert_eq!(labels.dims, [3, 2, 2]);
        assert_eq!(labels.voxels, vec![0, 0, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0]);

        let seen = server.join().unwrap();
        assert!(seen[0].starts_with("GET /info/ "));
        assert!(seen[1].starts_with("POST /infer/seg?output=image "));
        assert!(seen[2].starts_with("multipart/form-data; boundary="));
    }

    #[test]
    fn segmentation_knows_its_labels() {
        let seg = Segmentation::new(
            "k".into(),
            "m".into(),
            [2, 2, 2],
            vec![0, 3, 0, 0, 1, 0, 0, 3],
            vec![(1, "liver".into())],
        );
        assert_eq!(seg.found, vec![1, 3]);
        assert_eq!(seg.max_label(), 3);
        assert_eq!(seg.at(1, 0, 0), 3);
        assert_eq!(seg.at(0, 0, 1), 1);
        assert_eq!(seg.at(5, 0, 0), 0);
        assert_eq!(seg.slice(1), &[1, 0, 0, 3]);
        assert_eq!(seg.name(1), "liver");
        assert_eq!(seg.name(3), "label 3");
        assert_eq!(
            seg.structures[0],
            Structure {
                label: 1,
                voxels: 1,
                anchor: [0, 0, 1],
                slices: (1, 1),
            }
        );
        assert_eq!(seg.structures[1].voxels, 2);
        assert_eq!(seg.structures[1].slices, (0, 1));
    }

    /// An arc's centre lies outside it; its name goes on its own voxel
    /// nearest the centre.
    #[test]
    fn names_go_on_the_structure() {
        let (w, h) = (9usize, 9usize);
        let mut voxels = vec![0u16; w * h * 2];
        // A U shape on slice 1: two arms and a base.
        for y in 1..8 {
            voxels[w * h + y * w + 1] = 4;
            voxels[w * h + y * w + 7] = 4;
        }
        for x in 1..8 {
            voxels[w * h + 7 * w + x] = 4;
        }
        let seg = Segmentation::new("k".into(), "m".into(), [w, h, 2], voxels, vec![]);
        let s = &seg.structures[0];
        assert_eq!(s.slices, (1, 1));
        let [x, y, k] = s.anchor;
        assert_eq!(seg.at(x, y, k), 4, "anchor {:?}", s.anchor);
        // The centre is at (4, 5.1); the nearest voxel of the U is on its base.
        assert_eq!(s.anchor, [4, 7, 1]);
    }

    #[test]
    fn masks_hide_labels() {
        let hidden = [3u16, 300].into_iter().collect();
        let m = LabelMask::hiding(&hidden);
        assert!(m.shows(1) && !m.shows(3) && !m.shows(300) && !m.shows(0));
        assert_eq!(m.bytes().len(), 65536);
        assert_ne!(LabelMask::hiding(&hidden).generation, m.generation);
    }
}

/// Runs against a real MONAI Label server when one is named, e.g.
/// `MONAI_SERVER=http://127.0.0.1:8000 MONAI_MODEL=segmentation_spleen
/// MONAI_SERIES=/path/to/slices cargo test real_server -- --ignored`.
#[cfg(test)]
mod real_server {
    use super::*;

    #[test]
    #[ignore]
    fn real_server() {
        let var = |k: &str| std::env::var(k).unwrap();
        let mut paths: Vec<_> = std::fs::read_dir(var("MONAI_SERIES"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        paths.sort();
        let frames: Vec<FrameRef> = paths
            .into_iter()
            .map(|p| FrameRef {
                path: p.into(),
                frame: 0,
                multi: false,
            })
            .collect();
        let scan = read_scan(&frames, &AtomicBool::new(false), &|_| {}).unwrap();
        if let Ok(out) = std::env::var("MONAI_NIFTI") {
            let (header, voxels) = scan.encode();
            std::fs::write(out, [header, voxels].concat()).unwrap();
        }
        let server = var("MONAI_SERVER");
        println!("models: {:?}", models(&server).unwrap());
        let start = std::time::Instant::now();
        let labels = segment(&server, &var("MONAI_MODEL"), &scan).unwrap();
        let mut counts = std::collections::BTreeMap::new();
        labels
            .voxels
            .iter()
            .for_each(|&v| *counts.entry(v).or_insert(0usize) += 1);
        println!(
            "{:?} labels in {:.1} s: {counts:?}",
            labels.dims,
            start.elapsed().as_secs_f32()
        );
    }
}

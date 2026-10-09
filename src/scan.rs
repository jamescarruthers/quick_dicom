//! Walks a folder tree, reads DICOM headers in parallel and groups the
//! images into patients, studies and scrollable stacks.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError};
use dicom_dictionary_std::tags;
use dicom_object::file::ReadPreamble;
use dicom_object::{DefaultDicomObject, OpenFileOptions};
use rayon::prelude::*;

/// One frame of one file. Single-frame files always use frame 0.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FrameRef {
    pub path: Arc<Path>,
    pub frame: u32,
    /// The file holds more than one frame, so it is worth keeping open.
    pub multi: bool,
}

/// Header fields we need for grouping, sorting and labelling.
#[derive(Clone, Debug)]
pub struct ImageInfo {
    pub path: Arc<Path>,
    pub patient_name: String,
    pub patient_id: String,
    pub study_uid: String,
    pub study_date: String,
    pub study_desc: String,
    pub series_uid: String,
    pub series_number: Option<i32>,
    pub series_desc: String,
    pub modality: String,
    pub instance_number: Option<i32>,
    pub frames: u32,
    pub rows: u32,
    pub cols: u32,
    /// Position along the slice normal, for sorting slices in space.
    pub slice_pos: Option<f64>,
    pub fps: Option<f32>,
}

/// A list of frames the user can scrub through: either every single-frame
/// image in a series, or all frames of one multi-frame file.
#[derive(Debug)]
pub struct Stack {
    /// Stable identity that survives rebuilding the library mid-scan.
    pub key: String,
    pub label: String,
    pub patient: String,
    pub study: String,
    pub modality: String,
    pub frames: Vec<FrameRef>,
    /// Largest image dimensions in the stack, for memory estimates.
    pub max_rows: u32,
    pub max_cols: u32,
    pub fps: Option<f32>,
}

#[derive(Debug)]
pub struct Study {
    pub label: String,
    /// Indices into [`Library::stacks`].
    pub stacks: std::ops::Range<usize>,
}

#[derive(Debug)]
pub struct Patient {
    pub label: String,
    pub studies: Vec<Study>,
}

#[derive(Debug, Default)]
pub struct Library {
    pub patients: Vec<Patient>,
    /// All stacks in display order, so next/previous is a simple index step.
    pub stacks: Vec<Arc<Stack>>,
}

impl Library {
    pub fn find_stack(&self, key: &str) -> Option<usize> {
        self.stacks.iter().position(|s| s.key == key)
    }

    pub fn find_path(&self, path: &Path) -> Option<(usize, usize)> {
        self.stacks.iter().enumerate().find_map(|(i, s)| {
            s.frames
                .iter()
                .position(|f| &*f.path == path)
                .map(|j| (i, j))
        })
    }
}

pub enum ScanEvent {
    Progress {
        files_seen: usize,
        images: usize,
    },
    Library(Arc<Library>),
    Done {
        files_seen: usize,
        images: usize,
        elapsed: Duration,
    },
}

/// A running background scan. Dropping it cancels the scan.
pub struct Scanner {
    pub events: Receiver<ScanEvent>,
    cancel: Arc<AtomicBool>,
}

impl Drop for Scanner {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl Scanner {
    pub fn start(root: PathBuf, ctx: eframe::egui::Context) -> Self {
        let (event_tx, events) = crossbeam_channel::unbounded();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_flag = cancel.clone();

        std::thread::Builder::new()
            .name("scan".into())
            .spawn(move || {
                let start = Instant::now();
                let files_seen = AtomicUsize::new(0);
                let (info_tx, info_rx) = crossbeam_channel::unbounded::<ImageInfo>();

                std::thread::scope(|s| {
                    // Collect results and publish a fresh library a few times a
                    // second, so the user can start browsing before the scan ends.
                    let publisher = s.spawn(|| {
                        let mut infos = Vec::new();
                        let mut dirty = false;
                        let mut last_publish = Instant::now();
                        loop {
                            match info_rx.recv_timeout(Duration::from_millis(50)) {
                                Ok(info) => {
                                    infos.push(info);
                                    dirty = true;
                                }
                                Err(RecvTimeoutError::Timeout) => {}
                                Err(RecvTimeoutError::Disconnected) => break,
                            }
                            if last_publish.elapsed() > Duration::from_millis(250) {
                                last_publish = Instant::now();
                                if dirty {
                                    dirty = false;
                                    let lib = build_library(&infos);
                                    let _ = event_tx.send(ScanEvent::Library(Arc::new(lib)));
                                }
                                let _ = event_tx.send(ScanEvent::Progress {
                                    files_seen: files_seen.load(Ordering::Relaxed),
                                    images: infos.len(),
                                });
                                ctx.request_repaint();
                            }
                        }
                        if !cancel_flag.load(Ordering::Relaxed) {
                            let lib = build_library(&infos);
                            let _ = event_tx.send(ScanEvent::Library(Arc::new(lib)));
                            let _ = event_tx.send(ScanEvent::Done {
                                files_seen: files_seen.load(Ordering::Relaxed),
                                images: infos.len(),
                                elapsed: start.elapsed(),
                            });
                            ctx.request_repaint();
                        }
                    });

                    walkdir::WalkDir::new(&root)
                        .follow_links(true)
                        .into_iter()
                        .filter_map(Result::ok)
                        .take_while(|_| !cancel_flag.load(Ordering::Relaxed))
                        .filter(|e| e.file_type().is_file())
                        .par_bridge()
                        .for_each_with(info_tx, |tx, entry| {
                            if cancel_flag.load(Ordering::Relaxed) {
                                return;
                            }
                            files_seen.fetch_add(1, Ordering::Relaxed);
                            if let Some(info) = read_header(entry.path()) {
                                let _ = tx.send(info);
                            }
                        });

                    let _ = publisher.join();
                });
            })
            .expect("failed to spawn scan thread");

        Scanner { events, cancel }
    }
}

/// Where the "DICM" magic sits, which tells us whether there is a preamble.
fn sniff(f: &mut File) -> Option<ReadPreamble> {
    let mut buf = [0u8; 132];
    let mut n = 0;
    while n < buf.len() {
        match f.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(_) => return None,
        }
    }
    if n == 132 && &buf[128..132] == b"DICM" {
        Some(ReadPreamble::Always)
    } else if n >= 4 && &buf[..4] == b"DICM" {
        Some(ReadPreamble::Never)
    } else {
        None
    }
}

/// Open a DICOM file. With `header_only`, stop before the pixel data.
pub fn open_dicom(path: &Path, header_only: bool) -> Result<DefaultDicomObject, String> {
    // One open per file: sniff the magic, rewind, then parse the same handle.
    // That matters on network shares, where each open costs a round trip.
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let preamble = sniff(&mut file).ok_or_else(|| "not a DICOM file".to_string())?;
    file.rewind().map_err(|e| e.to_string())?;
    let mut opts = OpenFileOptions::new().read_preamble(preamble);
    if header_only {
        opts = opts.read_until(tags::PIXEL_DATA);
    }
    opts.from_reader(file).map_err(|e| e.to_string())
}

fn read_header(path: &Path) -> Option<ImageInfo> {
    let obj = open_dicom(path, true).ok()?;
    let rows = get_u32(&obj, tags::ROWS)?;
    let cols = get_u32(&obj, tags::COLUMNS)?;
    if rows == 0 || cols == 0 {
        return None;
    }

    let slice_pos = (|| {
        let ipp = obj
            .get(tags::IMAGE_POSITION_PATIENT)?
            .to_multi_float64()
            .ok()?;
        let iop = obj
            .get(tags::IMAGE_ORIENTATION_PATIENT)?
            .to_multi_float64()
            .ok()?;
        if ipp.len() < 3 || iop.len() < 6 {
            return None;
        }
        let n = [
            iop[1] * iop[5] - iop[2] * iop[4],
            iop[2] * iop[3] - iop[0] * iop[5],
            iop[0] * iop[4] - iop[1] * iop[3],
        ];
        Some(ipp[0] * n[0] + ipp[1] * n[1] + ipp[2] * n[2])
    })();

    let fps = get_f64(&obj, tags::FRAME_TIME)
        .filter(|&t| t > 0.0)
        .map(|t| (1000.0 / t) as f32)
        .or_else(|| get_f64(&obj, tags::RECOMMENDED_DISPLAY_FRAME_RATE).map(|r| r as f32))
        .or_else(|| get_f64(&obj, tags::CINE_RATE).map(|r| r as f32))
        .filter(|&r| r > 0.0 && r < 1000.0);

    Some(ImageInfo {
        path: Arc::from(path),
        patient_name: get_str(&obj, tags::PATIENT_NAME)
            .replace('^', " ")
            .trim()
            .to_string(),
        patient_id: get_str(&obj, tags::PATIENT_ID),
        study_uid: get_str(&obj, tags::STUDY_INSTANCE_UID),
        study_date: get_str(&obj, tags::STUDY_DATE),
        study_desc: get_str(&obj, tags::STUDY_DESCRIPTION),
        series_uid: get_str(&obj, tags::SERIES_INSTANCE_UID),
        series_number: get_i32(&obj, tags::SERIES_NUMBER),
        series_desc: get_str(&obj, tags::SERIES_DESCRIPTION),
        modality: get_str(&obj, tags::MODALITY),
        instance_number: get_i32(&obj, tags::INSTANCE_NUMBER),
        frames: get_u32(&obj, tags::NUMBER_OF_FRAMES).unwrap_or(1).max(1),
        rows,
        cols,
        slice_pos,
        fps,
    })
}

pub fn get_str(obj: &DefaultDicomObject, tag: dicom_core::Tag) -> String {
    obj.get(tag)
        .and_then(|e| e.to_str().ok())
        .map(|s| {
            s.trim_matches(|c: char| c.is_whitespace() || c == '\0')
                .to_string()
        })
        .unwrap_or_default()
}

pub fn get_f64(obj: &DefaultDicomObject, tag: dicom_core::Tag) -> Option<f64> {
    obj.get(tag).and_then(|e| e.to_float64().ok())
}

pub fn get_i32(obj: &DefaultDicomObject, tag: dicom_core::Tag) -> Option<i32> {
    obj.get(tag).and_then(|e| e.to_int::<i32>().ok())
}

pub fn get_u32(obj: &DefaultDicomObject, tag: dicom_core::Tag) -> Option<u32> {
    obj.get(tag).and_then(|e| e.to_int::<u32>().ok())
}

fn fallback_key(info: &ImageInfo) -> String {
    info.path
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn format_date(d: &str) -> String {
    if d.len() == 8 && d.bytes().all(|b| b.is_ascii_digit()) {
        format!("{}-{}-{}", &d[..4], &d[4..6], &d[6..])
    } else {
        d.to_string()
    }
}

/// Group a flat list of headers into the patient / study / stack tree.
pub fn build_library(infos: &[ImageInfo]) -> Library {
    type SeriesMap<'a> = HashMap<String, Vec<&'a ImageInfo>>;
    type StudyMap<'a> = HashMap<String, SeriesMap<'a>>;
    let mut by_patient: HashMap<(String, String), StudyMap> = HashMap::new();

    for info in infos {
        let study = if info.study_uid.is_empty() {
            fallback_key(info)
        } else {
            info.study_uid.clone()
        };
        let series = if info.series_uid.is_empty() {
            fallback_key(info)
        } else {
            info.series_uid.clone()
        };
        by_patient
            .entry((info.patient_name.clone(), info.patient_id.clone()))
            .or_default()
            .entry(study)
            .or_default()
            .entry(series)
            .or_default()
            .push(info);
    }

    let mut patients: Vec<_> = by_patient.into_iter().collect();
    patients.sort_by(|a, b| a.0.cmp(&b.0));

    let mut lib = Library::default();

    for ((name, id), studies) in patients {
        let patient_label = match (name.is_empty(), id.is_empty()) {
            (true, true) => "(unnamed patient)".to_string(),
            (false, true) => name.clone(),
            (true, false) => id.clone(),
            (false, false) => format!("{name}  ·  {id}"),
        };

        let mut studies: Vec<_> = studies.into_iter().collect();
        // Sort by date, then UID, using the first image of each study.
        let first = |s: &SeriesMap| {
            s.values()
                .next()
                .and_then(|v| v.first())
                .map(|i| (i.study_date.clone(), i.study_desc.clone()))
                .unwrap_or_default()
        };
        studies.sort_by_cached_key(|(uid, s)| (first(s), uid.clone()));

        let mut out_studies = Vec::new();
        for (_, series) in studies {
            let (date, desc) = first(&series);
            let study_label = match (date.is_empty(), desc.is_empty()) {
                (true, true) => "(no study description)".to_string(),
                (false, true) => format_date(&date),
                (true, false) => desc.clone(),
                (false, false) => format!("{}  {}", format_date(&date), desc),
            };

            let start = lib.stacks.len();
            let mut series: Vec<_> = series.into_iter().collect();
            series.sort_by_cached_key(|(uid, imgs)| {
                (imgs[0].series_number.unwrap_or(i32::MAX), uid.clone())
            });

            for (uid, mut imgs) in series {
                imgs.sort_by(|a, b| {
                    a.instance_number
                        .cmp(&b.instance_number)
                        .then_with(|| {
                            a.slice_pos
                                .partial_cmp(&b.slice_pos)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .then_with(|| a.path.cmp(&b.path))
                });
                let head = imgs[0];
                let base_label = {
                    let mut parts = Vec::new();
                    if let Some(n) = head.series_number {
                        parts.push(format!("#{n}"));
                    }
                    if !head.modality.is_empty() {
                        parts.push(head.modality.clone());
                    }
                    if !head.series_desc.is_empty() {
                        parts.push(head.series_desc.clone());
                    }
                    if parts.is_empty() {
                        parts.push("(series)".into());
                    }
                    parts.join(" ")
                };
                let make = |key: String, label: String, list: &[&ImageInfo]| {
                    let frames = list
                        .iter()
                        .flat_map(|i| {
                            (0..i.frames).map(|f| FrameRef {
                                path: i.path.clone(),
                                frame: f,
                                multi: i.frames > 1,
                            })
                        })
                        .collect();
                    Arc::new(Stack {
                        key,
                        label,
                        patient: patient_label.clone(),
                        study: study_label.clone(),
                        modality: list[0].modality.clone(),
                        frames,
                        max_rows: list.iter().map(|i| i.rows).max().unwrap_or(0),
                        max_cols: list.iter().map(|i| i.cols).max().unwrap_or(0),
                        fps: list[0].fps,
                    })
                };

                let (singles, multis): (Vec<&ImageInfo>, Vec<&ImageInfo>) =
                    imgs.iter().partition(|i| i.frames <= 1);
                if !singles.is_empty() {
                    let label = format!("{base_label}  [{}]", singles.len());
                    lib.stacks.push(make(uid.clone(), label, &singles));
                }
                for m in &multis {
                    let mut label = base_label.clone();
                    if let Some(n) = m.instance_number.filter(|_| multis.len() > 1) {
                        label.push_str(&format!(" · {n}"));
                    }
                    label.push_str(&format!("  [{} fr]", m.frames));
                    let key = format!("{uid}|{}", m.path.display());
                    lib.stacks.push(make(key, label, std::slice::from_ref(m)));
                }
            }

            out_studies.push(Study {
                label: study_label,
                stacks: start..lib.stacks.len(),
            });
        }

        lib.patients.push(Patient {
            label: patient_label,
            studies: out_studies,
        });
    }

    lib
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(path: &str, series: &str, instance: i32, frames: u32) -> ImageInfo {
        ImageInfo {
            path: Arc::from(Path::new(path)),
            patient_name: "Doe Jane".into(),
            patient_id: "P1".into(),
            study_uid: "1.1".into(),
            study_date: "20260102".into(),
            study_desc: "Chest".into(),
            series_uid: series.into(),
            series_number: Some(if series == "s1" { 1 } else { 2 }),
            series_desc: "Axial".into(),
            modality: "CT".into(),
            instance_number: Some(instance),
            frames,
            rows: 512,
            cols: 512,
            slice_pos: None,
            fps: None,
        }
    }

    #[test]
    fn groups_and_sorts_series() {
        let infos = vec![
            info("/d/c", "s2", 1, 1),
            info("/d/b", "s1", 2, 1),
            info("/d/a", "s1", 10, 1),
            info("/d/z", "s1", 1, 1),
            info("/d/cine", "s1", 3, 4),
        ];
        let lib = build_library(&infos);
        assert_eq!(lib.patients.len(), 1);
        assert_eq!(lib.patients[0].studies.len(), 1);
        assert_eq!(lib.patients[0].studies[0].label, "2026-01-02  Chest");

        // Series 1 single frames, then its multi-frame file, then series 2.
        let paths: Vec<Vec<String>> = lib
            .stacks
            .iter()
            .map(|s| {
                s.frames
                    .iter()
                    .map(|f| f.path.display().to_string())
                    .collect()
            })
            .collect();
        assert_eq!(paths[0], ["/d/z", "/d/b", "/d/a"]);
        assert_eq!(paths[1], ["/d/cine"; 4]);
        assert_eq!(paths[2], ["/d/c"]);
        assert!(lib.stacks[1].frames.iter().all(|f| f.multi));
        assert_eq!(
            lib.stacks[1]
                .frames
                .iter()
                .map(|f| f.frame)
                .collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        assert_eq!(lib.find_path(Path::new("/d/a")), Some((0, 2)));
    }

    #[test]
    fn slice_position_breaks_instance_ties() {
        let mut a = info("/d/a", "s1", 1, 1);
        let mut b = info("/d/b", "s1", 1, 1);
        a.slice_pos = Some(5.0);
        b.slice_pos = Some(-5.0);
        let lib = build_library(&[a, b]);
        assert_eq!(&*lib.stacks[0].frames[0].path, Path::new("/d/b"));
    }
}

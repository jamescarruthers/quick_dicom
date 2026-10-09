//! `quick_dicom --check <folder>`: scan and decode everything without a
//! window, then report what can and cannot be shown.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use crate::decode::decode_frame;
use crate::scan::{Library, ScanEvent, Scanner, open_dicom};

pub fn run(root: PathBuf) -> bool {
    let scanner = Scanner::start(root, eframe::egui::Context::default());
    let mut library = Arc::new(Library::default());
    for event in scanner.events.iter() {
        match event {
            ScanEvent::Library(lib) => library = lib,
            ScanEvent::Done {
                files_seen,
                images,
                elapsed,
            } => {
                println!(
                    "scanned {files_seen} files, {images} images, {} stacks in {:.2} s",
                    library.stacks.len(),
                    elapsed.as_secs_f32()
                );
                break;
            }
            ScanEvent::Progress { .. } => {}
        }
    }

    let mut all_ok = true;
    for stack in &library.stacks {
        let start = Instant::now();
        // Group frames by file so multi-frame files are read once.
        let mut files: Vec<(PathBuf, Vec<u32>)> = Vec::new();
        for f in &stack.frames {
            match files.last_mut() {
                Some((p, frames)) if p.as_path() == &*f.path => frames.push(f.frame),
                _ => files.push((f.path.to_path_buf(), vec![f.frame])),
            }
        }
        let results: Vec<Result<(u32, u32, bool), String>> = files
            .par_iter()
            .flat_map_iter(|(path, frames)| {
                let obj = open_dicom(path, false).map(Arc::new);
                frames.iter().map(move |&i| {
                    let obj = obj.clone()?;
                    let f = decode_frame(&obj, i, 16384)
                        .map_err(|e| format!("{}#{i}: {e}", path.display()))?;
                    Ok((f.width, f.height, f.is_color()))
                })
            })
            .collect();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let failed: Vec<&String> = results.iter().filter_map(|r| r.as_ref().err()).collect();
        let dims = results
            .iter()
            .find_map(|r| r.as_ref().ok())
            .map(|(w, h, c)| format!("{w}x{h}{}", if *c { " colour" } else { "" }))
            .unwrap_or_default();
        println!(
            "{:<4} {:<40} {:>5} frames {:>16}  {:>8.1} ms  {}",
            if failed.is_empty() { "ok" } else { "FAIL" },
            truncate(&format!("{} / {}", stack.patient, stack.label), 40),
            stack.frames.len(),
            dims,
            ms,
            failed.first().map(|s| s.as_str()).unwrap_or("")
        );
        all_ok &= failed.is_empty();
    }
    all_ok
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n - 1).chain(std::iter::once('…')).collect()
    }
}

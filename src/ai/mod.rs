//! AI segmentation inside the viewer: MONAI's published models, read from
//! their PyTorch weights, which are downloaded once, and run without
//! Python.

pub mod gpu;
pub mod net;
#[cfg(not(target_os = "macos"))]
pub mod onnx;
#[cfg(not(target_os = "macos"))]
pub mod ort_cpu;
pub mod pipeline;
pub mod torch;

use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam_channel::Receiver;
use sha2::{Digest, Sha256};

use crate::monai::{Segmentation, read_scan};
use crate::scan::Stack;
use net::Net;
use pipeline::{Blend, Grid, Intensity, Pad, Recipe};

/// A model from MONAI's model zoo.
pub struct Model {
    pub name: &'static str,
    pub about: &'static str,
    /// Where the weights come from, pinned to one version.
    url: &'static str,
    sha256: &'static str,
    pub megabytes: u32,
    /// The file name in the model folder.
    file: &'static str,
    arch: Arch,
    recipe: Recipe,
    /// Label names; the label is the index, 0 for background.
    labels: &'static [&'static str],
}

enum Arch {
    /// MONAI UNet, 16 to 256 channels, residual units.
    SpleenUNet,
    /// MONAI SegResNet, 32 filters, 104 structures.
    WholeBody,
}

pub static MODELS: [Model; 3] = [
    Model {
        name: "Spleen",
        about: "The spleen in CT of the abdomen. Medical Segmentation Decathlon data.",
        url: "https://huggingface.co/MONAI/spleen_ct_segmentation/resolve/\
              8c4da75225ff7cb30292729dd704cceccae5413c/models/model.pt",
        sha256: "502c312893994ec071a85c4f2e7d83a43f7789969faa73dae62f1291177f50fe",
        megabytes: 19,
        file: "spleen_ct_segmentation.pt",
        arch: Arch::SpleenUNet,
        recipe: Recipe {
            spacing: [1.5, 1.5, 2.0],
            intensity: Intensity::Window {
                lo: -57.0,
                hi: 164.0,
            },
            block: 96,
            overlap: 0.5,
            pad: Pad::Constant(0.0),
            blend: Blend::Constant,
            smooth: true,
        },
        labels: &["background", "spleen"],
    },
    Model {
        name: "Whole body, quick (3 mm)",
        about: "104 organs, bones, muscles and vessels in CT, at 3 mm. TotalSegmentator data.",
        url: "https://huggingface.co/MONAI/wholeBody_ct_segmentation/resolve/\
              13f6d33ae9de6664571d935ee647039f477c3fad/models/model_lowres.pt",
        sha256: "c3ab55eb979785fdcb30690872c210bbeee73d79a170c32fdaa1eca117779f90",
        megabytes: 75,
        file: "wholeBody_ct_segmentation_3mm.pt",
        arch: Arch::WholeBody,
        recipe: Recipe {
            spacing: [3.0, 3.0, 3.0],
            intensity: Intensity::NormalizeThenRange,
            block: 96,
            overlap: 0.25,
            pad: Pad::Edge,
            blend: Blend::Gaussian,
            smooth: false,
        },
        labels: &WHOLE_BODY,
    },
    Model {
        name: "Whole body, detailed (1.5 mm)",
        about: "The same 104 structures at 1.5 mm: finer, and eight times the work.",
        url: "https://huggingface.co/MONAI/wholeBody_ct_segmentation/resolve/\
              13f6d33ae9de6664571d935ee647039f477c3fad/models/model.pt",
        sha256: "80b429fb4b080df11c9ed0b0bdaa8a615ff083921bb213a512cf285afbc4e3fe",
        megabytes: 75,
        file: "wholeBody_ct_segmentation_1.5mm.pt",
        arch: Arch::WholeBody,
        recipe: Recipe {
            spacing: [1.5, 1.5, 1.5],
            intensity: Intensity::NormalizeThenRange,
            block: 96,
            overlap: 0.25,
            pad: Pad::Edge,
            blend: Blend::Gaussian,
            smooth: false,
        },
        labels: &WHOLE_BODY,
    },
];

#[rustfmt::skip]
static WHOLE_BODY: [&str; 105] = [
    "background", "spleen", "kidney right", "kidney left", "gallbladder", "liver", "stomach",
    "aorta", "inferior vena cava", "portal vein and splenic vein", "pancreas",
    "adrenal gland right", "adrenal gland left", "lung upper lobe left",
    "lung lower lobe left", "lung upper lobe right", "lung middle lobe right",
    "lung lower lobe right", "vertebrae L5", "vertebrae L4", "vertebrae L3", "vertebrae L2",
    "vertebrae L1", "vertebrae T12", "vertebrae T11", "vertebrae T10", "vertebrae T9",
    "vertebrae T8", "vertebrae T7", "vertebrae T6", "vertebrae T5", "vertebrae T4",
    "vertebrae T3", "vertebrae T2", "vertebrae T1", "vertebrae C7", "vertebrae C6",
    "vertebrae C5", "vertebrae C4", "vertebrae C3", "vertebrae C2", "vertebrae C1",
    "esophagus", "trachea", "heart myocardium", "heart atrium left", "heart ventricle left",
    "heart atrium right", "heart ventricle right", "pulmonary artery", "brain",
    "iliac artery left", "iliac artery right", "iliac vena left", "iliac vena right",
    "small bowel", "duodenum", "colon", "rib left 1", "rib left 2", "rib left 3", "rib left 4",
    "rib left 5", "rib left 6", "rib left 7", "rib left 8", "rib left 9", "rib left 10",
    "rib left 11", "rib left 12", "rib right 1", "rib right 2", "rib right 3", "rib right 4",
    "rib right 5", "rib right 6", "rib right 7", "rib right 8", "rib right 9", "rib right 10",
    "rib right 11", "rib right 12", "humerus left", "humerus right", "scapula left",
    "scapula right", "clavicula left", "clavicula right", "femur left", "femur right",
    "hip left", "hip right", "sacrum", "face", "gluteus maximus left", "gluteus maximus right",
    "gluteus medius left", "gluteus medius right", "gluteus minimus left",
    "gluteus minimus right", "autochthon left", "autochthon right", "iliopsoas left",
    "iliopsoas right", "urinary bladder",
];

impl Model {
    fn build(&self, bytes: &[u8]) -> Result<Net, String> {
        let mut weights = torch::read(bytes)?;
        match self.arch {
            Arch::SpleenUNet => net::unet(&mut weights, 1, 2, &[16, 32, 64, 128, 256]),
            Arch::WholeBody => net::segresnet(&mut weights, 1, 105, 32, &[1, 2, 2, 4], &[1, 1, 1]),
        }
    }

    fn path(&self) -> Option<PathBuf> {
        models_dir().map(|d| d.join(self.file))
    }

    /// Whether the weights are already on this computer.
    pub fn downloaded(&self) -> bool {
        self.path().is_some_and(|p| p.is_file())
    }

    fn names(&self) -> Vec<(u16, String)> {
        self.labels
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, n)| (i as u16, n.to_string()))
            .collect()
    }
}

/// Where downloaded weights are kept: `QUICK_DICOM_MODELS`, else the
/// system's cache folder.
pub fn models_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("QUICK_DICOM_MODELS") {
        return Some(d.into());
    }
    let env = |k: &str| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let base = if cfg!(windows) {
        env("LOCALAPPDATA")
    } else if cfg!(target_os = "macos") {
        env("HOME").map(|h| h.join("Library/Caches"))
    } else {
        env("XDG_CACHE_HOME").or_else(|| env("HOME").map(|h| h.join(".cache")))
    }?;
    Some(base.join("quick_dicom").join("models"))
}

/// Downloads the model's weights unless they are here already, checking
/// them against their SHA-256. Returns the file's bytes.
fn weights(model: &Model, cancel: &AtomicBool, say: &dyn Fn(String)) -> Result<Vec<u8>, String> {
    let path = model
        .path()
        .ok_or("no folder to keep models in; set QUICK_DICOM_MODELS")?;
    // A damaged file is downloaded again.
    if let Ok(bytes) = std::fs::read(&path)
        && sha256(&bytes) == model.sha256
    {
        return Ok(bytes);
    }
    let dir = path.parent().unwrap();
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot make {}: {e}", dir.display()))?;
    say(format!("Downloading the {} model…", model.name));
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(std::time::Duration::from_secs(15)))
        .build()
        .into();
    let response = agent
        .get(model.url)
        .call()
        .map_err(|e| format!("could not download the model: {e}"))?;
    let total = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok()?.parse::<u64>().ok());
    let mut reader = response.into_body().into_reader();
    let mut bytes = Vec::with_capacity(total.unwrap_or(0) as usize);
    let mut hash = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut last = 0;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let n = reader
            .read(&mut buf)
            .map_err(|e| format!("the download stopped: {e}"))?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
        bytes.extend_from_slice(&buf[..n]);
        let mb = bytes.len() >> 20;
        if mb != last {
            last = mb;
            say(match total {
                Some(t) => format!(
                    "Downloading the {} model… {mb} / {} MB",
                    model.name,
                    t >> 20
                ),
                None => format!("Downloading the {} model… {mb} MB", model.name),
            });
        }
    }
    let digest: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if digest != model.sha256 {
        return Err("the downloaded model is not the expected file".into());
    }
    // Write beside, then rename, so a half-written file never looks done.
    let part = path.with_extension("part");
    std::fs::File::create(&part)
        .and_then(|mut f| f.write_all(&bytes))
        .and_then(|_| std::fs::rename(&part, &path))
        .map_err(|e| format!("cannot save the model in {}: {e}", dir.display()))?;
    Ok(bytes)
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Runs a network on one cube at a time.
pub trait Engine: Send {
    /// Each class's scores for the cube, class by class.
    fn logits(&mut self, cube: &[f32], side: usize) -> Result<Vec<f32>, String>;
}

impl Engine for gpu::GpuNet {
    fn logits(&mut self, cube: &[f32], side: usize) -> Result<Vec<f32>, String> {
        gpu::GpuNet::logits(self, cube, side)
    }
}

#[cfg(not(target_os = "macos"))]
impl Engine for ort_cpu::OrtNet {
    fn logits(&mut self, cube: &[f32], side: usize) -> Result<Vec<f32>, String> {
        ort_cpu::OrtNet::logits(self, cube, [side; 3])
    }
}

/// Where a model runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Gpu,
    Cpu,
}

/// Whether ONNX Runtime is built in: on Linux and Windows.
pub const CPU: bool = cfg!(not(target_os = "macos"));

/// The name of the GPU the models would run on, if there is one that is
/// not a software stand-in. Looked for once.
pub fn gpu_name() -> Option<String> {
    static NAME: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    NAME.get_or_init(|| {
        let instance =
            eframe::egui_wgpu::wgpu::Instance::new(eframe::egui_wgpu::wgpu::InstanceDescriptor {
                backends: eframe::egui_wgpu::wgpu::Backends::PRIMARY,
                ..eframe::egui_wgpu::wgpu::InstanceDescriptor::new_without_display_handle()
            });
        let options = eframe::egui_wgpu::wgpu::RequestAdapterOptions {
            power_preference: eframe::egui_wgpu::wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        };
        let adapter = pollster::block_on(instance.request_adapter(&options)).ok()?;
        let info = adapter.get_info();
        // QUICK_DICOM_SOFTWARE_GPU lets a software GPU stand in, for tests.
        let software = std::env::var_os("QUICK_DICOM_SOFTWARE_GPU").is_some();
        (info.device_type != eframe::egui_wgpu::wgpu::DeviceType::Cpu || software)
            .then_some(info.name)
    })
    .clone()
}

/// An engine for `net` on `device`, and where it runs, for messages.
fn engine(net: &Net, device: Device) -> Result<(Box<dyn Engine>, String), String> {
    match device {
        Device::Gpu => {
            let software = std::env::var_os("QUICK_DICOM_SOFTWARE_GPU").is_some();
            let g = gpu::GpuNet::new(net, software)?;
            let place = format!("on {}", g.adapter);
            Ok((Box::new(g), place))
        }
        #[cfg(not(target_os = "macos"))]
        Device::Cpu => Ok((
            Box::new(ort_cpu::OrtNet::new(net)?),
            "on the CPU".to_string(),
        )),
        #[cfg(target_os = "macos")]
        Device::Cpu => Err("on a Mac the models run on the GPU".into()),
    }
}

/// The last engine built, kept for the next run of the same model on the
/// same device: the model, the device, the engine and where it runs.
type Kept = (&'static str, Device, Box<dyn Engine>, String);
static LAST: Mutex<Option<Kept>> = Mutex::new(None);

/// Labels for the scan's voxels, in its own order.
pub fn segment(
    scan: &crate::monai::Scan,
    model: &Model,
    engine: &mut dyn Engine,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Vec<u16>, String> {
    let r = &model.recipe;
    let grid = Grid::covering(scan, r.spacing);
    let mut values = pipeline::resample(scan, &grid);
    pipeline::scale(&mut values, r.intensity);
    let mut run = |cube: &[f32]| engine.logits(cube, r.block);
    let classes = model.labels.len();
    let budget = pipeline::AVERAGE_BUDGET;
    let output = pipeline::run_blocks(
        &values, &grid, r, classes, &mut run, budget, cancel, progress,
    )?;
    Ok(pipeline::to_scan(&output, &grid, scan))
}

/// How many of the scan's voxels each of the model's spans, along each of
/// the scan's axes: the size of the blocks its labels come back in. One
/// where the labels come back smooth.
fn blocks(scan: &crate::monai::Scan, model: &Model) -> [f32; 3] {
    if model.recipe.smooth {
        return [1.0; 3];
    }
    std::array::from_fn(|a| {
        // The scan's axis in the patient's space, and the model's voxel
        // measured along it.
        let axis: [f64; 3] = std::array::from_fn(|b| scan.affine[b][a]);
        let length = axis.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-9);
        let along: f64 = (0..3)
            .map(|b| (axis[b] / length).abs() * model.recipe.spacing[b])
            .sum();
        (along / length) as f32
    })
}

/// A built-in model segmenting a stack on a background thread. Dropping it
/// stops the work.
pub struct Job {
    stage: Arc<Mutex<String>>,
    cancel: Arc<AtomicBool>,
    result: Receiver<Result<Done, String>>,
}

/// A finished segmentation, the seconds the model took, and where it ran.
pub type Done = (Segmentation, f32, String);

impl Drop for Job {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl Job {
    pub fn start(
        stack: &Stack,
        model: &'static Model,
        device: Device,
        ctx: eframe::egui::Context,
    ) -> Job {
        let frames = stack.frames.clone();
        let key = stack.key.clone();
        let stage = Arc::new(Mutex::new(format!("Starting {}…", model.name)));
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, result) = crossbeam_channel::bounded(1);
        let (s, c) = (stage.clone(), cancel.clone());
        std::thread::Builder::new()
            .name("segment".into())
            .spawn(move || {
                let say = |text: String| {
                    *s.lock().unwrap() = text;
                    ctx.request_repaint();
                };
                let out = (|| {
                    let bytes = weights(model, &c, &say)?;
                    let total = frames.len();
                    let scan = read_scan(&frames, &c, &|n| {
                        say(format!("Reading slices… {n} / {total}"))
                    })?;
                    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
                    let (mut engine, place) = match last.take() {
                        Some((name, d, e, place)) if name == model.name && d == device => {
                            (e, place)
                        }
                        _ => {
                            say(format!("Preparing the {} model…", model.name));
                            engine(&model.build(&bytes)?, device)?
                        }
                    };
                    let start = std::time::Instant::now();
                    let mut progress = |done: usize, total: usize| {
                        say(format!(
                            "{}: block {} of {total}…",
                            model.name,
                            (done + 1).min(total)
                        ));
                    };
                    let voxels = segment(&scan, model, engine.as_mut(), &c, &mut progress);
                    *last = Some((model.name, device, engine, place.clone()));
                    let seconds = start.elapsed().as_secs_f32();
                    let seg = Segmentation::made_at(
                        key,
                        model.name.to_string(),
                        scan.dims,
                        voxels?,
                        model.names(),
                        blocks(&scan, model),
                    );
                    Ok((seg, seconds, place))
                })();
                let _ = tx.send(out);
                ctx.request_repaint();
            })
            .expect("failed to spawn the segmentation thread");
        Job {
            stage,
            cancel,
            result,
        }
    }

    pub fn stage(&self) -> String {
        self.stage.lock().unwrap().clone()
    }

    /// The segmentation, the seconds the model took and where it ran,
    /// once done.
    pub fn poll(&self) -> Option<Result<Done, String>> {
        self.result.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_are_hex() {
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn models_are_described_completely() {
        for m in &MODELS {
            assert_eq!(m.sha256.len(), 64, "{}", m.name);
            assert!(
                m.url.starts_with("https://huggingface.co/MONAI/"),
                "{}",
                m.name
            );
            assert!(!m.url.contains(' '), "{}", m.name);
            assert_eq!(m.labels[0], "background");
        }
        assert_eq!(WHOLE_BODY.len(), 105);
        assert_eq!(WHOLE_BODY[104], "urinary bladder");
    }

    /// The blocks labels come back in: a 3 mm model on 0.75 mm pixels and
    /// 5 mm slices, rows running back to front, spans four pixels each way
    /// and under one slice; a smooth recipe has none.
    #[test]
    fn blocks_follow_the_scan_axes() {
        let scan = crate::monai::Scan {
            dims: [4, 4, 4],
            affine: [
                [-0.75, 0.0, 0.0, 0.0],
                [0.0, -0.75, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
            ],
            values: Vec::new(),
        };
        let b = blocks(&scan, &MODELS[1]);
        assert!(
            b.iter()
                .zip([4.0, 4.0, 0.6])
                .all(|(x, y)| (x - y).abs() < 1e-5),
            "{b:?}"
        );
        assert_eq!(blocks(&scan, &MODELS[0]), [1.0; 3]);
    }

    /// The built-in models against MONAI's own bundle code on a real scan.
    /// `AI_SERIES` is a DICOM series, `AI_REF` holds the weights (as
    /// spleen.pt, wholebody_3mm.pt, wholebody.pt) and `AI_BUNDLE_OUT` the
    /// bundles' label files (spleen/, wb3/, wb15/). Run with
    /// `cargo test --release models_match_monai -- --ignored --nocapture`.
    #[cfg(not(target_os = "macos"))]
    #[test]
    #[ignore]
    fn models_match_monai_on_a_real_scan() {
        use crate::scan::FrameRef;
        let var = |k: &str| PathBuf::from(std::env::var(k).unwrap());
        let mut paths: Vec<_> = std::fs::read_dir(var("AI_SERIES"))
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
        for (model, weights, reference) in [
            (&MODELS[0], "spleen.pt", "spleen"),
            (&MODELS[1], "wholebody_3mm.pt", "wb3"),
            (&MODELS[2], "wholebody.pt", "wb15"),
        ] {
            let bytes = std::fs::read(var("AI_REF").join(weights)).unwrap();
            let device = match std::env::var("AI_DEVICE").as_deref() {
                Ok("gpu") => Device::Gpu,
                _ => Device::Cpu,
            };
            if std::env::var("AI_MODELS").is_ok_and(|m| !m.contains(reference)) {
                continue;
            }
            let (mut engine, place) = engine(&model.build(&bytes).unwrap(), device).unwrap();
            let start = std::time::Instant::now();
            let mut blocks = 0;
            let got = segment(
                &scan,
                model,
                engine.as_mut(),
                &AtomicBool::new(false),
                &mut |_, t| blocks = t,
            )
            .unwrap();
            let seconds = start.elapsed().as_secs_f32();
            if let Ok(dump) = std::env::var("AI_DUMP") {
                let bytes: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(PathBuf::from(dump).join(format!("{reference}.u16")), bytes)
                    .unwrap();
            }
            let file = var("AI_BUNDLE_OUT")
                .join(reference)
                .join("viewer/viewer_trans.nii.gz");
            let want = crate::nifti::read_labels(&std::fs::read(file).unwrap()).unwrap();
            assert_eq!(want.dims, scan.dims);
            // Dice for each structure big enough to judge.
            let n = model.labels.len();
            let (mut both, mut a, mut b) = (vec![0usize; n], vec![0usize; n], vec![0usize; n]);
            for (&g, &w) in got.iter().zip(&want.voxels) {
                a[g as usize] += 1;
                b[w as usize] += 1;
                if g == w {
                    both[g as usize] += 1;
                }
            }
            let mut dice: Vec<(f64, &str)> = (1..n)
                .filter(|&l| b[l] >= 1000)
                .map(|l| (2.0 * both[l] as f64 / (a[l] + b[l]) as f64, model.labels[l]))
                .collect();
            dice.sort_by(|x, y| x.0.total_cmp(&y.0));
            let mean = dice.iter().map(|d| d.0).sum::<f64>() / dice.len() as f64;
            println!(
                "{} {place}: {blocks} blocks in {seconds:.1} s; {} structures, mean Dice {mean:.4}, lowest {:?}",
                model.name,
                dice.len(),
                &dice[..dice.len().min(3)]
            );
            assert!(mean > 0.99, "{}: mean Dice {mean}", model.name);
        }
    }
}

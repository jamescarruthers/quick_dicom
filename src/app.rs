use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use eframe::egui::{
    self, Align, Align2, Color32, FontId, Key, Layout, Modifiers, PointerButton, Pos2, Rect, Sense,
    Vec2,
};
use eframe::egui_wgpu;

use crate::colormap::Colormap;
use crate::decode::{Frame, Spacing};
use crate::export::{self, Export};
use crate::loader::{Loader, Slot};
use crate::render::{ImageCallback, ImageRenderer, Uniforms};
use crate::render3d::{VolumeCallback, VolumeRenderer, VolumeUniforms};
use crate::scan::{FrameRef, Library, ScanEvent, Scanner, Stack};
use crate::volume::{Volume, VolumeJob};

/// Common CT windows as (name, centre, width).
const CT_PRESETS: [(&str, f32, f32); 5] = [
    ("Soft tissue", 40.0, 400.0),
    ("Lung", -600.0, 1500.0),
    ("Bone", 400.0, 1800.0),
    ("Brain", 40.0, 80.0),
    ("Liver", 60.0, 160.0),
];

struct View {
    /// 1.0 fits the image to the viewport.
    zoom: f32,
    /// Offset of the image centre from the viewport centre, in points.
    pan: Vec2,
    /// Window centre and width; `None` until the first frame of a stack shows.
    window: Option<(f32, f32)>,
    invert: bool,
    smooth: bool,
    colormap: Colormap,
    /// Show millimetre rulers when the file gives a pixel size.
    rulers: bool,
}

impl Default for View {
    fn default() -> Self {
        View {
            zoom: 1.0,
            pan: Vec2::ZERO,
            window: None,
            invert: false,
            smooth: true,
            colormap: Colormap::Grey,
            rulers: true,
        }
    }
}

/// How the 3D view combines its slices.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Combine {
    /// Each slice covers those behind it, more where it is brighter.
    Blend,
    /// Each screen pixel shows the brightest value along its line of sight.
    Maximum,
}

const DEFAULT_YAW: f32 = 0.6;
const DEFAULT_PITCH: f32 = -0.45;
/// Seconds the turn from flat slice to 3D stack takes.
const TRANSITION_SECONDS: f32 = 0.9;

struct ThreeD {
    /// The user wants the 3D view.
    on: bool,
    /// How far the turn into 3D has got: 0 is the flat slice, 1 the stack.
    t: f32,
    clock: Option<Instant>,
    yaw: f32,
    pitch: f32,
    /// Multiplies the true gap between slices.
    depth: f32,
    /// How quickly bright voxels become opaque, summed over the stack.
    density: f32,
    combine: Combine,
}

impl Default for ThreeD {
    fn default() -> Self {
        ThreeD {
            on: false,
            t: 0.0,
            clock: None,
            yaw: DEFAULT_YAW,
            pitch: DEFAULT_PITCH,
            depth: 1.0,
            density: 8.0,
            combine: Combine::Blend,
        }
    }
}

pub struct ViewerApp {
    loader: Loader,
    scanner: Option<Scanner>,
    scanning: bool,
    status: String,
    root: Option<PathBuf>,
    library: Arc<Library>,
    selected: Option<usize>,
    selected_key: Option<String>,
    index: usize,
    /// A file passed on the command line; select it once the scan finds it.
    pending_file: Option<PathBuf>,
    /// The last decoded frame shown, kept on screen while the next decodes.
    shown: Option<(String, Arc<Frame>)>,
    view: View,
    playing: bool,
    fps: f32,
    play_clock: Option<Instant>,
    play_accum: f32,
    wheel_accum: f32,
    requested: Option<(String, usize, usize)>,
    filter: String,
    scroll_to_selected: bool,
    /// A video being saved, if any.
    export: Option<Export>,
    /// Where the last video went, to offer the same folder next time.
    export_dir: Option<PathBuf>,
    three_d: ThreeD,
    volume: Option<Arc<Volume>>,
    volume_job: Option<VolumeJob>,
    /// Most slices the GPU can hold in one texture array.
    max_layers: u32,
}

impl ViewerApp {
    pub fn new(cc: &eframe::CreationContext, path: Option<PathBuf>) -> Self {
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        let rs = cc
            .wgpu_render_state
            .as_ref()
            .expect("the wgpu renderer is required");
        ImageRenderer::install(rs);
        VolumeRenderer::install(rs);
        let max_dim = rs.device.limits().max_texture_dimension_2d;
        let max_layers = rs.device.limits().max_texture_array_layers.min(256);

        let budget_mb = std::env::var("QUICK_DICOM_CACHE_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2048);

        let mut app = ViewerApp {
            loader: Loader::new(cc.egui_ctx.clone(), max_dim, budget_mb << 20),
            scanner: None,
            scanning: false,
            status: String::new(),
            root: None,
            library: Arc::default(),
            selected: None,
            selected_key: None,
            index: 0,
            pending_file: None,
            shown: None,
            view: View::default(),
            playing: false,
            fps: 15.0,
            play_clock: None,
            play_accum: 0.0,
            wheel_accum: 0.0,
            requested: None,
            filter: String::new(),
            scroll_to_selected: false,
            export: None,
            export_dir: None,
            three_d: ThreeD::default(),
            volume: None,
            volume_job: None,
            max_layers,
        };
        if let Some(p) = path {
            app.open(p, &cc.egui_ctx);
        }
        app
    }

    fn open(&mut self, path: PathBuf, ctx: &egui::Context) {
        let path = path.canonicalize().unwrap_or(path);
        let (root, file) = if path.is_file() {
            let parent = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
            (parent, Some(path))
        } else {
            (path, None)
        };
        self.scanner = Some(Scanner::start(root.clone(), ctx.clone()));
        self.scanning = true;
        self.status = "Scanning…".into();
        self.library = Arc::default();
        self.selected = None;
        self.selected_key = None;
        self.shown = None;
        self.requested = None;
        self.pending_file = file;
        self.loader.clear();
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(format!(
            "Quick DICOM — {}",
            root.display()
        )));
        self.root = Some(root);
    }

    fn pick_folder(&mut self, ctx: &egui::Context) {
        let mut dialog = rfd::FileDialog::new().set_title("Open a folder of DICOM files");
        if let Some(root) = &self.root {
            dialog = dialog.set_directory(root);
        }
        if let Some(dir) = dialog.pick_folder() {
            self.open(dir, ctx);
        }
    }

    /// Ask where to save, then write the current stack as an MP4 in the
    /// background with the current window, inversion and frame rate.
    fn save_video(&mut self, ctx: &egui::Context) {
        let Some(stack) = self.stack().cloned() else {
            return;
        };
        if self.export.is_some() || stack.frames.len() < 2 {
            return;
        }
        let mut dialog = rfd::FileDialog::new()
            .set_title("Save the series as a video")
            .add_filter("MP4 video", &["mp4"])
            .set_file_name(format!("{}.mp4", file_stem(&stack.label)));
        if let Some(dir) = &self.export_dir {
            dialog = dialog.set_directory(dir);
        }
        let Some(mut path) = dialog.save_file() else {
            return;
        };
        if !path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("mp4"))
        {
            path.as_mut_os_string().push(".mp4");
        }
        self.export_dir = path.parent().map(Path::to_path_buf);
        let settings = export::Settings {
            frames: stack.frames.clone(),
            fps: self.fps,
            window: self.view.window,
            invert: self.view.invert,
            smooth: self.view.smooth,
            colormap: self.view.colormap,
        };
        self.export = Some(Export::start(settings, path, ctx.clone()));
    }

    fn poll_export(&mut self) {
        let Some(result) = self.export.as_ref().and_then(Export::poll) else {
            return;
        };
        let export = self.export.take().expect("polled an export");
        let name = export
            .path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        self.status = match result {
            Ok(s) => {
                let mut msg = format!(
                    "Saved {name}: {} frames, {:.1} MB",
                    s.frames,
                    s.bytes as f64 / 1e6
                );
                if s.failed > 0 {
                    msg.push_str(&format!(", {} unreadable frames left black", s.failed));
                }
                msg
            }
            Err(e) if e == "cancelled" => "Video not saved".into(),
            Err(e) => format!("Could not save the video: {e}"),
        };
    }

    fn stack(&self) -> Option<&Arc<Stack>> {
        self.selected.and_then(|i| self.library.stacks.get(i))
    }

    fn select(&mut self, i: usize) {
        let Some(stack) = self.library.stacks.get(i).cloned() else {
            return;
        };
        self.selected = Some(i);
        self.selected_key = Some(stack.key.clone());
        // Start in the middle of a volume, where the anatomy usually is,
        // and at the start of a cine loop.
        self.index = if stack.frames.iter().all(|f| !f.multi) {
            stack.frames.len() / 2
        } else {
            0
        };
        self.view = View {
            smooth: self.view.smooth,
            colormap: self.view.colormap,
            rulers: self.view.rulers,
            ..View::default()
        };
        self.playing = false;
        self.fps = stack.fps.unwrap_or(15.0);
        self.scroll_to_selected = true;
    }

    fn set_library(&mut self, lib: Arc<Library>) {
        let current: Option<FrameRef> =
            self.stack().and_then(|s| s.frames.get(self.index)).cloned();
        self.library = lib;

        if let Some(path) = &self.pending_file
            && let Some((s, f)) = self.library.find_path(path)
        {
            self.pending_file = None;
            self.select(s);
            self.index = f;
            return;
        }

        if let Some(key) = self.selected_key.clone() {
            match self.library.find_stack(&key) {
                Some(i) => {
                    self.selected = Some(i);
                    let frames = &self.library.stacks[i].frames;
                    if let Some(j) = current.and_then(|c| frames.iter().position(|f| *f == c)) {
                        self.index = j;
                    }
                    self.index = self.index.min(frames.len().saturating_sub(1));
                }
                None => {
                    self.selected = None;
                    self.selected_key = None;
                }
            }
        }
        if self.selected.is_none() && self.pending_file.is_none() && !self.library.stacks.is_empty()
        {
            self.select(0);
        }
    }

    fn poll_scan(&mut self) {
        let Some(scanner) = &self.scanner else {
            return;
        };
        let events: Vec<ScanEvent> = scanner.events.try_iter().collect();
        for ev in events {
            match ev {
                ScanEvent::Progress { files_seen, images } => {
                    self.status = format!("Scanning… {files_seen} files, {images} images");
                }
                ScanEvent::Library(lib) => self.set_library(lib),
                ScanEvent::Done {
                    files_seen,
                    images,
                    elapsed,
                } => {
                    self.scanning = false;
                    self.status = format!(
                        "{images} images in {} stacks · {files_seen} files read in {:.1} s",
                        self.library.stacks.len(),
                        elapsed.as_secs_f32()
                    );
                    if self.pending_file.take().is_some() && self.selected.is_none() {
                        self.status.push_str(" · that file has no viewable image");
                        if !self.library.stacks.is_empty() {
                            self.select(0);
                        }
                    }
                }
            }
        }
    }

    fn step(&mut self, delta: i64) {
        if let Some(n) = self.stack().map(|s| s.frames.len()) {
            self.index = (self.index as i64 + delta).clamp(0, n as i64 - 1) as usize;
        }
    }

    fn step_stack(&mut self, delta: i64) {
        let n = self.library.stacks.len() as i64;
        if n == 0 {
            return;
        }
        let i = match self.selected {
            Some(i) => (i as i64 + delta).clamp(0, n - 1),
            None => 0,
        } as usize;
        if Some(i) != self.selected {
            self.select(i);
        }
    }

    fn reset_view(&mut self) {
        self.view.zoom = 1.0;
        self.view.pan = Vec2::ZERO;
        self.view.window = None;
        self.view.invert = false;
        self.three_d.yaw = DEFAULT_YAW;
        self.three_d.pitch = DEFAULT_PITCH;
    }

    fn can_3d(&self) -> bool {
        self.stack().is_some_and(|s| s.frames.len() >= 2)
    }

    /// Start or finish building the volume, and move the 3D transition on.
    fn tick_3d(&mut self, ctx: &egui::Context) {
        if let Some(result) = self.volume_job.as_ref().and_then(VolumeJob::poll) {
            self.volume_job = None;
            match result {
                Ok(volume) => {
                    if volume.failed > 0 {
                        self.status = format!(
                            "3D view: {} slices could not be read and are empty",
                            volume.failed
                        );
                    }
                    self.volume = Some(Arc::new(volume));
                }
                Err(e) if e == "cancelled" => {}
                Err(e) => {
                    self.status = format!("Could not build the 3D view: {e}");
                    self.three_d.on = false;
                }
            }
        }

        let key = self.stack().map(|s| s.key.clone());
        let ready = self
            .volume
            .as_ref()
            .is_some_and(|v| Some(&v.key) == key.as_ref());
        if !ready {
            // A volume of another stack must not be drawn for this one.
            self.three_d.t = 0.0;
            self.three_d.clock = None;
            if self.volume.is_some() && key.is_some() {
                self.volume = None;
            }
        }
        if self.three_d.on && !self.can_3d() {
            self.three_d.on = false;
        }
        if self.three_d.on && !ready {
            let building = self
                .volume_job
                .as_ref()
                .is_some_and(|j| Some(&j.key) == key.as_ref());
            if !building && let Some(stack) = self.stack().cloned() {
                self.volume_job = Some(VolumeJob::start(&stack, self.max_layers, ctx.clone()));
            }
        }
        if !self.three_d.on {
            self.volume_job = None;
        }

        let target = if self.three_d.on && ready { 1.0 } else { 0.0 };
        let now = Instant::now();
        if self.three_d.t != target {
            let dt = self
                .three_d
                .clock
                .map_or(0.0, |c| (now - c).as_secs_f32().min(0.1));
            let step = dt / TRANSITION_SECONDS;
            self.three_d.t = if target > self.three_d.t {
                (self.three_d.t + step).min(target)
            } else {
                (self.three_d.t - step).max(target)
            };
            self.three_d.clock = Some(now);
            ctx.request_repaint();
        } else {
            self.three_d.clock = None;
        }
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::O)) {
            self.pick_folder(ctx);
        }
        if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::S)) {
            self.save_video(ctx);
        }
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let n = self.stack().map_or(0, |s| s.frames.len());
        // Consume the keys so focused widgets do not also act on them.
        let mut pressed = |key: Key| {
            ctx.input_mut(|i| {
                let mut count = 0;
                while i.consume_key(Modifiers::NONE, key) {
                    count += 1;
                }
                count
            })
        };
        let forward = pressed(Key::ArrowRight) + pressed(Key::ArrowDown);
        let back = pressed(Key::ArrowLeft) + pressed(Key::ArrowUp);
        let home = pressed(Key::Home);
        let end = pressed(Key::End);
        let next_stack = pressed(Key::PageDown);
        let prev_stack = pressed(Key::PageUp);
        let play = pressed(Key::Space);
        let reset = pressed(Key::R);
        let fit = pressed(Key::F);
        let invert = pressed(Key::I);
        let smooth = pressed(Key::S);
        let colours = pressed(Key::C);
        let rulers = pressed(Key::M);
        let three_d = pressed(Key::D);
        let presets = [Key::Num1, Key::Num2, Key::Num3, Key::Num4, Key::Num5].map(&mut pressed);

        self.step(forward as i64 - back as i64);
        if home > 0 {
            self.index = 0;
        }
        if end > 0 {
            self.index = n.saturating_sub(1);
        }
        self.step_stack(next_stack as i64 - prev_stack as i64);
        if play % 2 == 1 && n > 1 {
            self.playing = !self.playing;
        }
        if reset > 0 {
            self.reset_view();
        }
        if fit > 0 {
            self.view.zoom = 1.0;
            self.view.pan = Vec2::ZERO;
        }
        if invert % 2 == 1 {
            self.view.invert = !self.view.invert;
        }
        if smooth % 2 == 1 {
            self.view.smooth = !self.view.smooth;
        }
        for _ in 0..colours {
            self.view.colormap = self.view.colormap.next();
        }
        if rulers % 2 == 1 {
            self.view.rulers = !self.view.rulers;
        }
        if three_d % 2 == 1 && self.can_3d() {
            self.three_d.on = !self.three_d.on;
        }
        for (p, (_, c, w)) in presets.iter().zip(CT_PRESETS) {
            if *p > 0 {
                self.view.window = Some((c, w));
            }
        }
    }

    fn tick_playback(&mut self, ctx: &egui::Context) {
        let Some(stack) = self.stack().cloned() else {
            self.playing = false;
            return;
        };
        if !self.playing || stack.frames.len() < 2 {
            self.playing = false;
            self.play_clock = None;
            self.play_accum = 0.0;
            return;
        }
        let now = Instant::now();
        if let Some(t) = self.play_clock {
            self.play_accum += (now - t).as_secs_f32();
        }
        self.play_clock = Some(now);
        let period = 1.0 / self.fps.max(0.5);
        while self.play_accum >= period {
            let next = (self.index + 1) % stack.frames.len();
            // Hold on the current frame until the next one is decoded.
            if self.loader.get(&stack.frames[next]).is_none() {
                self.play_accum = period;
                break;
            }
            self.index = next;
            self.play_accum -= period;
        }
        ctx.request_repaint();
    }

    /// Ask the loader for the current frame first, then outwards from it,
    /// as far as the cache budget allows.
    fn update_requests(&mut self) {
        let Some(stack) = self.stack().cloned() else {
            return;
        };
        let n = stack.frames.len();
        if n == 0 {
            return;
        }
        let sig = (stack.key.clone(), self.index, n);
        if self.requested.as_ref() == Some(&sig) {
            return;
        }
        self.requested = Some(sig);

        let per_frame = (stack.max_rows as usize * stack.max_cols as usize * 4).max(1);
        let limit = (self.loader.budget() / 4 * 3 / per_frame).clamp(1, n);
        let i = self.index;
        let mut wanted = Vec::with_capacity(limit);
        wanted.push(stack.frames[i].clone());
        let mut d = 1;
        while wanted.len() < limit && (i + d < n || i >= d) {
            // While playing, look further ahead than behind.
            if i + d < n {
                wanted.push(stack.frames[i + d].clone());
            } else if self.playing {
                wanted.push(stack.frames[(i + d) % n].clone());
            }
            if wanted.len() < limit && i >= d && (!self.playing || d % 2 == 0) {
                wanted.push(stack.frames[i - d].clone());
            }
            d += 1;
        }
        wanted.dedup();
        self.loader.request(&wanted);
    }

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .button("📂 Open folder…")
                .on_hover_text("Ctrl+O, or drop a folder on the window")
                .clicked()
            {
                self.pick_folder(ui.ctx());
            }
            ui.separator();
            ui.toggle_value(&mut self.view.invert, "Invert")
                .on_hover_text("I");
            ui.toggle_value(&mut self.view.smooth, "Smooth")
                .on_hover_text("S: bilinear filtering on or off");
            let is_ct = self.stack().is_some_and(|s| s.modality == "CT");
            ui.add_enabled_ui(is_ct, |ui| {
                egui::ComboBox::from_id_salt("preset")
                    .selected_text("CT window")
                    .show_ui(ui, |ui| {
                        for (k, (name, c, w)) in CT_PRESETS.iter().enumerate() {
                            if ui
                                .selectable_label(false, format!("{}  {name}  ({c} / {w})", k + 1))
                                .clicked()
                            {
                                self.view.window = Some((*c, *w));
                            }
                        }
                    });
            });
            egui::ComboBox::from_id_salt("colours")
                .selected_text(self.view.colormap.name())
                .show_ui(ui, |ui| {
                    for map in Colormap::ALL {
                        ui.selectable_value(&mut self.view.colormap, map, map.name());
                    }
                })
                .response
                .on_hover_text("C: colour map for greyscale images");
            ui.toggle_value(&mut self.view.rulers, "Rulers")
                .on_hover_text("M: millimetre rulers, when the file gives a pixel size");
            if ui
                .button("Reset")
                .on_hover_text("R, or double-click the image")
                .clicked()
            {
                self.reset_view();
            }
            ui.separator();
            let can_3d = self.can_3d();
            ui.add_enabled_ui(can_3d, |ui| {
                ui.toggle_value(&mut self.three_d.on, "3D").on_hover_text(
                    "D: stack the slices in 3D. Drag to turn, Ctrl+drag for \
                     window and level, wheel to zoom.",
                );
            });
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(&self.status);
                if self.scanning {
                    ui.spinner();
                }
            });
        });
    }

    fn bottom_bar(&mut self, ui: &mut egui::Ui) {
        let Some(stack) = self.stack().cloned() else {
            ui.add_space(4.0);
            ui.weak("No image selected");
            ui.add_space(4.0);
            return;
        };
        let n = stack.frames.len();
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let icon = if self.playing { "⏸" } else { "▶" };
            if ui
                .add_enabled(
                    n > 1,
                    egui::Button::new(icon).min_size(Vec2::new(28.0, 0.0)),
                )
                .on_hover_text("Space")
                .clicked()
            {
                self.playing = !self.playing;
            }
            ui.add(
                egui::DragValue::new(&mut self.fps)
                    .range(1.0..=240.0)
                    .speed(0.25)
                    .max_decimals(1)
                    .suffix(" fps"),
            );
            match &self.export {
                Some(e) => {
                    let (done, total) = (e.done(), e.total.max(1));
                    ui.add(
                        egui::ProgressBar::new(done as f32 / total as f32)
                            .desired_width(150.0)
                            .text(format!("Saving {done} / {total}")),
                    );
                    if ui.button("Cancel").clicked() {
                        e.cancel();
                    }
                }
                None => {
                    if ui
                        .add_enabled(n > 1, egui::Button::new("💾 Save MP4…"))
                        .on_hover_text(
                            "Ctrl+S: save this series as a video, \
                             with the current window and frame rate",
                        )
                        .clicked()
                    {
                        self.save_video(&ui.ctx().clone());
                    }
                }
            }
            if self.three_d.on {
                ui.separator();
                egui::ComboBox::from_id_salt("combine")
                    .selected_text(match self.three_d.combine {
                        Combine::Blend => "Blend",
                        Combine::Maximum => "MIP",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.three_d.combine, Combine::Blend, "Blend")
                            .on_hover_text("Bright parts are solid, dark parts clear");
                        ui.selectable_value(&mut self.three_d.combine, Combine::Maximum, "MIP")
                            .on_hover_text("Maximum intensity projection: the brightest value along each line of sight");
                    });
                ui.add_enabled(
                    self.three_d.combine == Combine::Blend,
                    egui::DragValue::new(&mut self.three_d.density)
                        .range(0.5..=200.0)
                        .speed(0.1)
                        .max_decimals(1)
                        .prefix("Opacity "),
                )
                .on_hover_text("How quickly bright parts block the view");
                ui.add(
                    egui::DragValue::new(&mut self.three_d.depth)
                        .range(0.1..=10.0)
                        .speed(0.02)
                        .max_decimals(2)
                        .prefix("Depth ×"),
                )
                .on_hover_text("Stretch or squash the gap between slices");
                ui.separator();
            }
            let digits = n.to_string().len();
            ui.monospace(format!("{:>digits$} / {n}", self.index + 1));

            let width = ui.available_width().max(50.0);
            ui.vertical(|ui| {
                ui.spacing_mut().slider_width = width;
                let mut idx = self.index;
                let slider = egui::Slider::new(&mut idx, 0..=n.saturating_sub(1))
                    .show_value(false)
                    .smart_aim(false);
                if ui.add_enabled(n > 1, slider).changed() {
                    self.index = idx;
                }
                // Which frames are already decoded.
                let (bar, _) = ui.allocate_exact_size(Vec2::new(width, 3.0), Sense::hover());
                let mask = self.loader.cached_mask(&stack.frames);
                let painter = ui.painter();
                painter.rect_filled(bar, 0.0, Color32::from_gray(45));
                let cell = bar.width() / n as f32;
                let mut i = 0;
                while i < n {
                    if mask[i] {
                        let start = i;
                        while i < n && mask[i] {
                            i += 1;
                        }
                        let r = Rect::from_min_max(
                            Pos2::new(bar.left() + start as f32 * cell, bar.top()),
                            Pos2::new(bar.left() + i as f32 * cell, bar.bottom()),
                        );
                        painter.rect_filled(r, 0.0, Color32::from_rgb(70, 130, 180));
                    }
                    i += 1;
                }
            });
        });
        ui.add_space(2.0);
    }

    fn browser(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.add(
            egui::TextEdit::singleline(&mut self.filter)
                .hint_text("Filter series…")
                .desired_width(f32::INFINITY),
        );
        ui.add_space(4.0);
        let lib = self.library.clone();
        let filter = self.filter.trim().to_lowercase();
        let matches = |s: &Stack| {
            filter.is_empty()
                || s.label.to_lowercase().contains(&filter)
                || s.study.to_lowercase().contains(&filter)
                || s.patient.to_lowercase().contains(&filter)
        };
        let mut clicked = None;
        let scroll_to = std::mem::take(&mut self.scroll_to_selected);

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if lib.stacks.is_empty() {
                    ui.weak(if self.scanning {
                        "Looking for images…"
                    } else {
                        "No images yet. Open a folder to start."
                    });
                }
                let open_patients = lib.patients.len() <= 8;
                for (pi, p) in lib.patients.iter().enumerate() {
                    let any = p
                        .studies
                        .iter()
                        .any(|s| s.stacks.clone().any(|i| matches(&lib.stacks[i])));
                    if !any {
                        continue;
                    }
                    let holds_selection = self
                        .selected
                        .is_some_and(|sel| p.studies.iter().any(|s| s.stacks.contains(&sel)));
                    let force_open =
                        (!filter.is_empty() || (scroll_to && holds_selection)).then_some(true);
                    egui::CollapsingHeader::new(egui::RichText::new(&p.label).strong())
                        .id_salt(("patient", pi, &p.label))
                        .default_open(open_patients)
                        .open(force_open)
                        .show(ui, |ui| {
                            for (si, s) in p.studies.iter().enumerate() {
                                if !s.stacks.clone().any(|i| matches(&lib.stacks[i])) {
                                    continue;
                                }
                                egui::CollapsingHeader::new(&s.label)
                                    .id_salt(("study", pi, si, &s.label))
                                    .default_open(true)
                                    .open(force_open)
                                    .show(ui, |ui| {
                                        for i in s.stacks.clone() {
                                            let st = &lib.stacks[i];
                                            if !matches(st) {
                                                continue;
                                            }
                                            let sel = self.selected == Some(i);
                                            let r = ui.selectable_label(sel, &st.label);
                                            if r.clicked() {
                                                clicked = Some(i);
                                            }
                                            if sel && scroll_to {
                                                r.scroll_to_me(None);
                                            }
                                        }
                                    });
                            }
                        });
                }
            });

        if let Some(i) = clicked
            && Some(i) != self.selected
        {
            self.select(i);
            self.scroll_to_selected = false;
        }
    }

    fn viewer(&mut self, ui: &mut egui::Ui) {
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::BLACK);

        let text_color = Color32::from_gray(210);
        let font = FontId::proportional(13.0);
        let hint = |text: &str| {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                text,
                FontId::proportional(16.0),
                Color32::from_gray(150),
            );
        };

        if ui.ctx().input(|i| !i.raw.hovered_files.is_empty()) {
            painter.rect_filled(rect, 0.0, Color32::from_rgba_unmultiplied(70, 130, 180, 60));
            hint("Drop to open");
            return;
        }

        let Some(stack) = self.stack().cloned() else {
            if self.scanning {
                hint("Scanning…");
            } else {
                hint("Open a folder (Ctrl+O), drop one here, or pass a path on the command line.");
            }
            return;
        };
        let n = stack.frames.len();
        self.index = self.index.min(n.saturating_sub(1));

        let mut error = None;
        match self.loader.get(&stack.frames[self.index]) {
            Some(Slot::Ready(f)) => self.shown = Some((stack.key.clone(), f)),
            Some(Slot::Failed(e)) => {
                self.shown = None;
                error = Some(e);
            }
            None => {}
        }
        let frame = self
            .shown
            .as_ref()
            .filter(|(k, _)| *k == stack.key)
            .map(|(_, f)| f.clone());

        let volume = self
            .volume
            .clone()
            .filter(|v| v.key == stack.key && self.three_d.t > 0.0);
        if let Some(volume) = volume {
            let range = frame.as_ref().map_or(volume.window.1, |f| f.max - f.min);
            self.interact_3d(ui, &resp, rect, range);
            let info = self.draw_3d(&painter, rect, &volume, ui.ctx().pixels_per_point());
            self.overlay_text(&painter, rect, &stack, Some(info), None, font, text_color);
            return;
        }
        self.interact_2d(ui, &resp, rect, frame.as_deref());

        let Some(frame) = frame else {
            match error {
                Some(e) => hint(&format!("Cannot show this frame: {e}")),
                None => hint("Loading…"),
            }
            self.overlay_text(&painter, rect, &stack, None, None, font, text_color);
            return;
        };

        if self.view.window.is_none() {
            self.view.window = Some(frame.window);
        }
        let (wc, ww) = self.view.window.unwrap_or(frame.window);

        let (w, h) = (frame.width as f32, frame.height as f32);
        let fit = (rect.width() / w).min(rect.height() / h);
        let scale = fit * self.view.zoom;
        let img = Rect::from_center_size(rect.center() + self.view.pan, Vec2::new(w, h) * scale);
        let ndc = |p: Pos2| {
            [
                (p.x - rect.left()) / rect.width() * 2.0 - 1.0,
                1.0 - (p.y - rect.top()) / rect.height() * 2.0,
            ]
        };
        let (a, b) = (ndc(img.min), ndc(img.max));
        let color = frame.is_color();
        let uniforms = Uniforms {
            rect: [a[0], a[1], b[0], b[1]],
            tex: [w, h, 0.0, 0.0],
            wl: [
                wc,
                ww,
                if color { 255.0 } else { 1.0 },
                (frame.inverted != self.view.invert) as u8 as f32,
            ],
            mode: [
                color as u8 as f32,
                self.view.smooth as u8 as f32,
                0.0,
                self.view.colormap.shader_id(),
            ],
        };
        painter.add(egui_wgpu::Callback::new_paint_callback(
            rect,
            ImageCallback {
                frame: frame.clone(),
                uniforms,
            },
        ));

        if self.view.rulers
            && let Some(spacing) = frame.spacing
        {
            draw_rulers(&painter, rect, spacing, scale);
        }
        if self.three_d.on
            && let Some(job) = &self.volume_job
        {
            painter.text(
                rect.center_top() + Vec2::new(0.0, 48.0),
                Align2::CENTER_TOP,
                format!("Building the 3D view… {} / {}", job.done(), job.total),
                FontId::proportional(15.0),
                Color32::from_gray(230),
            );
        }

        let probe = resp.hover_pos().and_then(|p| {
            let x = ((p.x - img.left()) / scale).floor();
            let y = ((p.y - img.top()) / scale).floor();
            (x >= 0.0 && y >= 0.0)
                .then(|| {
                    frame
                        .value_at(x as u32, y as u32)
                        .map(|v| format!("({x}, {y})  {v}"))
                })
                .flatten()
        });
        let mut info = format!("{} × {}", frame.width, frame.height);
        if let Some(sp) = frame.spacing {
            info.push_str(&format!("\nPixel {}", pixel_size(sp)));
        }
        info.push_str(&format!(
            "\nW {:.0}  L {:.0}\nZoom {:.0}%",
            ww,
            wc,
            scale * ui.ctx().pixels_per_point() * 100.0
        ));
        self.overlay_text(&painter, rect, &stack, Some(info), probe, font, text_color);
    }

    /// Wheel scrubs, Ctrl/Cmd + wheel or pinch zooms, left-drag sets the
    /// window, right/middle/Shift-drag pans.
    fn interact_2d(
        &mut self,
        ui: &egui::Ui,
        resp: &egui::Response,
        rect: Rect,
        frame: Option<&Frame>,
    ) {
        if resp.hovered() {
            let (zoom, lines) = ui.ctx().input(|i| (i.zoom_delta(), wheel_lines(i)));
            self.wheel_accum -= lines;
            let steps = self.wheel_accum.trunc();
            if steps != 0.0 {
                self.wheel_accum -= steps;
                self.step(steps as i64);
            }
            self.zoom_about(resp, rect, zoom);
        }

        let shift = ui.ctx().input(|i| i.modifiers.shift);
        if resp.dragged_by(PointerButton::Secondary)
            || resp.dragged_by(PointerButton::Middle)
            || (resp.dragged_by(PointerButton::Primary) && shift)
        {
            self.view.pan += resp.drag_delta();
        } else if resp.dragged_by(PointerButton::Primary)
            && let Some(f) = frame
        {
            self.adjust_window(f.max - f.min, f.is_color(), resp.drag_delta());
        }
        if resp.double_clicked() {
            self.reset_view();
        }
    }

    /// Left-drag turns the stack, Ctrl/Cmd-drag sets the window, wheel
    /// zooms, right/middle/Shift-drag pans.
    fn interact_3d(&mut self, ui: &egui::Ui, resp: &egui::Response, rect: Rect, range: f32) {
        if resp.hovered() {
            let (zoom, lines) = ui.ctx().input(|i| (i.zoom_delta(), wheel_lines(i)));
            self.zoom_about(resp, rect, zoom * 1.1f32.powf(lines));
        }
        let m = ui.ctx().input(|i| i.modifiers);
        let d = resp.drag_delta();
        if resp.dragged_by(PointerButton::Secondary)
            || resp.dragged_by(PointerButton::Middle)
            || (resp.dragged_by(PointerButton::Primary) && m.shift)
        {
            self.view.pan += d;
        } else if resp.dragged_by(PointerButton::Primary) && (m.command || m.ctrl) {
            self.adjust_window(range, false, d);
        } else if resp.dragged_by(PointerButton::Primary) {
            self.three_d.yaw += d.x * 0.01;
            self.three_d.pitch = (self.three_d.pitch - d.y * 0.01).clamp(-1.55, 1.55);
        }
        if resp.double_clicked() {
            self.reset_view();
        }
    }

    /// Zoom by `factor`, keeping the point under the pointer still.
    fn zoom_about(&mut self, resp: &egui::Response, rect: Rect, factor: f32) {
        if factor == 1.0 {
            return;
        }
        if let Some(p) = resp.hover_pos() {
            let new_zoom = (self.view.zoom * factor).clamp(0.05, 100.0);
            let f = new_zoom / self.view.zoom;
            let center = rect.center() + self.view.pan;
            self.view.pan = (p - (p - center) * f) - rect.center();
            self.view.zoom = new_zoom;
        }
    }

    /// Drag right for a wider window, down for a higher level.
    fn adjust_window(&mut self, range: f32, colour: bool, d: Vec2) {
        if let Some((c, w)) = self.view.window {
            let speed = (range.max(1.0) / 600.0).max(if colour { 0.5 } else { 0.01 });
            self.view.window = Some((c + d.y * speed, (w + d.x * speed).max(speed)));
        }
    }

    /// Queue the 3D view for drawing; returns the text for the info corner.
    fn draw_3d(
        &mut self,
        painter: &egui::Painter,
        rect: Rect,
        volume: &Arc<Volume>,
        ppp: f32,
    ) -> String {
        let t = smoothstep(self.three_d.t);
        let (w, h, d) = (
            volume.width as f32,
            volume.height as f32,
            volume.depth as f32,
        );
        // Shrink a little as the stack turns, so its corners stay in view.
        let fit = (rect.width() / w).min(rect.height() / h);
        let s = fit * self.view.zoom * (1.0 - 0.3 * t);
        let centre = rect.center() + self.view.pan;
        let r = rotation(self.three_d.yaw * t, self.three_d.pitch * t);
        // Without positions in the files, make the stack half as deep as wide.
        let gap = volume.slice_gap.unwrap_or(w.max(h) * 0.5 / d.max(1.0)) * self.three_d.depth;
        let current = ((self.index / volume.stride) as f32).min(d - 1.0);
        // Turn about the current slice at first, the middle of the stack at the end.
        let pivot = current + ((d - 1.0) * 0.5 - current) * t;
        let dz = gap * t;
        let (kx, ky) = (2.0 * s / rect.width(), 2.0 * s / rect.height());
        let mx = [
            kx * r[0][0],
            kx * r[0][1],
            kx * r[0][2],
            2.0 * (centre.x - rect.left()) / rect.width() - 1.0,
        ];
        let my = [
            -ky * r[1][0],
            -ky * r[1][1],
            -ky * r[1][2],
            1.0 - 2.0 * (centre.y - rect.top()) / rect.height(),
        ];
        // Screen depth grows away from the viewer: draw the far end first.
        let reverse = r[2][2] * dz > 0.0;
        if self.view.window.is_none() {
            self.view.window = Some(volume.window);
        }
        let (wc, ww) = self.view.window.unwrap_or(volume.window);
        let invert = volume.inverted != self.view.invert;
        let maximum = self.three_d.combine == Combine::Maximum;
        let uniforms = VolumeUniforms {
            mx,
            my,
            dims: [w, h, d, current],
            slices: [dz, pivot, t, reverse as u8 as f32],
            wl: [wc, ww, invert as u8 as f32, self.view.colormap.shader_id()],
            look: [self.three_d.density, maximum as u8 as f32, 0.0, 0.0],
        };
        let size_px = (
            (rect.width() * ppp).round() as u32,
            (rect.height() * ppp).round() as u32,
        );
        painter.add(egui_wgpu::Callback::new_paint_callback(
            rect,
            VolumeCallback {
                volume: volume.clone(),
                uniforms,
                size_px,
                maximum_intensity: maximum,
            },
        ));

        let mut info = format!("3D · {} slices", volume.depth);
        if volume.stride > 1 {
            info.push_str(&format!(" (every {})", ordinal(volume.stride)));
        }
        info.push_str(&format!(
            "\n{}\nW {:.0}  L {:.0}",
            if maximum {
                "Maximum intensity"
            } else {
                "Blend"
            },
            ww,
            wc
        ));
        info
    }

    #[allow(clippy::too_many_arguments)]
    fn overlay_text(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        stack: &Stack,
        info: Option<String>,
        probe: Option<String>,
        font: FontId,
        color: Color32,
    ) {
        let m = 8.0;
        let put = |pos: Pos2, align: Align2, text: &str| {
            let shadow = Color32::from_black_alpha(220);
            painter.text(pos + Vec2::splat(1.0), align, text, font.clone(), shadow);
            painter.text(pos, align, text, font.clone(), color);
        };
        put(
            rect.left_top() + Vec2::new(m, m),
            Align2::LEFT_TOP,
            &format!("{}\n{}", stack.patient, stack.study),
        );
        put(
            rect.right_top() + Vec2::new(-m, m),
            Align2::RIGHT_TOP,
            &stack.label,
        );
        let mut bottom_left = format!("Frame {} / {}", self.index + 1, stack.frames.len());
        if let Some(p) = probe {
            bottom_left = format!("{p}\n{bottom_left}");
        }
        put(
            rect.left_bottom() + Vec2::new(m, -m),
            Align2::LEFT_BOTTOM,
            &bottom_left,
        );
        if let Some(info) = info {
            put(
                rect.right_bottom() + Vec2::new(-m, -m),
                Align2::RIGHT_BOTTOM,
                &info,
            );
        }
    }
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_scan();
        self.poll_export();
        if let Some(path) = ctx.input(|i| {
            i.raw.dropped_files.iter().find_map(|f| {
                let p = f.path();
                (!p.as_os_str().is_empty()).then(|| p.to_path_buf())
            })
        }) {
            self.open(path, &ctx);
        }
        self.handle_keys(&ctx);
        self.tick_3d(&ctx);
        self.tick_playback(&ctx);
        self.update_requests();

        egui::Panel::top("toolbar").show(ui, |ui| {
            ui.add_space(3.0);
            self.top_bar(ui);
            ui.add_space(1.0);
        });
        egui::Panel::bottom("transport").show(ui, |ui| self.bottom_bar(ui));
        egui::Panel::left("browser")
            .resizable(true)
            .default_size(320.0)
            .min_size(160.0)
            .show(ui, |ui| self.browser(ui));
        egui::CentralPanel::no_frame().show(ui, |ui| self.viewer(ui));

        // Requests may have changed while drawing (slider, wheel, clicks).
        self.update_requests();
    }
}

/// A file name from a stack label: "#3 XA Coronary run  [30 fr]" becomes
/// "3_XA_Coronary_run".
fn file_stem(label: &str) -> String {
    let base = label.split('[').next().unwrap_or(label);
    let mut out = String::new();
    for c in base.chars() {
        if c.is_alphanumeric() || c == '-' {
            out.push(c);
        } else if !out.is_empty() && !out.ends_with('_') {
            out.push('_');
        }
    }
    let out = out.trim_end_matches('_');
    if out.is_empty() {
        "series".into()
    } else {
        out.into()
    }
}

/// Wheel movement this frame in lines, ignoring Ctrl/Cmd (that zooms).
fn wheel_lines(i: &egui::InputState) -> f32 {
    let mut lines = 0.0;
    for ev in &i.raw.events {
        if let egui::Event::MouseWheel {
            unit,
            delta,
            modifiers,
            ..
        } = ev
        {
            if modifiers.command || modifiers.ctrl {
                continue;
            }
            lines += match unit {
                egui::MouseWheelUnit::Line => delta.y,
                egui::MouseWheelUnit::Point => delta.y / 40.0,
                egui::MouseWheelUnit::Page => delta.y * 10.0,
            };
        }
    }
    lines
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Rotation turning by `yaw` about the screen's vertical axis, then by
/// `pitch` about its horizontal axis. Rows are screen x (right), y (down)
/// and z (into the screen).
fn rotation(yaw: f32, pitch: f32) -> [[f32; 3]; 3] {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    [
        [cy, 0.0, sy],
        [sp * sy, cp, -sp * cy],
        [-cp * sy, sp, cp * cy],
    ]
}

fn ordinal(n: usize) -> String {
    let suffix = match (n % 10, n % 100) {
        (1, 11) | (2, 12) | (3, 13) => "th",
        (1, _) => "st",
        (2, _) => "nd",
        (3, _) => "rd",
        _ => "th",
    };
    format!("{n}{suffix}")
}

fn pixel_size(sp: Spacing) -> String {
    let mut text = if (sp.row - sp.col).abs() < 1e-4 {
        format!("{:.2} mm", sp.col)
    } else {
        format!("{:.2} × {:.2} mm", sp.col, sp.row)
    };
    if sp.at_detector {
        text.push_str(" at detector");
    }
    text
}

/// A round length near `target` millimetres: 1, 2 or 5 times a power of ten.
fn nice_length(target: f32) -> f32 {
    let base = 10f32.powf(target.max(1e-3).log10().floor());
    [5.0, 2.0, 1.0]
        .into_iter()
        .map(|m| m * base)
        .find(|&l| l <= target)
        .unwrap_or(base)
}

fn length_label(mm: f32) -> String {
    if mm >= 10.0 && (mm / 10.0).fract() == 0.0 {
        format!("{} cm", mm / 10.0)
    } else if mm >= 1.0 {
        format!("{mm} mm")
    } else {
        format!("{mm:.1} mm")
    }
}

/// Millimetre rulers along the bottom and the right edge of the view.
fn draw_rulers(painter: &egui::Painter, rect: Rect, sp: Spacing, points_per_pixel: f32) {
    let colour = Color32::from_gray(225);
    let shadow = Color32::from_black_alpha(200);
    let font = FontId::proportional(12.0);
    // Draw each line twice: a dark shadow, then the line, so it shows on
    // bright and dark images alike.
    let line = |a: Pos2, b: Pos2| {
        let off = Vec2::splat(1.0);
        painter.line_segment([a + off, b + off], (1.5, shadow));
        painter.line_segment([a, b], (1.5, colour));
    };
    let label = |pos: Pos2, align: Align2, text: &str| {
        painter.text(pos + Vec2::splat(1.0), align, text, font.clone(), shadow);
        painter.text(pos, align, text, font.clone(), colour);
    };
    let ticks = |mm: f32| -> usize {
        match (mm / 10f32.powf(mm.log10().floor())).round() as u32 {
            1 => 10,
            2 => 4,
            _ => 5,
        }
    };
    let suffix = if sp.at_detector { " at detector" } else { "" };

    // Horizontal: centred near the bottom.
    let mm_per_point = sp.col / points_per_pixel;
    let mm = nice_length(150.0 * mm_per_point);
    let len = mm / mm_per_point;
    let y = rect.bottom() - 30.0;
    let x0 = rect.center().x - len / 2.0;
    line(Pos2::new(x0, y), Pos2::new(x0 + len, y));
    let n = ticks(mm);
    for i in 0..=n {
        let x = x0 + len * i as f32 / n as f32;
        let tall = i == 0 || i == n || 2 * i == n;
        line(
            Pos2::new(x, y),
            Pos2::new(x, y - if tall { 8.0 } else { 4.0 }),
        );
    }
    label(
        Pos2::new(rect.center().x, y - 10.0),
        Align2::CENTER_BOTTOM,
        &format!("{}{suffix}", length_label(mm)),
    );

    // Vertical: centred on the right edge.
    let mm_per_point = sp.row / points_per_pixel;
    let mm = nice_length(150.0 * mm_per_point);
    let len = mm / mm_per_point;
    let x = rect.right() - 24.0;
    let y0 = rect.center().y - len / 2.0;
    line(Pos2::new(x, y0), Pos2::new(x, y0 + len));
    let n = ticks(mm);
    for i in 0..=n {
        let y = y0 + len * i as f32 / n as f32;
        let tall = i == 0 || i == n || 2 * i == n;
        line(
            Pos2::new(x, y),
            Pos2::new(x - if tall { 8.0 } else { 4.0 }, y),
        );
    }
    label(
        Pos2::new(x - 12.0, rect.center().y),
        Align2::RIGHT_CENTER,
        &length_label(mm),
    );
}

#[cfg(test)]
mod tests {
    use super::file_stem;

    #[test]
    fn file_names_come_from_labels() {
        assert_eq!(
            file_stem("#3 XA Coronary run  [30 fr]"),
            "3_XA_Coronary_run"
        );
        assert_eq!(file_stem("#2 CT Axial 3mm · 4  [80]"), "2_CT_Axial_3mm_4");
        assert_eq!(file_stem("[1]"), "series");
    }

    #[test]
    fn ruler_lengths_are_round() {
        use super::{length_label, nice_length};
        assert_eq!(nice_length(73.0), 50.0);
        assert_eq!(nice_length(19.9), 10.0);
        assert_eq!(nice_length(20.0), 20.0);
        assert_eq!(nice_length(0.7), 0.5);
        assert_eq!(length_label(50.0), "5 cm");
        assert_eq!(length_label(5.0), "5 mm");
        assert_eq!(length_label(0.5), "0.5 mm");
    }

    #[test]
    fn rotation_is_identity_at_rest() {
        assert_eq!(
            super::rotation(0.0, 0.0),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );
    }
}

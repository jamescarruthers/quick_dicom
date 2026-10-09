//! Background decoding with a memory-bounded frame cache.
//!
//! The UI tells the loader which frames it wants, nearest first. Worker
//! threads decode them in that order, so the frame on screen comes first
//! and its neighbours follow, which makes scrubbing instant once warm.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use dicom_object::DefaultDicomObject;

use crate::decode::{Frame, decode_frame};
use crate::scan::{FrameRef, open_dicom};

#[derive(Clone)]
pub enum Slot {
    Ready(Arc<Frame>),
    Failed(Arc<str>),
}

struct Entry {
    slot: Slot,
    bytes: usize,
    last_used: u64,
}

type OpenFile = Arc<OnceLock<Result<Arc<DefaultDicomObject>, String>>>;

/// Multi-frame files stay open so each frame does not re-read the file.
const OPEN_FILES: usize = 2;

struct State {
    queue: VecDeque<FrameRef>,
    in_flight: HashSet<FrameRef>,
    entries: HashMap<FrameRef, Entry>,
    bytes: usize,
    tick: u64,
    open_files: VecDeque<(Arc<Path>, OpenFile)>,
    shutdown: bool,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    ctx: eframe::egui::Context,
    max_dim: u32,
    budget: usize,
}

pub struct Loader {
    shared: Arc<Shared>,
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().shutdown = true;
        self.shared.wake.notify_all();
    }
}

impl Loader {
    pub fn new(ctx: eframe::egui::Context, max_dim: u32, budget: usize) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                queue: VecDeque::new(),
                in_flight: HashSet::new(),
                entries: HashMap::new(),
                bytes: 0,
                tick: 0,
                open_files: VecDeque::new(),
                shutdown: false,
            }),
            wake: Condvar::new(),
            ctx,
            max_dim,
            budget,
        });
        let workers = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1))
            .unwrap_or(2)
            .clamp(1, 12);
        for i in 0..workers {
            let shared = shared.clone();
            std::thread::Builder::new()
                .name(format!("decode-{i}"))
                .spawn(move || worker(shared))
                .expect("failed to spawn decode thread");
        }
        Loader { shared }
    }

    pub fn budget(&self) -> usize {
        self.shared.budget
    }

    pub fn get(&self, key: &FrameRef) -> Option<Slot> {
        let mut st = self.shared.state.lock().unwrap();
        st.tick += 1;
        let tick = st.tick;
        st.entries.get_mut(key).map(|e| {
            e.last_used = tick;
            e.slot.clone()
        })
    }

    /// Replace the work queue. `wanted` is in priority order, most urgent
    /// first. Cached frames in the list are marked as recently used, in
    /// reverse order, so the least urgent ones are evicted first.
    pub fn request(&self, wanted: &[FrameRef]) {
        let mut st = self.shared.state.lock().unwrap();
        st.queue.clear();
        for key in wanted.iter().rev() {
            st.tick += 1;
            let tick = st.tick;
            if let Some(e) = st.entries.get_mut(key) {
                e.last_used = tick;
            }
        }
        for key in wanted {
            if !st.entries.contains_key(key) && !st.in_flight.contains(key) {
                st.queue.push_back(key.clone());
            }
        }
        drop(st);
        self.shared.wake.notify_all();
    }

    /// Which of `frames` are decoded, for the cache bar under the slider.
    pub fn cached_mask(&self, frames: &[FrameRef]) -> Vec<bool> {
        let st = self.shared.state.lock().unwrap();
        frames.iter().map(|f| st.entries.contains_key(f)).collect()
    }

    pub fn clear(&self) {
        let mut st = self.shared.state.lock().unwrap();
        st.queue.clear();
        st.entries.clear();
        st.open_files.clear();
        st.bytes = 0;
    }
}

fn worker(shared: Arc<Shared>) {
    loop {
        let key = {
            let mut st = shared.state.lock().unwrap();
            loop {
                if st.shutdown {
                    return;
                }
                if let Some(k) = st.queue.pop_front() {
                    if !st.entries.contains_key(&k) && st.in_flight.insert(k.clone()) {
                        break k;
                    }
                    continue;
                }
                st = shared.wake.wait(st).unwrap();
            }
        };

        let slot = match load(&shared, &key) {
            Ok(frame) => Slot::Ready(Arc::new(frame)),
            Err(e) => Slot::Failed(Arc::from(e.as_str())),
        };
        let bytes = match &slot {
            Slot::Ready(f) => f.bytes(),
            Slot::Failed(_) => 0,
        };

        let mut st = shared.state.lock().unwrap();
        st.in_flight.remove(&key);
        st.tick += 1;
        let tick = st.tick;
        st.bytes += bytes;
        if let Some(old) = st.entries.insert(
            key,
            Entry {
                slot,
                bytes,
                last_used: tick,
            },
        ) {
            st.bytes -= old.bytes;
        }
        evict(&mut st, shared.budget);
        drop(st);
        shared.ctx.request_repaint();
    }
}

fn evict(st: &mut State, budget: usize) {
    if st.bytes <= budget {
        return;
    }
    let mut by_age: Vec<(u64, FrameRef)> = st
        .entries
        .iter()
        .map(|(k, e)| (e.last_used, k.clone()))
        .collect();
    by_age.sort_unstable_by_key(|(t, _)| *t);
    // Leave headroom so we do not evict on every insert.
    let target = budget / 10 * 9;
    for (_, k) in by_age {
        if st.bytes <= target {
            break;
        }
        if let Some(e) = st.entries.remove(&k) {
            st.bytes -= e.bytes;
        }
    }
}

fn load(shared: &Shared, key: &FrameRef) -> Result<Frame, String> {
    if !key.multi {
        // Most files hold one frame; read, decode, and let it go.
        let obj = open_dicom(&key.path, false)?;
        return decode_frame(&obj, 0, shared.max_dim);
    }
    let cell = file_cell(shared, &key.path);
    let obj = cell
        .get_or_init(|| open_dicom(&key.path, false).map(Arc::new))
        .clone()?;
    decode_frame(&obj, key.frame, shared.max_dim)
}

/// The shared open-file slot for `path`. Workers that ask for the same file
/// at once wait for one read rather than each reading it.
fn file_cell(shared: &Shared, path: &Arc<Path>) -> OpenFile {
    let mut st = shared.state.lock().unwrap();
    if let Some(i) = st.open_files.iter().position(|(p, _)| p == path) {
        let entry = st.open_files.remove(i).unwrap();
        let cell = entry.1.clone();
        st.open_files.push_back(entry);
        return cell;
    }
    let cell: OpenFile = Arc::new(OnceLock::new());
    st.open_files.push_back((path.clone(), cell.clone()));
    while st.open_files.len() > OPEN_FILES {
        st.open_files.pop_front();
    }
    cell
}

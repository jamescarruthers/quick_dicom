// Hide the console window on Windows release builds.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod app;
mod check;
mod colormap;
mod decode;
mod export;
mod loader;
mod mp4;
mod render;
mod render3d;
mod scan;
mod volume;

use std::path::PathBuf;
use std::sync::Arc;

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};

fn main() {
    let mut args = std::env::args_os().skip(1);
    let first = args.next();
    if first.as_deref() == Some("--check".as_ref()) {
        attach_console();
        let Some(dir) = args.next() else {
            eprintln!("usage: quick_dicom --check <folder>");
            std::process::exit(2);
        };
        std::process::exit(if check::run(PathBuf::from(dir)) { 0 } else { 1 });
    }
    let path = first.map(PathBuf::from);

    let mut options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Quick DICOM")
            .with_inner_size([1280.0, 860.0])
            .with_min_inner_size([480.0, 320.0])
            .with_drag_and_drop(true),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    // Ask for the largest textures the GPU offers, so big radiographs and
    // mammograms upload at full resolution.
    if let egui_wgpu::WgpuSetup::CreateNew(setup) = &mut options.wgpu_options.wgpu_setup {
        setup.device_descriptor = Arc::new(|adapter: &wgpu::Adapter| {
            let base = if adapter.get_info().backend == wgpu::Backend::Gl {
                wgpu::Limits::downlevel_webgl2_defaults()
            } else {
                wgpu::Limits::default()
            };
            wgpu::DeviceDescriptor {
                label: Some("quick_dicom"),
                required_limits: wgpu::Limits {
                    max_texture_dimension_2d: adapter.limits().max_texture_dimension_2d.min(16384),
                    ..base
                },
                ..Default::default()
            }
        });
    }

    let result = eframe::run_native(
        "Quick DICOM",
        options,
        Box::new(move |cc| Ok(Box::new(app::ViewerApp::new(cc, path)))),
    );
    if let Err(e) = result {
        let msg = format!(
            "Quick DICOM could not start: {e}\n\n\
             It needs a graphics driver with Vulkan, Metal, DirectX 12 or OpenGL."
        );
        eprintln!("{msg}");
        rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Error)
            .set_title("Quick DICOM")
            .set_description(msg)
            .show();
        std::process::exit(1);
    }
}

/// Release builds on Windows have no console of their own, so borrow the
/// terminal that started us for `--check` output.
fn attach_console() {
    #[cfg(windows)]
    // SAFETY: AttachConsole takes no pointers and fails harmlessly when
    // there is no parent console.
    unsafe {
        use windows_sys::Win32::System::Console::{ATTACH_PARENT_PROCESS, AttachConsole};
        AttachConsole(ATTACH_PARENT_PROCESS);
    }
}

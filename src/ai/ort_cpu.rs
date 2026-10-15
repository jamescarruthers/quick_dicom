//! Runs a network with ONNX Runtime on the CPU. ONNX Runtime publishes no
//! build for Intel Macs, which use the GPU instead.

use std::sync::OnceLock;

use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;

use super::net::Net;
use super::onnx;

/// ONNX Runtime allows one environment per process.
fn environment() -> Result<Environment, String> {
    static ENV: OnceLock<Result<Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| {
        ort::init()
            .with_name("quick_dicom")
            .build()
            .map_err(|e| e.to_string())
    })
    .clone()
}

pub struct OrtNet {
    session: Session,
}

impl OrtNet {
    pub fn new(net: &Net) -> Result<OrtNet, String> {
        let model = onnx::model(net);
        let session = Session::builder(&environment()?)
            .and_then(|mut b| b.commit_from_memory(&model))
            .map_err(|e| format!("ONNX Runtime: {e}"))?;
        Ok(OrtNet { session })
    }

    /// Scores for each class, class by class, for an input of `size`.
    pub fn logits(&mut self, input: &[f32], size: [usize; 3]) -> Result<Vec<f32>, String> {
        let channels = input.len() / (size[0] * size[1] * size[2]);
        let shape = [1usize, channels, size[0], size[1], size[2]];
        let tensor = Tensor::from_array((shape, input.to_vec())).map_err(|e| e.to_string())?;
        let out = self
            .session
            .run(ort::inputs!["t0" => tensor])
            .map_err(|e| format!("ONNX Runtime: {e}"))?;
        let (_, data) = out["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        Ok(data.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::net::{run_cpu, tests::*};

    fn worst(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// Every layer type, through ONNX Runtime and the plain CPU code, on
    /// small networks with made-up weights and odd sizes.
    #[test]
    fn matches_the_plain_cpu_code() {
        for (net, size) in [(small_unet(), [8, 12, 4]), (small_segresnet(), [12, 8, 16])] {
            let input = random_input(size.iter().product(), 5);
            let want = run_cpu(&net, &input, size);
            let got = OrtNet::new(&net).unwrap().logits(&input, size).unwrap();
            let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            assert!(
                worst(&got, &want) < 1e-4 * scale.max(1.0),
                "{}",
                worst(&got, &want)
            );
        }
    }

    /// The real models through ONNX Runtime against PyTorch, and their
    /// speed on a 96-voxel block. Run with
    /// `AI_REF=<folder> cargo test --release real_weights -- --ignored`.
    #[test]
    #[ignore]
    fn real_weights_through_onnx_runtime() {
        let dir = std::path::PathBuf::from(std::env::var("AI_REF").unwrap());
        let floats = |name: &str| -> Vec<f32> {
            let b = std::fs::read(dir.join(name)).unwrap();
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };
        let input = floats("input32.f32");
        for (name, file) in [("spleen", "spleen.pt"), ("wholebody", "wholebody.pt")] {
            let mut weights =
                crate::ai::torch::read(&std::fs::read(dir.join(file)).unwrap()).unwrap();
            let net = if name == "spleen" {
                crate::ai::net::unet(&mut weights, 1, 2, &[16, 32, 64, 128, 256]).unwrap()
            } else {
                crate::ai::net::segresnet(&mut weights, 1, 105, 32, &[1, 2, 2, 4], &[1, 1, 1])
                    .unwrap()
            };
            let want = floats(&format!("{name}_logits32.f32"));
            let mut ort = OrtNet::new(&net).unwrap();
            let got = ort.logits(&input, [32, 32, 32]).unwrap();
            println!("{name}: worst difference {}", worst(&got, &want));
            assert!(worst(&got, &want) < 1e-3);
            let block = random_input(96 * 96 * 96, 9);
            ort.logits(&block, [96, 96, 96]).unwrap();
            let start = std::time::Instant::now();
            ort.logits(&block, [96, 96, 96]).unwrap();
            println!("{name}: {:.2} s for 96^3", start.elapsed().as_secs_f32());
        }
    }
}

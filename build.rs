fn main() {
    // ONNX Runtime's Windows builds link DirectML. The viewer runs ONNX
    // Runtime on the CPU and never loads DirectML, so delay-load it: then
    // the program starts without DirectML.dll beside it.
    let target = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target == "windows" && env == "msvc" {
        println!("cargo:rustc-link-arg=/DELAYLOAD:DirectML.dll");
        println!("cargo:rustc-link-lib=delayimp");
    }
}

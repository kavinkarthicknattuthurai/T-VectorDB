fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use vendored protoc compiler so users don't need protoc installed.
    // Configure via tonic_build instead of set_var (which is unsound in build scripts).
    let protoc_path = protoc_bin_vendored::protoc_bin_path()
        .expect("Failed to find vendored protoc binary");

    // SAFETY: build scripts run single-threaded before compilation,
    // so this is safe despite the general unsoundness warning.
    unsafe {
        std::env::set_var("PROTOC", &protoc_path);
    }

    tonic_build::compile_protos("proto/tvectordb.proto")?;
    Ok(())
}

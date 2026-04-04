fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Inject vendored protoc compiler so users don't need protoc installed
    std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path().unwrap());
    
    tonic_build::compile_protos("proto/tvectordb.proto")?;
    Ok(())
}

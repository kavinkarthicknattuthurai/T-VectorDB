//! # T-VectorDB V2 — Zero-Latency, Configurable Compressed Vector Database
//!
//! Usage:
//!   cargo run --release                       # d=1536, 3-bit, port 3000
//!   cargo run --release -- --dim 384 --bits 4 # 384-dim, 4-bit (near-lossless)
//!   cargo run --release -- --bits 2           # 2-bit (maximum compression)

use std::sync::Arc;

mod turbo_math;
mod storage_engine;
mod execution_engine;
mod hnsw;
mod api_server;
mod grpc_server;

use turbo_math::{BitWidth, TurboIndex};
use storage_engine::Database;
use api_server::{AppState, create_router};
use grpc_server::{TVectorService, pb::t_vector_server::TVectorServer};
use tonic::transport::Server;

fn parse_bitwidth(s: &str) -> BitWidth {
    match s {
        "2" => BitWidth::Bits2,
        "3" => BitWidth::Bits3,
        "4" => BitWidth::Bits4,
        _ => {
            eprintln!("Invalid bit-width '{}'. Must be 2, 3, or 4. Defaulting to 3.", s);
            BitWidth::Bits3
        }
    }
}

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .init();

    // Parse CLI arguments
    let dimension: usize = std::env::args()
        .position(|a| a == "--dim")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1536);

    let bits: BitWidth = std::env::args()
        .position(|a| a == "--bits")
        .and_then(|i| std::env::args().nth(i + 1))
        .map(|s| parse_bitwidth(&s))
        .unwrap_or(BitWidth::Bits3);

    let rest_port: u16 = std::env::args()
        .position(|a| a == "--port")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);

    let grpc_port: u16 = std::env::args()
        .position(|a| a == "--grpc-port")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(50051);

    let data_dir = std::env::args()
        .position(|a| a == "--data")
        .and_then(|i| std::env::args().nth(i + 1))
        .unwrap_or_else(|| "./data".to_string());

    // Banner
    println!();
    println!("  ╔══════════════════════════════════════════════════════════╗");
    println!("  ║                                                          ║");
    println!("  ║   ████████╗  ██╗   ██╗███████╗ ██████╗████████╗██████╗   ║");
    println!("  ║   ╚══██╔══╝  ██║   ██║██╔════╝██╔════╝╚══██╔══╝██╔══██╗ ║");
    println!("  ║      ██║     ██║   ██║█████╗  ██║        ██║   ██║  ██║  ║");
    println!("  ║      ██║     ╚██╗ ██╔╝██╔══╝  ██║        ██║   ██║  ██║  ║");
    println!("  ║      ██║      ╚████╔╝ ███████╗╚██████╗   ██║   ██████╔╝  ║");
    println!("  ║      ╚═╝       ╚═══╝  ╚══════╝ ╚═════╝   ╚═╝   ╚═════╝  ║");
    println!("  ║                                                          ║");
    println!("  ║   Zero-Latency · Configurable · LLM-Native VectorDB     ║");
    println!("  ║   Built on TurboQuant (ICLR 2026, Google Research)       ║");
    println!("  ║                                                          ║");
    println!("  ╚══════════════════════════════════════════════════════════╝");
    println!();

    // Initialize TurboIndex
    tracing::info!("Initializing TurboIndex: d={}, bits={}, compression={:.1}x ...",
        dimension, bits.bits(), bits.compression_ratio());
    let start = std::time::Instant::now();
    let index = TurboIndex::new(dimension, bits);
    let init_time = start.elapsed();
    tracing::info!("TurboIndex ready in {:.2?}", init_time);

    // Initialize Database (with persistence)
    tracing::info!("Opening database at: {}", data_dir);
    let db = Database::new(&data_dir).expect("Failed to open database");
    tracing::info!("Database ready. Vectors loaded: {}", db.len());

    // Rebuild HNSW graph from persisted vectors
    db.rebuild_hnsw(&index);

    // Build shared state
    let state = Arc::new(AppState {
        db,
        index,
    });

    println!();
    println!("  🚀 T-VectorDB v{} is live!", env!("CARGO_PKG_VERSION"));
    println!("  ⚙️  Config: d={}, {}-bit, {:.1}x compression", dimension, bits.bits(), bits.compression_ratio());
    println!();
    println!("  [REST API] http://localhost:{}", rest_port);
    println!("  📊 Stats:            GET    /stats");
    println!("  📥 Insert:           POST   /insert");
    println!("  🔍 Search:           POST   /search");
    println!();
    println!("  [gRPC BINDING] tcp://0.0.0.0:{}", grpc_port);
    println!("  ⚡ Binary protocol active for extreme throughput.");
    println!();

    // 1. Start REST Server (Axum)
    let rest_addr = format!("0.0.0.0:{}", rest_port);
    let router = create_router(state.clone());
    let rest_listener = tokio::net::TcpListener::bind(&rest_addr).await.expect("Failed to bind REST port");
    
    let rest_server = tokio::spawn(async move {
        axum::serve(rest_listener, router).await.expect("REST Server crashed")
    });

    // 2. Start gRPC Server (Tonic)
    let grpc_addr = format!("0.0.0.0:{}", grpc_port).parse().unwrap();
    let tv_service = TVectorService { state };
    
    let grpc_server = tokio::spawn(async move {
        Server::builder()
            .add_service(TVectorServer::new(tv_service))
            .serve(grpc_addr)
            .await
            .expect("gRPC Server crashed");
    });

    // Wait for both servers (they run indefinitely)
    let _ = tokio::try_join!(rest_server, grpc_server);
}

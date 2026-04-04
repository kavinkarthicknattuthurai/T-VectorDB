//! # T-VectorDB V2 — Zero-Latency, Configurable Compressed Vector Database
//!
//! Usage:
//!   cargo run --release                       # d=1536, 3-bit, port 3000
//!   cargo run --release -- --dim 384 --bits 4 # 384-dim, 4-bit (near-lossless)
//!   cargo run --release -- --bits 2           # 2-bit (maximum compression)

use std::sync::{Arc, RwLock};

mod turbo_math;
mod storage_engine;
mod execution_engine;
mod api_server;

use turbo_math::{BitWidth, TurboIndex};
use storage_engine::Database;
use api_server::{AppState, create_router};

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

    let port: u16 = std::env::args()
        .position(|a| a == "--port")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);

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

    // Build shared state
    let state = Arc::new(AppState {
        db: RwLock::new(db),
        index,
    });

    // Start server
    let addr = format!("0.0.0.0:{}", port);
    let router = create_router(state);
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("Failed to bind");

    println!();
    println!("  🚀 T-VectorDB v{} is live!", env!("CARGO_PKG_VERSION"));
    println!("  ⚙️  Config: d={}, {}-bit, {:.1}x compression", dimension, bits.bits(), bits.compression_ratio());
    println!();
    println!("  📊 Stats:            GET    http://localhost:{}/stats", port);
    println!("  📥 Insert:           POST   http://localhost:{}/insert", port);
    println!("  📥 Batch Insert:     POST   http://localhost:{}/insert_batch", port);
    println!("  🔍 Search:           POST   http://localhost:{}/search", port);
    println!("  🔍 Batch Search:     POST   http://localhost:{}/search_batch", port);
    println!("  🗑️  Delete:           DELETE http://localhost:{}/vectors/{{id}}", port);
    println!();

    axum::serve(listener, router).await.expect("Server crashed");
}

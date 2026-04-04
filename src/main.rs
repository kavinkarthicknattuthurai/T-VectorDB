//! # T-VectorDB — Zero-Latency, 16x Compressed Vector Database
//!
//! Built on the TurboQuant paper (ICLR 2026, Google Research).
//! Compresses Float32 vectors to 3 bits per dimension with zero training time.
//!
//! Usage:
//!   cargo run                  # Starts server on 0.0.0.0:3000 with d=1536
//!   cargo run -- --dim 768     # Custom dimension
//!   cargo run -- --port 8080   # Custom port

use std::sync::{Arc, RwLock};

mod turbo_math;
mod storage_engine;
mod execution_engine;
mod api_server;

use turbo_math::TurboIndex;
use storage_engine::Database;
use api_server::{AppState, create_router};

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .init();

    // Configuration (could be extended with clap for CLI args)
    let dimension: usize = std::env::args()
        .position(|a| a == "--dim")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1536);

    let port: u16 = std::env::args()
        .position(|a| a == "--port")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);

    let data_dir = std::env::args()
        .position(|a| a == "--data")
        .and_then(|i| std::env::args().nth(i + 1))
        .unwrap_or_else(|| "./data".to_string());

    // --- Banner ---
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
    println!("  ║   Zero-Latency · 16x Compressed · LLM-Native VectorDB   ║");
    println!("  ║   Built on TurboQuant (ICLR 2026, Google Research)       ║");
    println!("  ║                                                          ║");
    println!("  ╚══════════════════════════════════════════════════════════╝");
    println!();

    // --- Initialize TurboIndex ---
    tracing::info!("Initializing TurboIndex with dimension d={}...", dimension);
    let start = std::time::Instant::now();
    let index = TurboIndex::new(dimension);
    let init_time = start.elapsed();
    tracing::info!("TurboIndex ready in {:.2?}", init_time);

    // --- Initialize Database ---
    tracing::info!("Opening database at: {}", data_dir);
    let db = Database::new(&data_dir).expect("Failed to open database");
    tracing::info!("Database ready. Vectors in RAM: {}", db.len());

    // --- Build shared state ---
    let state = Arc::new(AppState {
        db: RwLock::new(db),
        index,
    });

    // --- Start HTTP server ---
    let addr = format!("0.0.0.0:{}", port);
    tracing::info!("Starting HTTP server on {}", addr);

    let router = create_router(state);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind address");

    println!();
    println!("  🚀 T-VectorDB is live at http://localhost:{}", port);
    println!("  📊 Stats:          GET  http://localhost:{}/stats", port);
    println!("  📥 Insert vector:  POST http://localhost:{}/insert", port);
    println!("  🔍 Search:         POST http://localhost:{}/search", port);
    println!();

    axum::serve(listener, router)
        .await
        .expect("Server crashed");
}

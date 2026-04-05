#![allow(clippy::all)]
#![allow(unused)]

//! # T-VectorDB V3 — Zero-Latency, Configurable Compressed Vector Database
//!
//! Usage:
//!   cargo run --release                       # d=1536, 3-bit, port 3000
//!   cargo run --release -- --dim 384 --bits 4 # 384-dim, 4-bit (near-lossless)
//!   cargo run --release -- --bits 2           # 2-bit (maximum compression)
//!
//! Environment variables (useful for Docker):
//!   TVECTORDB_DIM=384
//!   TVECTORDB_BITS=4
//!   TVECTORDB_PORT=3000
//!   TVECTORDB_GRPC_PORT=50051
//!   TVECTORDB_DATA_DIR=/app/data

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

/// Read a config value: CLI flag takes precedence, then env var, then default.
fn get_config<T: std::str::FromStr>(flag: &str, env_var: &str, default: T) -> T {
    // Check CLI args first
    if let Some(val) = std::env::args()
        .position(|a| a == flag)
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
    {
        return val;
    }
    // Check environment variable
    if let Ok(val) = std::env::var(env_var) {
        if let Ok(parsed) = val.parse() {
            return parsed;
        }
    }
    default
}

fn get_config_string(flag: &str, env_var: &str, default: &str) -> String {
    // Check CLI args first
    if let Some(val) = std::env::args()
        .position(|a| a == flag)
        .and_then(|i| std::env::args().nth(i + 1))
    {
        return val;
    }
    // Check environment variable
    if let Ok(val) = std::env::var(env_var) {
        return val;
    }
    default.to_string()
}

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .init();

    // Parse configuration (CLI > ENV > defaults)
    let dimension: usize = get_config("--dim", "TVECTORDB_DIM", 1536);

    let bits_str = get_config_string("--bits", "TVECTORDB_BITS", "3");
    let bits = parse_bitwidth(&bits_str);

    let rest_port: u16 = get_config("--port", "TVECTORDB_PORT", 3000);
    let grpc_port: u16 = get_config("--grpc-port", "TVECTORDB_GRPC_PORT", 50051);
    let data_dir = get_config_string("--data", "TVECTORDB_DATA_DIR", "./data");

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

    // Rebuild HNSW graph from persisted vectors (skips if graph loaded from disk)
    db.rebuild_hnsw(&index);

    // Build shared state
    let state = Arc::new(AppState {
        db,
        index,
    });

    println!();
    println!("  🚀 T-VectorDB v{} is live!", env!("CARGO_PKG_VERSION"));
    println!("  ⚙️  Config: d={}, {}-bit, {:.1}x compression", dimension, bits.bits(), bits.compression_ratio());
    println!("  📁 Data:   {}", data_dir);
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
    let tv_service = TVectorService { state: state.clone() };

    let grpc_server = tokio::spawn(async move {
        Server::builder()
            .add_service(TVectorServer::new(tv_service))
            .serve(grpc_addr)
            .await
            .expect("gRPC Server crashed");
    });

    // 3. Graceful shutdown handler
    let shutdown_state = state.clone();
    let shutdown_handler = tokio::spawn(async move {
        // Wait for Ctrl+C or SIGTERM
        tokio::signal::ctrl_c().await.ok();
        tracing::info!("Shutdown signal received. Flushing database...");

        if let Err(e) = shutdown_state.db.flush() {
            tracing::error!("Failed to flush database on shutdown: {}", e);
        } else {
            tracing::info!("Database flushed successfully. Goodbye!");
        }

        std::process::exit(0);
    });

    // Wait for any server to finish (they run indefinitely)
    let _ = tokio::try_join!(rest_server, grpc_server, shutdown_handler);
}

//! Benchmark script for T-VectorDB
//! Runs a heavy workload to prove performance metrics for the README.

use rand::Rng;
use std::time::Instant;
use tvectordb::execution_engine::search_ram_store;
use tvectordb::storage_engine::{compress_vector, PackedVector};
use tvectordb::turbo_math::TurboIndex;

fn main() {
    let d = 1536; // Standard OpenAI embedding size
    let num_vectors = 100_000;
    
    println!("🚀 Starting T-VectorDB Benchmarks");
    println!("========================================");
    println!("Dimension: {}", d);
    println!("Target Count: {} vectors", num_vectors);
    println!();

    // 1. Math Initialization
    print!("Initializing TurboIndex (QR Decomposition of {}x{} matrix)... ", d, d);
    let start = Instant::now();
    let index = TurboIndex::new(d);
    let init_time = start.elapsed();
    println!("{:.2?}", init_time);

    // 2. Data Generation
    print!("Generating random vectors... ");
    let mut rng = rand::thread_rng();
    let mut raw_vectors = Vec::with_capacity(num_vectors);
    for _ in 0..num_vectors {
        let v: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() - 0.5).collect();
        raw_vectors.push(v);
    }
    println!("Done.");

    // 3. Compression / Insertion Latency
    print!("Compressing {} vectors to 3 bits... ", num_vectors);
    let start = Instant::now();
    let mut db: Vec<PackedVector> = Vec::with_capacity(num_vectors);
    for (i, v) in raw_vectors.iter().enumerate() {
        db.push(compress_vector(&index, v, i as u64));
    }
    let compress_time = start.elapsed();
    println!("{:.2?} (Avg: {:.2?} per vector)", compress_time, compress_time / num_vectors as u32);
    
    // 4. Memory footprint
    let original_size = num_vectors * d * 4; // 4 bytes per f32
    let packed_size = num_vectors * ((d / 4) + (d / 8) + 4 + 8); // mse + qjl + residual(4) + id(8)
    println!();
    println!("🧠 Memory Footprint:");
    println!("Original (Float32): {:.2} MB", original_size as f32 / 1_048_576.0);
    println!("T-VectorDB (3-bit): {:.2} MB", packed_size as f32 / 1_048_576.0);
    println!("Compression Ratio:  {:.1}x smaller", original_size as f32 / packed_size as f32);
    println!();

    // 5. Search Latency
    let query = &raw_vectors[num_vectors / 2]; // Use a known vector
    let search_runs = 100;
    print!("Benchmarking {} consecutive searches over fully packed DB... ", search_runs);
    let start = Instant::now();
    for _ in 0..search_runs {
        // Find top 10
        let _results = search_ram_store(&index, &db, query, 10);
    }
    let total_search_time = start.elapsed();
    let avg_search_time = total_search_time / search_runs as u32;
    println!("Done.");
    
    let queries_per_sec = 1.0 / avg_search_time.as_secs_f64();

    println!();
    println!("⚡ Search Performance:");
    println!("Avg Latency per query: {:.2?}", avg_search_time);
    println!("Throughput:            {:.0} QPS (Queries Per Second)", queries_per_sec);
    println!("========================================");
    println!("Result: Crushing it. 🔨");
}

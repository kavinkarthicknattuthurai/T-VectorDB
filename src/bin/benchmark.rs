//! Benchmark V2: Tests all bit-widths and generates comparison tables

use rand::Rng;
use std::time::Instant;
use tvectordb::execution_engine::search_ram_store;
use tvectordb::storage_engine::{compress_vector, PackedVector};
use tvectordb::turbo_math::{BitWidth, TurboIndex};

struct BenchmarkResult {
    bits: usize,
    num_vectors: usize,
    compress_time_ms: f64,
    avg_compress_us: f64,
    original_mb: f64,
    compressed_mb: f64,
    compression_ratio: f64,
    avg_search_ms: f64,
    qps: f64,
    self_recall: f64,
}

fn run_benchmark(d: usize, bits: BitWidth, num_vectors: usize) -> BenchmarkResult {
    let index = TurboIndex::new(d, bits);
    let mut rng = rand::thread_rng();

    // Generate random vectors
    let mut raw_vectors: Vec<Vec<f32>> = Vec::with_capacity(num_vectors);
    for _ in 0..num_vectors {
        let v: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() - 0.5).collect();
        raw_vectors.push(v);
    }

    // Compress
    let start = Instant::now();
    let mut db: Vec<PackedVector> = Vec::with_capacity(num_vectors);
    for (i, v) in raw_vectors.iter().enumerate() {
        db.push(compress_vector(&index, v, i as u64));
    }
    let compress_time = start.elapsed();

    // Memory
    let original_size = num_vectors * d * 4;
    let compressed_size: usize = db.iter().map(|v| v.size_bytes()).sum();

    // Search latency (search for 50 known vectors, check if they are top-1)
    let search_runs = 50.min(num_vectors);
    let start = Instant::now();
    let mut hits = 0;
    for i in 0..search_runs {
        let query = &raw_vectors[i];
        let results = search_ram_store(&index, &db, query, 1);
        if !results.is_empty() && results[0].0 == i as u64 {
            hits += 1;
        }
    }
    let total_search_time = start.elapsed();
    let avg_search_time = total_search_time / search_runs as u32;

    BenchmarkResult {
        bits: bits.bits(),
        num_vectors,
        compress_time_ms: compress_time.as_secs_f64() * 1000.0,
        avg_compress_us: compress_time.as_secs_f64() * 1_000_000.0 / num_vectors as f64,
        original_mb: original_size as f64 / 1_048_576.0,
        compressed_mb: compressed_size as f64 / 1_048_576.0,
        compression_ratio: original_size as f64 / compressed_size as f64,
        avg_search_ms: avg_search_time.as_secs_f64() * 1000.0,
        qps: 1.0 / avg_search_time.as_secs_f64(),
        self_recall: hits as f64 / search_runs as f64 * 100.0,
    }
}

fn main() {
    let d = 384;  // BGE-small-en-v1.5 dimension (comparable to turboqvec benchmarks)
    let num_vectors = 10_000;

    println!("🚀 T-VectorDB Benchmark Suite V2");
    println!("========================================");
    println!("Dimension: {}", d);
    println!("Vectors:   {}", num_vectors);
    println!();

    println!("Running 2-bit benchmark...");
    let r2 = run_benchmark(d, BitWidth::Bits2, num_vectors);
    println!("Running 3-bit benchmark...");
    let r3 = run_benchmark(d, BitWidth::Bits3, num_vectors);
    println!("Running 4-bit benchmark...");
    let r4 = run_benchmark(d, BitWidth::Bits4, num_vectors);

    let results = [&r2, &r3, &r4];

    println!();
    println!("📊 COMPRESSION & MEMORY");
    println!("┌────────┬───────────┬──────────────┬──────────────┬─────────────┐");
    println!("│  Bits  │  Vectors  │ Original (MB)│Compressed(MB)│  Ratio      │");
    println!("├────────┼───────────┼──────────────┼──────────────┼─────────────┤");
    for r in &results {
        println!("│ {}-bit  │ {:>9} │ {:>12.2} │ {:>12.2} │ {:>9.1}x  │",
            r.bits, r.num_vectors, r.original_mb, r.compressed_mb, r.compression_ratio);
    }
    println!("└────────┴───────────┴──────────────┴──────────────┴─────────────┘");

    println!();
    println!("⚡ SEARCH PERFORMANCE");
    println!("┌────────┬─────────────────┬─────────────┬──────────────┐");
    println!("│  Bits  │ Avg Search (ms) │    QPS      │ Self-Recall  │");
    println!("├────────┼─────────────────┼─────────────┼──────────────┤");
    for r in &results {
        println!("│ {}-bit  │ {:>15.2} │ {:>11.0} │ {:>10.1}%  │",
            r.bits, r.avg_search_ms, r.qps, r.self_recall);
    }
    println!("└────────┴─────────────────┴─────────────┴──────────────┘");

    println!();
    println!("🔧 INGESTION SPEED");
    println!("┌────────┬────────────────────┬──────────────────┐");
    println!("│  Bits  │ Total Compress (ms)│ Avg per vec (µs) │");
    println!("├────────┼────────────────────┼──────────────────┤");
    for r in &results {
        println!("│ {}-bit  │ {:>18.1} │ {:>16.1} │",
            r.bits, r.compress_time_ms, r.avg_compress_us);
    }
    println!("└────────┴────────────────────┴──────────────────┘");

    println!();
    println!("========================================");
    println!("✅ Benchmark complete. These numbers don't lie. 🔥");
}

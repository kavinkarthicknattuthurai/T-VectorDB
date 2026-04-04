//! # Recall Validation — Scientific Accuracy Measurement
//!
//! Computes actual recall@K and cosine error by comparing
//! TurboQuant approximate search against brute-force exact search.
//! This is the test that proves our numbers are real.

use rand::Rng;
use std::time::Instant;
use tvectordb::execution_engine::search_ram_store;
use tvectordb::storage_engine::{compress_vector, l2_normalize, PackedVector};
use tvectordb::turbo_math::{BitWidth, TurboIndex};
use nalgebra::DVector;

/// Brute-force exact cosine similarity search (the ground truth).
fn brute_force_search(
    raw_vectors: &[Vec<f32>],
    query: &[f32],
    top_k: usize,
) -> Vec<(usize, f32)> {
    let mut q = DVector::from_column_slice(query);
    let q_norm = q.norm();
    if q_norm > 1e-10 { q /= q_norm; }

    let mut scores: Vec<(usize, f32)> = raw_vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut dv = DVector::from_column_slice(v);
            let n = dv.norm();
            if n > 1e-10 { dv /= n; }
            (i, q.dot(&dv))
        })
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.truncate(top_k);
    scores
}

/// Compute recall@K: what fraction of the true top-K did our
/// approximate search actually find?
fn compute_recall(
    approx_ids: &[u64],
    exact_ids: &[usize],
) -> f64 {
    let hits = approx_ids
        .iter()
        .filter(|&&id| exact_ids.contains(&(id as usize)))
        .count();
    hits as f64 / exact_ids.len() as f64
}

/// Compute mean cosine error between approximate and exact scores.
fn compute_cosine_error(
    index: &TurboIndex,
    db: &[PackedVector],
    raw_vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
) -> (f64, f64) {
    // For each query, compare the approximate score of the top-1 result
    // against the exact cosine similarity
    let mut total_error = 0.0f64;
    let mut total_drift = 0.0f64;
    let mut count = 0;

    for query in queries {
        let approx_results = search_ram_store(index, db, query, 1);
        if approx_results.is_empty() { continue; }

        let (approx_id, approx_score) = approx_results[0];
        let exact_results = brute_force_search(raw_vectors, query, 1);
        if exact_results.is_empty() { continue; }

        // Get exact score for the vector that our approximate search returned
        let mut q = DVector::from_column_slice(query);
        let qn = q.norm();
        if qn > 1e-10 { q /= qn; }

        let mut v = DVector::from_column_slice(&raw_vectors[approx_id as usize]);
        let vn = v.norm();
        if vn > 1e-10 { v /= vn; }

        let true_score = q.dot(&v);
        let error = (approx_score as f64 - true_score as f64).abs();
        let drift = approx_score as f64 - true_score as f64;

        total_error += error;
        total_drift += drift;
        count += 1;
    }

    if count > 0 {
        (total_error / count as f64, total_drift / count as f64)
    } else {
        (0.0, 0.0)
    }
}

fn main() {
    let d = 384;
    let num_vectors = 1000;
    let num_queries = 200;

    println!("🔬 T-VectorDB Recall Validation");
    println!("========================================");
    println!("Dimension:    {}", d);
    println!("DB Vectors:   {}", num_vectors);
    println!("Test Queries: {}", num_queries);
    println!();

    let mut rng = rand::thread_rng();

    // Generate random vectors (simulating embeddings)
    let raw_vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| (0..d).map(|_| rng.gen::<f32>() - 0.5).collect())
        .collect();

    // Generate random queries
    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| (0..d).map(|_| rng.gen::<f32>() - 0.5).collect())
        .collect();

    // Also test "self-recall": can we find vectors we inserted?
    let self_queries: Vec<Vec<f32>> = raw_vectors[0..num_queries.min(num_vectors)]
        .to_vec();

    println!("┌────────┬─────────┬─────────┬──────────┬──────────────┬────────────┐");
    println!("│  Bits  │  R@1    │  R@5    │  R@10    │ Mean Cos Err │ Mean Drift │");
    println!("├────────┼─────────┼─────────┼──────────┼──────────────┼────────────┤");

    for bits in [BitWidth::Bits2, BitWidth::Bits3, BitWidth::Bits4] {
        let index = TurboIndex::new(d, bits);

        // Build compressed DB
        let db: Vec<PackedVector> = raw_vectors
            .iter()
            .enumerate()
            .map(|(i, v)| compress_vector(&index, v, i as u64))
            .collect();

        // Compute Recall@1, @5, @10
        let mut recall_1_total = 0.0;
        let mut recall_5_total = 0.0;
        let mut recall_10_total = 0.0;
        let query_count = queries.len();

        for query in &queries {
            // Approximate results
            let approx_1 = search_ram_store(&index, &db, query, 1);
            let approx_5 = search_ram_store(&index, &db, query, 5);
            let approx_10 = search_ram_store(&index, &db, query, 10);

            // Exact ground truth
            let exact_10 = brute_force_search(&raw_vectors, query, 10);
            let exact_1_ids: Vec<usize> = exact_10.iter().take(1).map(|(i, _)| *i).collect();
            let exact_5_ids: Vec<usize> = exact_10.iter().take(5).map(|(i, _)| *i).collect();
            let exact_10_ids: Vec<usize> = exact_10.iter().map(|(i, _)| *i).collect();

            let a1: Vec<u64> = approx_1.iter().map(|(id, _)| *id).collect();
            let a5: Vec<u64> = approx_5.iter().map(|(id, _)| *id).collect();
            let a10: Vec<u64> = approx_10.iter().map(|(id, _)| *id).collect();

            recall_1_total += compute_recall(&a1, &exact_1_ids);
            recall_5_total += compute_recall(&a5, &exact_5_ids);
            recall_10_total += compute_recall(&a10, &exact_10_ids);
        }

        let r1 = recall_1_total / query_count as f64 * 100.0;
        let r5 = recall_5_total / query_count as f64 * 100.0;
        let r10 = recall_10_total / query_count as f64 * 100.0;

        // Compute cosine error
        let (mean_error, mean_drift) = compute_cosine_error(&index, &db, &raw_vectors, &queries);

        println!(
            "│ {}-bit  │ {:>5.1}%  │ {:>5.1}%  │ {:>5.1}%   │ {:>12.4} │ {:>10.4} │",
            bits.bits(), r1, r5, r10, mean_error, mean_drift
        );
    }

    println!("└────────┴─────────┴─────────┴──────────┴──────────────┴────────────┘");

    println!();
    println!("========================================");
    println!("✅ Validation complete. These are real numbers.");
}

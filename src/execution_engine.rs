//! # Execution Engine — Asymmetric Distance Computation (ADC) Search
//!
//! Implements the search loop from MID Section 4C:
//! - LUT-based MSE scoring (never decompress the database!)
//! - QJL sign-bit scoring for residual correction
//! - Min-heap top-k selection
//! - Two-tier hybrid search with disk-based re-ranking

use crate::storage_engine::{unpack_mse_byte, unpack_qjl_byte, Database, PackedVector};
use crate::turbo_math::TurboIndex;
use nalgebra::DVector;
use ordered_float::OrderedFloat;
use std::collections::BinaryHeap;
use std::cmp::Reverse;

/// A scored result: (score, id). Using Reverse for min-heap behavior.
type ScoredItem = Reverse<(OrderedFloat<f32>, u64)>;

/// Perform approximate search over the compressed RAM store using ADC.
///
/// This is the core hot-path. It transforms the query once, then streams
/// over all compressed vectors, scoring each one using lookup tables.
///
/// **CRITICAL**: We never decompress the database. The query is transformed
/// into the same space as the compressed vectors.
///
/// Returns the top `top_k` results as `(id, score)` pairs, sorted by score descending.
pub fn search_ram_store(
    index: &TurboIndex,
    db: &[PackedVector],
    query: &[f32],
    top_k: usize,
) -> Vec<(u64, f32)> {
    let d = index.d;
    assert_eq!(query.len(), d, "Query dimension mismatch");

    // Normalize the query
    let mut q = DVector::from_column_slice(query);
    let q_norm = q.norm();
    if q_norm > 1e-10 {
        q /= q_norm;
    }

    // --- Step 1: Rotate query ---
    // q_rot = Π · q
    let q_rot = &index.pi * &q;

    // --- Step 2: Build Look-Up Table (LUT) ---
    // LUT[i][j] = q_rot[i] × centroids[j]
    // Shape: d × 4
    // This precomputes all possible partial dot-products.
    let mut lut = vec![[0.0f32; 4]; d];
    for i in 0..d {
        for j in 0..4 {
            lut[i][j] = q_rot[i] * index.centroids[j];
        }
    }

    // --- Step 3: Project query for QJL ---
    // q_qjl = S · q
    let q_qjl = &index.s * &q;

    // --- Step 4: Score every vector in the database ---
    // Precompute the QJL scaling factor: γ × √(π / (2d))
    let qjl_scale_base = (std::f32::consts::PI / (2.0 * d as f32)).sqrt();

    // Min-heap to track top-k highest scores
    let mut heap: BinaryHeap<ScoredItem> = BinaryHeap::with_capacity(top_k + 1);

    for packed in db.iter() {
        // --- MSE Score: sum of LUT lookups ---
        let mut mse_score: f32 = 0.0;
        let mut dim_idx = 0;

        for &byte in packed.mse_bits.iter() {
            let (a, b, c, d_val) = unpack_mse_byte(byte);
            mse_score += lut[dim_idx][a as usize];
            mse_score += lut[dim_idx + 1][b as usize];
            mse_score += lut[dim_idx + 2][c as usize];
            mse_score += lut[dim_idx + 3][d_val as usize];
            dim_idx += 4;
        }

        // --- QJL Score: sign-bit dot product ---
        let mut qjl_score: f32 = 0.0;
        let mut bit_idx = 0;

        for &byte in packed.qjl_bits.iter() {
            let signs = unpack_qjl_byte(byte);
            for &sign in signs.iter() {
                // Convert {0, 1} → {-1, +1}: (2 * sign - 1)
                let sign_val = 2.0 * sign as f32 - 1.0;
                qjl_score += sign_val * q_qjl[bit_idx];
                bit_idx += 1;
            }
        }

        // --- Final Score ---
        // score = MSE_Score + γ × √(π/(2d)) × QJL_Score
        let final_score = mse_score + packed.residual_norm * qjl_scale_base * qjl_score;

        // Maintain top-k using min-heap
        if heap.len() < top_k {
            heap.push(Reverse((OrderedFloat(final_score), packed.id)));
        } else if let Some(&Reverse((min_score, _))) = heap.peek() {
            if OrderedFloat(final_score) > min_score {
                heap.pop();
                heap.push(Reverse((OrderedFloat(final_score), packed.id)));
            }
        }
    }

    // Extract results sorted by score descending
    let mut results: Vec<(u64, f32)> = heap
        .into_iter()
        .map(|Reverse((score, id))| (id, score.into_inner()))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Two-tier hybrid search: approximate RAM scan → exact disk re-ranking.
///
/// 1. Get top-100 candidates from the compressed RAM store (fast, approximate)
/// 2. Fetch their full-precision vectors from RocksDB (disk)
/// 3. Compute exact dot-product scores
/// 4. Return the true top-k results
pub fn hybrid_search(
    db: &Database,
    index: &TurboIndex,
    query: &[f32],
    top_k: usize,
) -> Vec<(u64, f32)> {
    let shortlist_size = 100.min(db.len());

    // Tier 1: Approximate search over compressed vectors
    let shortlist = search_ram_store(index, &db.ram, query, shortlist_size);

    // Normalize the query for exact dot-product
    let mut q = DVector::from_column_slice(query);
    let q_norm = q.norm();
    if q_norm > 1e-10 {
        q /= q_norm;
    }

    // Tier 2: Exact re-ranking using full-precision vectors from disk
    let mut exact_scores: Vec<(u64, f32)> = Vec::with_capacity(shortlist.len());

    for (id, _approx_score) in shortlist.iter() {
        if let Some(raw_vec) = db.get_raw_vector(*id, index.d) {
            // L2-normalize the stored vector
            let mut v = raw_vec;
            let v_norm = v.norm();
            if v_norm > 1e-10 {
                v /= v_norm;
            }
            // Exact cosine similarity = dot product of unit vectors
            let exact_score = q.dot(&v);
            exact_scores.push((*id, exact_score));
        }
    }

    // Sort by exact score descending
    exact_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Return top-k
    exact_scores.truncate(top_k);
    exact_scores
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage_engine::compress_vector;
    use crate::turbo_math::TurboIndex;
    use rand::Rng;

    fn random_vector(d: usize, rng: &mut impl Rng) -> Vec<f32> {
        (0..d).map(|_| rng.gen::<f32>() - 0.5).collect()
    }

    #[test]
    fn test_search_finds_inserted_vector() {
        let d = 64;
        let index = TurboIndex::new(d);
        let mut rng = rand::thread_rng();

        // Insert 50 random vectors
        let mut db: Vec<PackedVector> = Vec::new();
        let mut raw_vectors: Vec<Vec<f32>> = Vec::new();

        for i in 0..50 {
            let v = random_vector(d, &mut rng);
            db.push(compress_vector(&index, &v, i as u64));
            raw_vectors.push(v);
        }

        // Search with vector #25 as the query
        let query = &raw_vectors[25];
        let results = search_ram_store(&index, &db, query, 5);

        // The query vector itself should be the top result (or very close to it)
        assert!(!results.is_empty());
        let top_ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
        assert!(
            top_ids.contains(&25),
            "Expected vector 25 in top-5 results, got: {:?}",
            top_ids
        );
    }

    #[test]
    fn test_search_returns_correct_count() {
        let d = 64;
        let index = TurboIndex::new(d);
        let mut rng = rand::thread_rng();

        let mut db: Vec<PackedVector> = Vec::new();
        for i in 0..20 {
            let v = random_vector(d, &mut rng);
            db.push(compress_vector(&index, &v, i as u64));
        }

        let query = random_vector(d, &mut rng);

        // Ask for top-5
        let results = search_ram_store(&index, &db, &query, 5);
        assert_eq!(results.len(), 5);

        // Ask for more than available
        let results = search_ram_store(&index, &db, &query, 100);
        assert_eq!(results.len(), 20);  // Only 20 vectors in DB
    }

    #[test]
    fn test_scores_are_finite() {
        let d = 64;
        let index = TurboIndex::new(d);
        let mut rng = rand::thread_rng();

        let mut db: Vec<PackedVector> = Vec::new();
        for i in 0..10 {
            let v = random_vector(d, &mut rng);
            db.push(compress_vector(&index, &v, i as u64));
        }

        let query = random_vector(d, &mut rng);
        let results = search_ram_store(&index, &db, &query, 5);

        for (_, score) in &results {
            assert!(score.is_finite(), "Score is not finite: {}", score);
        }
    }
}

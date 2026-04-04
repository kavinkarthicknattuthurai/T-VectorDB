//! # Execution Engine V2 — Multi Bit-Width ADC Search
//!
//! Supports 2/3/4-bit quantized search with variable-width LUT,
//! hybrid two-tier re-ranking, and batch search.

use crate::storage_engine::{unpack_indices, unpack_qjl_byte, Database, PackedVector};
use crate::turbo_math::TurboIndex;
use nalgebra::DVector;
use ordered_float::OrderedFloat;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

type ScoredItem = Reverse<(OrderedFloat<f32>, u64)>;

/// Perform approximate search over the compressed RAM store using ADC.
///
/// Uses MSE-only scoring by default (best for normalized embeddings).
/// Set `use_qjl=true` for KV cache / attention vectors where QJL correction helps.
pub fn search_ram_store(
    index: &TurboIndex,
    db: &[PackedVector],
    query: &[f32],
    top_k: usize,
) -> Vec<(u64, f32)> {
    search_ram_store_with_options(index, db, query, top_k, false)
}

/// Search with explicit QJL control.
pub fn search_ram_store_with_options(
    index: &TurboIndex,
    db: &[PackedVector],
    query: &[f32],
    top_k: usize,
    use_qjl: bool,
) -> Vec<(u64, f32)> {
    let d = index.d;
    assert_eq!(query.len(), d, "Query dimension mismatch");

    // Normalize query
    let mut q = DVector::from_column_slice(query);
    let q_norm = q.norm();
    if q_norm > 1e-10 { q /= q_norm; }

    // Step 1: Rotate query
    let q_rot = &index.pi * &q;

    // Step 2: Build LUT — shape: d × num_levels
    let num_levels = index.num_levels;
    let mut lut = vec![vec![0.0f32; num_levels]; d];
    for i in 0..d {
        for j in 0..num_levels {
            lut[i][j] = q_rot[i] * index.centroids[j];
        }
    }

    // Optional: Project query for QJL
    let q_qjl = if use_qjl { Some(&index.s * &q) } else { None };
    let qjl_scale_base = (std::f32::consts::PI / (2.0 * d as f32)).sqrt();

    // Step 3: Score every vector
    let mut heap: BinaryHeap<ScoredItem> = BinaryHeap::with_capacity(top_k + 1);

    for packed in db.iter() {
        // Unpack MSE indices
        let indices = unpack_indices(&packed.mse_packed, d, index.bits);

        // MSE Score via LUT lookup
        let mut mse_score: f32 = 0.0;
        for (i, &idx) in indices.iter().enumerate() {
            mse_score += lut[i][idx as usize];
        }

        // QJL correction (optional — only helps for non-normalized vectors)
        let final_score = if use_qjl {
            if let Some(ref q_proj) = q_qjl {
                let mut qjl_score: f32 = 0.0;
                let mut bit_idx = 0;
                for &byte in packed.qjl_bits.iter() {
                    let signs = unpack_qjl_byte(byte);
                    for &sign in signs.iter() {
                        if bit_idx < d {
                            let sign_val = 2.0 * sign as f32 - 1.0;
                            qjl_score += sign_val * q_proj[bit_idx];
                            bit_idx += 1;
                        }
                    }
                }
                mse_score + packed.residual_norm * qjl_scale_base * qjl_score
            } else {
                mse_score
            }
        } else {
            mse_score
        };

        // Min-heap top-k
        if heap.len() < top_k {
            heap.push(Reverse((OrderedFloat(final_score), packed.id)));
        } else if let Some(&Reverse((min_score, _))) = heap.peek() {
            if OrderedFloat(final_score) > min_score {
                heap.pop();
                heap.push(Reverse((OrderedFloat(final_score), packed.id)));
            }
        }
    }

    let mut results: Vec<(u64, f32)> = heap
        .into_iter()
        .map(|Reverse((score, id))| (id, score.into_inner()))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Two-tier hybrid search: approximate RAM scan → exact disk re-ranking.
pub fn hybrid_search(
    db: &Database,
    index: &TurboIndex,
    query: &[f32],
    top_k: usize,
) -> Vec<(u64, f32)> {
    let shortlist_size = 100.min(db.len());
    
    // Acquire temporary read lock to scan RAM
    let ram_guard = db.ram.read().unwrap();
    let shortlist = search_ram_store(index, &ram_guard, query, shortlist_size);
    drop(ram_guard);

    let mut q = DVector::from_column_slice(query);
    let q_norm = q.norm();
    if q_norm > 1e-10 { q /= q_norm; }

    let mut exact_scores: Vec<(u64, f32)> = Vec::with_capacity(shortlist.len());
    for (id, _) in shortlist.iter() {
        if let Some(raw_vec) = db.get_raw_vector(*id, index.d) {
            let mut v = raw_vec;
            let v_norm = v.norm();
            if v_norm > 1e-10 { v /= v_norm; }
            let exact_score = q.dot(&v);
            exact_scores.push((*id, exact_score));
        }
    }

    exact_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    exact_scores.truncate(top_k);
    exact_scores
}

/// Batch search: run multiple queries and return results for each.
pub fn batch_search(
    index: &TurboIndex,
    db: &[PackedVector],
    queries: &[Vec<f32>],
    top_k: usize,
) -> Vec<Vec<(u64, f32)>> {
    queries
        .iter()
        .map(|q| search_ram_store(index, db, q, top_k))
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage_engine::compress_vector;
    use crate::turbo_math::{BitWidth, TurboIndex};
    use rand::Rng;

    fn random_vector(d: usize, rng: &mut impl Rng) -> Vec<f32> {
        (0..d).map(|_| rng.gen::<f32>() - 0.5).collect()
    }

    #[test]
    fn test_search_all_bitwidths() {
        let d = 64;
        let mut rng = rand::thread_rng();

        for bits in [BitWidth::Bits2, BitWidth::Bits3, BitWidth::Bits4] {
            let index = TurboIndex::new(d, bits);
            let mut db: Vec<PackedVector> = Vec::new();
            let mut raw_vectors: Vec<Vec<f32>> = Vec::new();

            for i in 0..50 {
                let v = random_vector(d, &mut rng);
                db.push(compress_vector(&index, &v, i as u64));
                raw_vectors.push(v);
            }

            let query = &raw_vectors[25];
            let results = search_ram_store(&index, &db, query, 5);

            assert!(!results.is_empty());
            let top_ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
            assert!(
                top_ids.contains(&25),
                "bits={}: Expected vector 25 in top-5, got: {:?}",
                bits.bits(), top_ids
            );
        }
    }

    #[test]
    fn test_scores_are_finite() {
        let d = 64;
        let index = TurboIndex::new(d, BitWidth::Bits4);
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

    #[test]
    fn test_batch_search() {
        let d = 64;
        let index = TurboIndex::new(d, BitWidth::Bits3);
        let mut rng = rand::thread_rng();

        let mut db: Vec<PackedVector> = Vec::new();
        for i in 0..20 {
            let v = random_vector(d, &mut rng);
            db.push(compress_vector(&index, &v, i as u64));
        }

        let queries: Vec<Vec<f32>> = (0..3).map(|_| random_vector(d, &mut rng)).collect();
        let all_results = batch_search(&index, &db, &queries, 5);

        assert_eq!(all_results.len(), 3);
        for results in &all_results {
            assert_eq!(results.len(), 5);
        }
    }
}

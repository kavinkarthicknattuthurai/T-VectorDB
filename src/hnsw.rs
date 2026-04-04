//! # HNSW — Hierarchical Navigable Small World Graph Index
//!
//! Provides O(log n) approximate nearest neighbor search by building
//! a multi-layer navigable graph over the compressed TurboQuant vectors.
//!
//! Key innovation: graph traversal uses the same LUT-based scoring as
//! the linear scan, so all distance computations run on compressed data.
//!
//! Algorithm (Malkov & Yashunin, 2018):
//! - Insert: assign random layer, greedy descend, connect to M nearest at each layer
//! - Search: descend from top layer, expand with beam width `ef` at layer 0

use crate::storage_engine::{unpack_indices, PackedVector};
use crate::turbo_math::TurboIndex;
use nalgebra::DVector;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

// ============================================================================
// LUT (Lookup Table) — Extracted for reuse by both linear scan and HNSW
// ============================================================================

/// Precomputed lookup table for scoring compressed vectors against a single query.
/// Built once per query, reused across all vector comparisons.
pub struct QueryLut {
    /// LUT[dim][centroid_index] = dot product contribution
    pub table: Vec<Vec<f32>>,
    pub d: usize,
}

/// Build a LUT for a query vector against the TurboIndex centroids.
pub fn build_lut(index: &TurboIndex, query: &[f32]) -> QueryLut {
    let d = index.d;

    // Normalize query
    let mut q = DVector::from_column_slice(query);
    let q_norm = q.norm();
    if q_norm > 1e-10 { q /= q_norm; }

    // Rotate query: q_rot = Π · q
    let q_rot = &index.pi * &q;

    // Build LUT: lut[i][j] = q_rot[i] * centroid[j]
    let num_levels = index.num_levels;
    let mut table = vec![vec![0.0f32; num_levels]; d];
    for i in 0..d {
        for j in 0..num_levels {
            table[i][j] = q_rot[i] * index.centroids[j];
        }
    }

    QueryLut { table, d }
}

/// Score a single packed vector against a precomputed LUT.
/// Returns the approximate dot product (higher = more similar).
#[inline]
pub fn score_packed(lut: &QueryLut, packed: &PackedVector, bits: crate::turbo_math::BitWidth) -> f32 {
    let indices = unpack_indices(&packed.mse_packed, lut.d, bits);
    let mut score: f32 = 0.0;
    for (i, &idx) in indices.iter().enumerate() {
        score += lut.table[i][idx as usize];
    }
    score
}

// ============================================================================
// HNSW Graph
// ============================================================================

/// Configuration for the HNSW graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Max neighbors per node at layers > 0 (default: 16)
    pub m: usize,
    /// Max neighbors at layer 0 — typically 2*M (default: 32)
    pub m_max0: usize,
    /// Beam width during construction (default: 200)
    pub ef_construction: usize,
    /// Level generation multiplier: 1/ln(M)
    pub ml: f64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16;
        HnswConfig {
            m,
            m_max0: m * 2,
            ef_construction: 200,
            ml: 1.0 / (m as f64).ln(),
        }
    }
}

/// The HNSW graph index.
///
/// Stores neighbor lists for each node at each layer. Nodes are indexed
/// by their position in the `Database.ram` vector (not by vector ID).
#[derive(Clone, Serialize, Deserialize)]
pub struct HnswGraph {
    /// Config parameters
    pub config: HnswConfig,
    /// layers[L] = vec of neighbor lists. layers[L][node_idx] = neighbor indices
    layers: Vec<Vec<Vec<usize>>>,
    /// Maximum layer assigned to each node
    node_max_layer: Vec<usize>,
    /// Entry point: index of the node at the highest layer
    entry_point: Option<usize>,
    /// Highest layer in the graph
    max_layer: usize,
    /// Total number of nodes
    num_nodes: usize,
}

impl HnswGraph {
    /// Create a new empty HNSW graph.
    pub fn new(config: HnswConfig) -> Self {
        HnswGraph {
            config,
            layers: Vec::new(),
            node_max_layer: Vec::new(),
            entry_point: None,
            max_layer: 0,
            num_nodes: 0,
        }
    }

    /// Number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.num_nodes
    }

    pub fn is_empty(&self) -> bool {
        self.num_nodes == 0
    }

    /// Number of layers in the graph.
    pub fn max_layer_count(&self) -> usize {
        self.max_layer + 1
    }

    /// Randomly assign a layer for a new node using exponential decay.
    fn random_level(&self) -> usize {
        let r: f64 = rand::random::<f64>();
        let level = (-r.ln() * self.config.ml).floor() as usize;
        level
    }

    /// Insert a new node (index `node_idx`) into the graph.
    ///
    /// `db` is the full RAM vector store so we can compute distances.
    /// `raw_vector` is the original float32 vector (if available) for high-quality LUT.
    pub fn insert(
        &mut self,
        node_idx: usize,
        db: &[PackedVector],
        index: &TurboIndex,
    ) {
        self.insert_with_raw(node_idx, db, index, None);
    }

    /// Insert with an optional raw vector for better graph quality.
    pub fn insert_with_raw(
        &mut self,
        node_idx: usize,
        db: &[PackedVector],
        index: &TurboIndex,
        raw_vector: Option<&[f32]>,
    ) {
        let node_level = self.random_level();

        // Ensure we have enough layers
        while self.layers.len() <= node_level {
            self.layers.push(Vec::new());
        }

        // Ensure all layers have room for this node
        for layer in self.layers.iter_mut() {
            while layer.len() <= node_idx {
                layer.push(Vec::new());
            }
        }

        // Track node's assigned level
        while self.node_max_layer.len() <= node_idx {
            self.node_max_layer.push(0);
        }
        self.node_max_layer[node_idx] = node_level;

        if self.entry_point.is_none() {
            // First node — just set it as entry point
            self.entry_point = Some(node_idx);
            self.max_layer = node_level;
            self.num_nodes += 1;
            return;
        }

        let ep = self.entry_point.unwrap();

        // Build LUT: use raw vector if available (much better), else reconstruct
        let lut = if let Some(raw) = raw_vector {
            build_lut(index, raw)
        } else {
            let approx = self.get_approx_vector(&db[node_idx], index);
            build_lut(index, &approx)
        };

        // Phase 1: Greedy descent from top layer to node_level + 1
        let mut current_ep = ep;
        let top = self.max_layer;
        let start_layer = if top > node_level { top } else { node_level };

        for layer in (node_level + 1..=start_layer).rev() {
            if layer < self.layers.len() {
                current_ep = self.greedy_closest(current_ep, &lut, db, index, layer);
            }
        }

        // Phase 2: Insert at each layer from node_level down to 0
        let mut entry_points = vec![current_ep];

        for layer in (0..=node_level.min(start_layer)).rev() {
            if layer >= self.layers.len() { continue; }

            let m_max = if layer == 0 { self.config.m_max0 } else { self.config.m };

            // Search for nearest neighbors at this layer
            let candidates = self.search_layer(
                &entry_points,
                &lut,
                db,
                index,
                layer,
                self.config.ef_construction,
            );

            // Select M closest as neighbors
            let neighbors: Vec<usize> = candidates.iter()
                .take(m_max)
                .map(|&(idx, _)| idx)
                .collect();

            // Add bidirectional edges
            // Ensure capacity
            while self.layers[layer].len() <= node_idx {
                self.layers[layer].push(Vec::new());
            }
            self.layers[layer][node_idx] = neighbors.clone();

            for &neighbor in &neighbors {
                while self.layers[layer].len() <= neighbor {
                    self.layers[layer].push(Vec::new());
                }
                self.layers[layer][neighbor].push(node_idx);

                // Prune if over capacity
                if self.layers[layer][neighbor].len() > m_max {
                    self.prune_neighbors(neighbor, m_max, &lut, db, index, layer);
                }
            }

            // Use neighbors as entry points for the next layer down
            entry_points = neighbors;
            if entry_points.is_empty() {
                entry_points = vec![current_ep];
            }
        }

        // Update entry point if this node is at a higher level
        if node_level > self.max_layer {
            self.entry_point = Some(node_idx);
            self.max_layer = node_level;
        }

        self.num_nodes += 1;
    }

    /// Search the HNSW graph for the `top_k` nearest neighbors of a query.
    ///
    /// `ef` controls the search beam width (higher = more accurate but slower).
    /// Recommended: ef >= top_k, typically ef = 50-200.
    pub fn search(
        &self,
        query: &[f32],
        index: &TurboIndex,
        db: &[PackedVector],
        top_k: usize,
        ef: usize,
        valid_ids: Option<&HashSet<u64>>,
    ) -> Vec<(u64, f32)> {
        if self.entry_point.is_none() || db.is_empty() {
            return vec![];
        }

        let lut = build_lut(index, query);
        let ep = self.entry_point.unwrap();

        // Phase 1: Greedy descent from top layer to layer 1
        let mut current_ep = ep;
        for layer in (1..=self.max_layer).rev() {
            if layer < self.layers.len() {
                current_ep = self.greedy_closest(current_ep, &lut, db, index, layer);
            }
        }

        // Phase 2: Expand search at layer 0 with beam width ef
        let candidates = self.search_layer(
            &[current_ep],
            &lut,
            db,
            index,
            0,
            ef.max(top_k),
        );

        // Apply metadata filter and return top_k
        let mut results: Vec<(u64, f32)> = candidates.into_iter()
            .filter(|&(idx, _)| {
                if let Some(valid) = valid_ids {
                    valid.contains(&db[idx].id)
                } else {
                    true
                }
            })
            .take(top_k)
            .map(|(idx, score)| (db[idx].id, score))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    /// Greedy search: from `ep`, follow the single best neighbor at `layer`.
    fn greedy_closest(
        &self,
        ep: usize,
        lut: &QueryLut,
        db: &[PackedVector],
        index: &TurboIndex,
        layer: usize,
    ) -> usize {
        let mut best = ep;
        let mut best_score = score_packed(lut, &db[ep], index.bits);

        let mut changed = true;
        while changed {
            changed = false;
            if layer < self.layers.len() && best < self.layers[layer].len() {
                for &neighbor in &self.layers[layer][best] {
                    if neighbor < db.len() {
                        let s = score_packed(lut, &db[neighbor], index.bits);
                        if s > best_score {
                            best_score = s;
                            best = neighbor;
                            changed = true;
                        }
                    }
                }
            }
        }

        best
    }

    /// Beam search at a single layer. Returns candidates sorted by score (descending).
    fn search_layer(
        &self,
        entry_points: &[usize],
        lut: &QueryLut,
        db: &[PackedVector],
        index: &TurboIndex,
        layer: usize,
        ef: usize,
    ) -> Vec<(usize, f32)> {
        if layer >= self.layers.len() {
            return vec![];
        }

        let mut visited: HashSet<usize> = HashSet::new();
        // We keep a simple working list and result list
        let mut candidates: Vec<(usize, f32)> = Vec::new(); // sorted best-first
        let mut results: Vec<(usize, f32)> = Vec::new();

        for &ep in entry_points {
            if ep >= db.len() { continue; }
            visited.insert(ep);
            let s = score_packed(lut, &db[ep], index.bits);
            candidates.push((ep, s));
            results.push((ep, s));
        }

        // Sort candidates best-first (highest score first)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut candidate_idx = 0;

        while candidate_idx < candidates.len() {
            let (c_idx, c_score) = candidates[candidate_idx];
            candidate_idx += 1;

            // Get worst result score
            let worst_result_score = results.iter()
                .map(|(_, s)| *s)
                .fold(f32::INFINITY, f32::min);

            // If we have enough results and the best unprocessed candidate
            // is worse than our worst result, we can stop
            if results.len() >= ef && c_score < worst_result_score {
                break;
            }

            // Expand neighbors
            if c_idx < self.layers[layer].len() {
                for &neighbor in &self.layers[layer][c_idx] {
                    if neighbor >= db.len() || visited.contains(&neighbor) {
                        continue;
                    }
                    visited.insert(neighbor);

                    let s = score_packed(lut, &db[neighbor], index.bits);

                    let should_add = if results.len() < ef {
                        true
                    } else {
                        s > worst_result_score
                    };

                    if should_add {
                        // Insert into candidates in sorted position
                        let pos = candidates[candidate_idx..].iter()
                            .position(|(_, cs)| s > *cs)
                            .map(|p| p + candidate_idx)
                            .unwrap_or(candidates.len());
                        candidates.insert(pos, (neighbor, s));

                        results.push((neighbor, s));

                        // Keep results at ef size
                        if results.len() > ef {
                            // Remove worst result
                            let worst_idx = results.iter()
                                .enumerate()
                                .min_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                                .map(|(i, _)| i)
                                .unwrap();
                            results.swap_remove(worst_idx);
                        }
                    }
                }
            }
        }

        // Sort results by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Prune a node's neighbor list to `m_max` by keeping only the closest.
    fn prune_neighbors(
        &mut self,
        node: usize,
        m_max: usize,
        _lut: &QueryLut,
        db: &[PackedVector],
        index: &TurboIndex,
        layer: usize,
    ) {
        if layer >= self.layers.len() || node >= self.layers[layer].len() {
            return;
        }

        let neighbors = &self.layers[layer][node];
        if neighbors.len() <= m_max { return; }

        // Score all neighbors against the node
        // Use the node's own vector to build a LUT for neighbor scoring
        let node_vec = self.get_approx_vector(&db[node], index);
        let node_lut = build_lut(index, &node_vec);

        let mut scored: Vec<(usize, f32)> = neighbors.iter()
            .filter(|&&n| n < db.len())
            .map(|&n| (n, score_packed(&node_lut, &db[n], index.bits)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(m_max);

        self.layers[layer][node] = scored.into_iter().map(|(idx, _)| idx).collect();
    }

    /// Approximate reconstruction of a vector from its compressed form.
    /// Used only for building LUTs when we need to score against a specific node.
    fn get_approx_vector(&self, packed: &PackedVector, index: &TurboIndex) -> Vec<f32> {
        let d = index.d;
        let indices = unpack_indices(&packed.mse_packed, d, index.bits);

        // Reconstruct in rotated space: y_hat[i] = centroid[indices[i]]
        let y_hat = DVector::from_fn(d, |i, _| index.centroids[indices[i] as usize]);

        // Un-rotate: x ≈ Πᵀ · ŷ
        let x_approx = &index.pi_t * &y_hat;

        x_approx.as_slice().to_vec()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage_engine::compress_vector;
    use crate::turbo_math::BitWidth;
    use rand::Rng;

    fn random_vector(d: usize, rng: &mut impl Rng) -> Vec<f32> {
        (0..d).map(|_| rng.gen::<f32>() - 0.5).collect()
    }

    #[test]
    fn test_hnsw_insert_and_search() {
        let d = 64;
        let n = 200;
        let index = TurboIndex::new(d, BitWidth::Bits4);
        let mut rng = rand::thread_rng();

        // Build compressed DB
        let mut raw_vectors: Vec<Vec<f32>> = Vec::new();
        let mut db: Vec<PackedVector> = Vec::new();
        for i in 0..n {
            let v = random_vector(d, &mut rng);
            db.push(compress_vector(&index, &v, i as u64));
            raw_vectors.push(v);
        }

        // Build HNSW graph (using raw vectors for high-quality edges)
        let config = HnswConfig { m: 16, m_max0: 32, ef_construction: 100, ..Default::default() };
        let mut graph = HnswGraph::new(config);

        for i in 0..n {
            graph.insert_with_raw(i, &db, &index, Some(&raw_vectors[i]));
        }

        assert_eq!(graph.len(), n);

        // Search for a known vector — it should be in the top results
        let query = &raw_vectors[42];
        let results = graph.search(query, &index, &db, 10, 200, None);

        assert!(!results.is_empty(), "HNSW search returned no results");
        let top_ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
        assert!(
            top_ids.contains(&42),
            "HNSW: Expected vector 42 in top-10, got: {:?}", top_ids
        );
    }

    #[test]
    fn test_hnsw_recall_vs_linear() {
        let d = 128;  // Higher dims = better quantization = better graph
        let n = 500;
        let index = TurboIndex::new(d, BitWidth::Bits4);
        let mut rng = rand::thread_rng();

        let mut raw_vectors: Vec<Vec<f32>> = Vec::new();
        let mut db: Vec<PackedVector> = Vec::new();
        for i in 0..n {
            let v = random_vector(d, &mut rng);
            db.push(compress_vector(&index, &v, i as u64));
            raw_vectors.push(v);
        }

        // Build graph (using raw vectors for high-quality edges)
        let config = HnswConfig { m: 16, m_max0: 32, ef_construction: 200, ..Default::default() };
        let mut graph = HnswGraph::new(config);
        for i in 0..n {
            graph.insert_with_raw(i, &db, &index, Some(&raw_vectors[i]));
        }

        // Compare recall: HNSW vs linear scan on 20 random queries
        let num_queries = 20;
        let top_k = 10;
        let mut total_overlap = 0;
        let mut total_possible = 0;

        for _ in 0..num_queries {
            let query = random_vector(d, &mut rng);

            // Linear scan (ground truth for compressed)
            let linear = crate::execution_engine::search_ram_store(
                &index, &db, &query, top_k, None,
            );
            let linear_ids: HashSet<u64> = linear.iter().map(|(id, _)| *id).collect();

            // HNSW search with high beam width
            let hnsw = graph.search(&query, &index, &db, top_k, 200, None);
            let hnsw_ids: HashSet<u64> = hnsw.iter().map(|(id, _)| *id).collect();

            let overlap = linear_ids.intersection(&hnsw_ids).count();
            total_overlap += overlap;
            total_possible += top_k;
        }

        let recall = total_overlap as f64 / total_possible as f64;
        eprintln!("HNSW recall vs linear scan: {:.1}%", recall * 100.0);
        assert!(
            recall >= 0.50,
            "HNSW recall too low: {:.1}% (expected >= 50%)", recall * 100.0
        );
    }

    #[test]
    fn test_build_lut_and_score() {
        let d = 64;
        let index = TurboIndex::new(d, BitWidth::Bits4);
        let mut rng = rand::thread_rng();

        let v = random_vector(d, &mut rng);
        let packed = compress_vector(&index, &v, 0);

        let lut = build_lut(&index, &v);
        let score = score_packed(&lut, &packed, index.bits);

        assert!(score.is_finite(), "LUT score should be finite");
        assert!(score > 0.0, "Self-score should be positive");
    }
}

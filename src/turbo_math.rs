//! # TurboMath — The TurboQuant Mathematical Engine
//!
//! Implements the core math from the TurboQuant paper:
//! - Random orthogonal rotation matrix Π (via QR decomposition)
//! - Random Gaussian projection matrix S (for QJL residuals)
//! - Hardcoded optimal centroids scaled by 1/√d

use nalgebra::DMatrix;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// The core index structure holding all precomputed matrices and centroids.
///
/// Once constructed for a given dimensionality `d`, this is immutable and
/// shared across all threads via `Arc<TurboIndex>`.
pub struct TurboIndex {
    /// Vector dimensionality (e.g., 1536 for OpenAI embeddings)
    pub d: usize,

    /// Π: d×d orthogonal rotation matrix (from QR decomposition of random Gaussian matrix).
    /// Rotating any vector by Π forces its coordinates into a Gaussian distribution,
    /// enabling data-oblivious quantization.
    pub pi: DMatrix<f32>,

    /// Πᵀ: Transpose of Π, precomputed for fast un-rotation during compression.
    pub pi_t: DMatrix<f32>,

    /// S: d×d random Gaussian projection matrix for the QJL residual stage.
    pub s: DMatrix<f32>,

    /// The 4 hardcoded optimal centroids for 2-bit quantization of a standard
    /// Gaussian, scaled by 1/√d to account for the concentration of measure.
    pub centroids: [f32; 4],
}

impl TurboIndex {
    /// Create a new TurboIndex for vectors of dimension `d`.
    ///
    /// This performs two d×d matrix generations and one QR decomposition.
    /// For d=1536, this takes a few seconds — it's a one-time startup cost.
    ///
    /// Uses a fixed seed for reproducibility. In production, you'd persist
    /// the matrices to disk so the index is deterministic across restarts.
    pub fn new(d: usize) -> Self {
        let seed: u64 = 42;
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0f32, 1.0f32).unwrap();

        // --- Step 1: Generate Π via QR decomposition ---
        // Fill a d×d matrix with N(0,1) entries
        let random_matrix = DMatrix::from_fn(d, d, |_, _| normal.sample(&mut rng));

        // QR decomposition: random_matrix = Q * R
        // Q is our orthogonal rotation matrix Π
        let qr = nalgebra::linalg::QR::new(random_matrix);
        let pi = qr.q();
        let pi_t = pi.transpose();

        // --- Step 2: Generate S (random Gaussian projection matrix) ---
        let s = DMatrix::from_fn(d, d, |_, _| normal.sample(&mut rng));

        // --- Step 3: Hardcoded optimal centroids ---
        // These values are the optimal 4-level scalar quantizer boundaries
        // for a standard Gaussian distribution, from the TurboQuant paper.
        // Scaled by 1/√d due to concentration of measure after rotation.
        let sqrt_d = (d as f32).sqrt();
        let centroids = [
            -1.51 / sqrt_d,
            -0.453 / sqrt_d,
            0.453 / sqrt_d,
            1.51 / sqrt_d,
        ];

        tracing::info!(
            "TurboIndex initialized: d={}, centroids=[{:.6}, {:.6}, {:.6}, {:.6}]",
            d, centroids[0], centroids[1], centroids[2], centroids[3]
        );

        TurboIndex {
            d,
            pi,
            pi_t,
            s,
            centroids,
        }
    }

    /// Find the index (0–3) of the nearest centroid for a given scalar value.
    #[inline]
    pub fn nearest_centroid(&self, value: f32) -> u8 {
        let mut best_idx: u8 = 0;
        let mut best_dist = f32::MAX;

        for (i, &c) in self.centroids.iter().enumerate() {
            let dist = (value - c).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = i as u8;
            }
        }
        best_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_qr_produces_orthogonal_matrix() {
        // Use a small dimension for fast testing
        let index = TurboIndex::new(64);

        // Q × Qᵀ should ≈ Identity
        let product = &index.pi * &index.pi_t;
        let identity: DMatrix<f32> = DMatrix::identity(64, 64);

        for i in 0..64 {
            for j in 0..64 {
                let val: f32 = product[(i, j)] - identity[(i, j)];
                let diff = val.abs();
                assert!(
                    diff < 1e-4,
                    "Q*Qᵀ not identity at ({}, {}): diff={}",
                    i, j, diff
                );
            }
        }
    }

    #[test]
    fn test_centroids_scale_with_sqrt_d() {
        let index_64 = TurboIndex::new(64);
        let index_256 = TurboIndex::new(256);

        // Centroid[3] for d=64: 1.51 / √64 = 1.51 / 8 = 0.188750
        let expected_64 = 1.51 / (64.0f32).sqrt();
        assert!((index_64.centroids[3] - expected_64).abs() < 1e-6);

        // Centroid[3] for d=256: 1.51 / √256 = 1.51 / 16 = 0.094375
        let expected_256 = 1.51 / (256.0f32).sqrt();
        assert!((index_256.centroids[3] - expected_256).abs() < 1e-6);

        // Higher dimension → smaller centroids
        assert!(index_256.centroids[3] < index_64.centroids[3]);
    }

    #[test]
    fn test_nearest_centroid() {
        let index = TurboIndex::new(64);
        // The most negative value should map to centroid 0
        assert_eq!(index.nearest_centroid(-10.0), 0);
        // The most positive value should map to centroid 3
        assert_eq!(index.nearest_centroid(10.0), 3);
        // Zero should map to centroid 1 or 2 (both equidistant, we get 1 due to < comparison)
        let mid = index.nearest_centroid(0.0);
        assert!(mid == 1 || mid == 2);
    }
}

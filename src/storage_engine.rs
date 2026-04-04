//! # Storage Engine — Vector Compression, Bit-Packing, and Database
//!
//! Implements:
//! - `PackedVector`: 3-bit compressed representation (2-bit MSE + 1-bit QJL)
//! - `compress_vector()`: Full TurboQuant compression pipeline
//! - `Database`: Two-tier RAM (compressed) + Disk (full precision) store

use crate::turbo_math::TurboIndex;
use nalgebra::DVector;

// ============================================================================
// Bit-Packing Helpers
// ============================================================================

/// Pack four 2-bit values (0–3) into a single byte.
/// Layout: `(a << 6) | (b << 4) | (c << 2) | d`
#[inline]
pub fn pack_mse_byte(a: u8, b: u8, c: u8, d: u8) -> u8 {
    (a << 6) | (b << 4) | (c << 2) | d
}

/// Unpack a single byte into four 2-bit values (0–3).
#[inline]
pub fn unpack_mse_byte(byte: u8) -> (u8, u8, u8, u8) {
    (
        (byte >> 6) & 0x03,
        (byte >> 4) & 0x03,
        (byte >> 2) & 0x03,
        byte & 0x03,
    )
}

/// Pack eight 1-bit boolean values into a single byte.
/// Layout: `(b0 << 7) | (b1 << 6) | ... | b7`
#[inline]
pub fn pack_qjl_byte(bits: &[u8; 8]) -> u8 {
    (bits[0] << 7)
        | (bits[1] << 6)
        | (bits[2] << 5)
        | (bits[3] << 4)
        | (bits[4] << 3)
        | (bits[5] << 2)
        | (bits[6] << 1)
        | bits[7]
}

/// Unpack a single byte into eight 1-bit boolean values (0 or 1).
#[inline]
pub fn unpack_qjl_byte(byte: u8) -> [u8; 8] {
    [
        (byte >> 7) & 1,
        (byte >> 6) & 1,
        (byte >> 5) & 1,
        (byte >> 4) & 1,
        (byte >> 3) & 1,
        (byte >> 2) & 1,
        (byte >> 1) & 1,
        byte & 1,
    ]
}

// ============================================================================
// PackedVector
// ============================================================================

/// A compressed vector using TurboQuant's 3-bit encoding.
///
/// - `mse_bits`: 2 bits per dimension, packed 4-per-byte → `d/4` bytes
/// - `qjl_bits`: 1 bit per dimension, packed 8-per-byte → `d/8` bytes
/// - `residual_norm`: γ = ||r||₂, the L2 norm of the quantization residual
///
/// Total storage: ~3 bits per dimension = **16x compression** vs Float32.
#[derive(Clone, Debug)]
pub struct PackedVector {
    pub id: u64,
    pub mse_bits: Vec<u8>,     // Size: d / 4
    pub qjl_bits: Vec<u8>,     // Size: d / 8
    pub residual_norm: f32,    // γ = ||r||₂
}

// ============================================================================
// Compression Pipeline
// ============================================================================

/// L2-normalize a vector in-place. Returns the original norm.
pub fn l2_normalize(v: &mut DVector<f32>) -> f32 {
    let norm = v.norm();
    if norm > 1e-10 {
        *v /= norm;
    }
    norm
}

/// Compress a single vector using the full TurboQuant pipeline.
///
/// Implements MID Section 4B:
/// 1. L2-normalize the input
/// 2. Rotate: y = Π · x
/// 3. Quantize each coordinate to nearest centroid → 2-bit indices
/// 4. Bit-pack indices into mse_bits
/// 5. Reconstruct → un-rotate → compute residual r
/// 6. Project residual: z = S · r, extract sign bits
/// 7. Bit-pack signs into qjl_bits
pub fn compress_vector(index: &TurboIndex, vector: &[f32], id: u64) -> PackedVector {
    let d = index.d;
    assert_eq!(vector.len(), d, "Vector dimension mismatch: expected {}, got {}", d, vector.len());

    // Step 1: L2-normalize
    let mut x = DVector::from_column_slice(vector);
    l2_normalize(&mut x);

    // Step 2: Rotate — y = Π · x
    let y = &index.pi * &x;

    // Step 3: Quantize each coordinate to nearest centroid
    let mut indices = Vec::with_capacity(d);
    for i in 0..d {
        indices.push(index.nearest_centroid(y[i]));
    }

    // Step 4: Bit-pack MSE indices (4 values per byte)
    let mse_byte_count = d / 4;
    let mut mse_bits = Vec::with_capacity(mse_byte_count);
    for chunk in indices.chunks(4) {
        mse_bits.push(pack_mse_byte(chunk[0], chunk[1], chunk[2], chunk[3]));
    }

    // Step 5a: Reconstruct quantized vector in rotated space
    let y_hat = DVector::from_fn(d, |i, _| index.centroids[indices[i] as usize]);

    // Step 5b: Un-rotate to get MSE approximation: x̃_mse = Πᵀ · ŷ
    let x_mse = &index.pi_t * &y_hat;

    // Step 5c: Residual: r = x - x̃_mse
    let residual = &x - &x_mse;

    // Step 5d: γ = ||r||₂
    let gamma = residual.norm();

    // Step 6: Project residual through S: z = S · r
    let z = &index.s * &residual;

    // Step 7: Extract sign bits and bit-pack into qjl_bits
    let qjl_byte_count = d / 8;
    let mut qjl_bits = Vec::with_capacity(qjl_byte_count);
    for chunk_start in (0..d).step_by(8) {
        let mut bits = [0u8; 8];
        for j in 0..8 {
            bits[j] = if z[chunk_start + j] > 0.0 { 1 } else { 0 };
        }
        qjl_bits.push(pack_qjl_byte(&bits));
    }

    PackedVector {
        id,
        mse_bits,
        qjl_bits,
        residual_norm: gamma,
    }
}

// ============================================================================
// Database (RAM + Disk)
// ============================================================================

/// Two-tier database: compressed vectors in RAM, full-precision vectors on disk.
pub struct Database {
    /// Tier 1: Compressed 3-bit vectors in RAM for fast approximate search.
    pub ram: Vec<PackedVector>,

    /// Tier 2: Full-precision Float32 vectors on disk (sled) for exact re-ranking.
    pub disk: sled::Db,
}

impl Database {
    /// Create or open the database at the given path.
    pub fn new(path: &str) -> Result<Self, sled::Error> {
        let disk = sled::open(path)?;

        tracing::info!("Database opened at: {}", path);

        Ok(Database {
            ram: Vec::new(),
            disk,
        })
    }

    /// Insert a vector into both tiers of the database.
    ///
    /// 1. Compress the vector and push to RAM store
    /// 2. Save the raw float bytes to disk (sled) keyed by ID
    pub fn insert(
        &mut self,
        index: &TurboIndex,
        vector: &[f32],
        id: u64,
    ) -> Result<(), sled::Error> {
        // Tier 1: Compress and store in RAM
        let packed = compress_vector(index, vector, id);
        self.ram.push(packed);

        // Tier 2: Store raw float bytes on disk
        let key = id.to_be_bytes();
        let value: Vec<u8> = vector
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        self.disk.insert(key, value)?;

        Ok(())
    }

    /// Retrieve the full-precision vector from disk by ID.
    pub fn get_raw_vector(&self, id: u64, d: usize) -> Option<DVector<f32>> {
        let key = id.to_be_bytes();
        match self.disk.get(key) {
            Ok(Some(bytes)) => {
                let floats: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|chunk| {
                        let arr: [u8; 4] = chunk.try_into().unwrap();
                        f32::from_le_bytes(arr)
                    })
                    .collect();
                if floats.len() == d {
                    Some(DVector::from_vec(floats))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get the number of vectors stored.
    pub fn len(&self) -> usize {
        self.ram.len()
    }

    /// Check if the database is empty.
    pub fn is_empty(&self) -> bool {
        self.ram.is_empty()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turbo_math::TurboIndex;

    #[test]
    fn test_mse_bit_packing_roundtrip() {
        // Pack [3, 1, 2, 0] into a byte
        let packed = pack_mse_byte(3, 1, 2, 0);
        let (a, b, c, d) = unpack_mse_byte(packed);
        assert_eq!((a, b, c, d), (3, 1, 2, 0));

        // Pack [0, 0, 0, 0]
        let packed = pack_mse_byte(0, 0, 0, 0);
        let (a, b, c, d) = unpack_mse_byte(packed);
        assert_eq!((a, b, c, d), (0, 0, 0, 0));

        // Pack [3, 3, 3, 3]
        let packed = pack_mse_byte(3, 3, 3, 3);
        let (a, b, c, d) = unpack_mse_byte(packed);
        assert_eq!((a, b, c, d), (3, 3, 3, 3));
    }

    #[test]
    fn test_qjl_bit_packing_roundtrip() {
        let bits: [u8; 8] = [1, 0, 1, 1, 0, 0, 1, 0];
        let packed = pack_qjl_byte(&bits);
        let unpacked = unpack_qjl_byte(packed);
        assert_eq!(unpacked, bits);

        // All zeros
        let bits: [u8; 8] = [0; 8];
        let packed = pack_qjl_byte(&bits);
        let unpacked = unpack_qjl_byte(packed);
        assert_eq!(unpacked, bits);

        // All ones
        let bits: [u8; 8] = [1; 8];
        let packed = pack_qjl_byte(&bits);
        assert_eq!(packed, 0xFF);
        let unpacked = unpack_qjl_byte(packed);
        assert_eq!(unpacked, bits);
    }

    #[test]
    fn test_compress_vector_sizes() {
        let d = 64;
        let index = TurboIndex::new(d);

        // Create a random-ish vector
        let vector: Vec<f32> = (0..d).map(|i| (i as f32) * 0.01 + 0.1).collect();
        let packed = compress_vector(&index, &vector, 42);

        // Check packed sizes
        assert_eq!(packed.id, 42);
        assert_eq!(packed.mse_bits.len(), d / 4);  // 16 bytes for d=64
        assert_eq!(packed.qjl_bits.len(), d / 8);  // 8 bytes for d=64
        assert!(packed.residual_norm >= 0.0);
        assert!(packed.residual_norm.is_finite());
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = DVector::from_vec(vec![3.0, 4.0]);
        let norm = l2_normalize(&mut v);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }
}

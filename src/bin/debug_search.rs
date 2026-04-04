//! Quick debug script to find the bit-packing bug

use tvectordb::storage_engine::{pack_indices, unpack_indices, compress_vector};
use tvectordb::turbo_math::{BitWidth, TurboIndex};

fn main() {
    // Test 1: Pack/unpack roundtrip for 3-bit with real dimension
    println!("=== Test 1: Pack/Unpack Roundtrip ===");
    
    for bits in [BitWidth::Bits2, BitWidth::Bits3, BitWidth::Bits4] {
        let d = 64;
        let max_val = bits.num_levels() as u8;
        let indices: Vec<u8> = (0..d).map(|i| (i as u8) % max_val).collect();
        
        let packed = pack_indices(&indices, bits);
        let unpacked = unpack_indices(&packed, d, bits);
        
        let matches = indices.iter().zip(unpacked.iter()).all(|(a, b)| a == b);
        println!("{}-bit: pack/unpack match = {} (packed {} bytes for {} values)",
            bits.bits(), matches, packed.len(), d);
        
        if !matches {
            for i in 0..d.min(20) {
                if indices[i] != unpacked[i] {
                    println!("  MISMATCH at {}: expected {}, got {}", i, indices[i], unpacked[i]);
                }
            }
        }
    }
    
    // Test 2: Compress + search a single known vector
    println!("\n=== Test 2: Single Vector Self-Search ===");
    
    for bits in [BitWidth::Bits2, BitWidth::Bits3, BitWidth::Bits4] {
        let d = 64;
        let index = TurboIndex::new(d, bits);
        
        // Create a simple vector
        let vector: Vec<f32> = (0..d).map(|i| (i as f32) / d as f32).collect();
        
        // Compress it
        let packed = compress_vector(&index, &vector, 0);
        
        // Now search for it
        let results = tvectordb::execution_engine::search_ram_store(
            &index, &[packed.clone()], &vector, 1
        );
        
        if let Some((id, score)) = results.first() {
            println!("{}-bit: id={}, score={:.4}", bits.bits(), id, score);
        } else {
            println!("{}-bit: NO RESULT!", bits.bits());
        }
    }
    
    // Test 3: Check if quantize produces correct range
    println!("\n=== Test 3: Quantize Range Check ===");
    
    for bits in [BitWidth::Bits2, BitWidth::Bits3, BitWidth::Bits4] {
        let d = 64;
        let index = TurboIndex::new(d, bits);
        let max_level = bits.num_levels() as u8;
        
        // Check boundaries
        let test_vals = vec![-10.0, -0.1, 0.0, 0.1, 10.0];
        let mut all_valid = true;
        for v in &test_vals {
            let idx = index.quantize(*v);
            if idx >= max_level {
                println!("  {}-bit: quantize({}) = {} (OVERFLOW, max={})", bits.bits(), v, idx, max_level - 1);
                all_valid = false;
            }
        }
        println!("{}-bit: all quantized values in range = {}", bits.bits(), all_valid);
    }
}

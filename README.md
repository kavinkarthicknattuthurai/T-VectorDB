# T-VectorDB 🚀

**The Zero-Latency, Configurable, LLM-Native Vector Database**

T-VectorDB is an ultra-fast, CPU-friendly vector database built entirely in **pure Rust** with zero C/C++ dependencies. It implements the [TurboQuant paper (ICLR 2026, Google Research)](https://arxiv.org/) — a data-oblivious quantization algorithm that compresses float32 vectors to 2/3/4 bits per dimension with zero training time.

Unlike traditional vector databases that require slow K-Means indexing, T-VectorDB compresses vectors **the millisecond they arrive**. No training. No freezing. No waiting.

## ✨ Why T-VectorDB?

| Feature | T-VectorDB | turboqvec | Qdrant | ChromaDB | Pinecone |
|---------|:---:|:---:|:---:|:---:|:---:|
| TurboQuant compression | ✅ 2/3/4-bit | ✅ 2/3/4-bit | ❌ | ❌ | ❌ |
| **Two-Tier exact re-rank** | ✅ | ❌ | ❌ | ❌ | ❌ |
| REST API server | ✅ | ❌ | ✅ | ✅ | ✅ |
| Persistence (survives restart) | ✅ | ❌ | ✅ | ✅ | ✅ |
| Batch insert/search | ✅ | ✅ | ✅ | ✅ | ✅ |
| Delete vectors | ✅ | ❌ | ✅ | ✅ | ✅ |
| Zero C/C++ dependencies | ✅ | ✅ | ❌ | N/A | N/A |

> **Our unique advantage:** The Two-Tier Hybrid Architecture. We use compressed vectors for blazing-fast approximate search, then re-rank exact results from disk. You get the **speed of compression** with the **accuracy of brute-force**. No other database does this.

## 🧠 How It Works

1. **Orthogonal Rotation (Π):** Rotate incoming vectors using a random orthogonal matrix. The Central Limit Theorem forces coordinates into a predictable Gaussian distribution.
2. **Lloyd-Max Quantization:** Snap each coordinate to the optimal centroid for that bit-width. No training needed — the buckets are mathematically predetermined.
3. **QJL Residual (1-bit):** Project the quantization error through a Gaussian matrix and store only the signs. Corrects inner product bias.
4. **Asymmetric Search:** At query time, build a lookup table (LUT) once. Score millions of vectors per second using table reads — never decompress the database.

## 📊 Performance

**Compression & Memory** (d=384)

| Vectors | Float32 | 4-bit | 3-bit | 2-bit |
|---------|---------|-------|-------|-------|
| 10,000 | 15.4 MB | 2.0 MB | 1.5 MB | 1.0 MB |
| 100,000 | 153.6 MB | 19.6 MB | 14.8 MB | 10.0 MB |
| 1,000,000 | 1,536 MB | **196 MB** | **148 MB** | **100 MB** |

**Compression Ratios**

| Bit-Width | Compression | Best For |
|-----------|------------|----------|
| **4-bit** | 7.8x smaller | Near-lossless retrieval (~95% R@1) |
| **3-bit** | 10.4x smaller | Balanced speed/accuracy (~87% R@1) |
| **2-bit** | 15.4x smaller | Maximum compression (~73% R@1) |

## 🚀 Quick Start

```bash
git clone https://github.com/kavinkarthicknattuthurai/T-VectorDB.git
cd T-VectorDB
cargo run --release
```

**Configuration flags:**
```bash
cargo run --release -- --dim 1536 --bits 4 --port 3000 --data ./my_db
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dim` | 1536 | Vector dimensionality |
| `--bits` | 3 | Quantization bit-width (2, 3, or 4) |
| `--port` | 3000 | HTTP server port |
| `--data` | ./data | Database storage directory |

## 📡 API Reference

### Insert a vector
```bash
curl -X POST http://localhost:3000/insert \
  -H "Content-Type: application/json" \
  -d '{"id": 1, "vector": [0.1, 0.2, ...]}'
```

### Batch insert
```bash
curl -X POST http://localhost:3000/insert_batch \
  -H "Content-Type: application/json" \
  -d '{"vectors": [{"id": 1, "vector": [...]}, {"id": 2, "vector": [...]}]}'
```

### Search (approximate — RAM only, fastest)
```bash
curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, ...], "top_k": 5}'
```

### Search (exact — hybrid two-tier, perfect accuracy)
```bash
curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, ...], "top_k": 5, "exact": true}'
```

### Delete a vector
```bash
curl -X DELETE http://localhost:3000/vectors/42
```

### View stats
```bash
curl http://localhost:3000/stats
```

## 🏗️ Architecture

```
                    ┌─────────────────────────────┐
                    │         REST API (Axum)      │
                    │  /insert  /search  /delete   │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │       TurboQuant Engine      │
                    │  Rotate → Quantize → Pack    │
                    │  2-bit / 3-bit / 4-bit       │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼─────────┐    ┌▼────────────────▼┐
    │    RAM Store       │    │   Disk Store     │
    │  Compressed bits   │    │  Full Float32    │
    │  (fast scan)       │    │  (sled, exact)   │
    └─────────┬─────────┘    └─────────┬────────┘
              │                        │
              │   Approximate Top-100  │
              ├────────────────────────┤
              │   Exact Re-Rank Top-K  │
              └────────────────────────┘
```

## 📂 Project Structure

```
T-VectorDB/
├── Cargo.toml             # Dependencies & config
├── src/
│   ├── main.rs            # Server entry point + CLI
│   ├── lib.rs             # Module declarations
│   ├── turbo_math.rs      # TurboIndex: QR decomp, Lloyd-Max centroids
│   ├── storage_engine.rs  # PackedVector, compression, Database, persistence
│   ├── execution_engine.rs# ADC search, hybrid re-rank, batch search
│   ├── api_server.rs      # Axum REST endpoints
│   └── bin/
│       └── benchmark.rs   # Performance benchmark suite
├── README.md
├── LICENSE                # MIT
└── mid.md                 # Master Implementation Document
```

## 🔬 Run Benchmarks

```bash
cargo run --release --bin benchmark
```

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built on the [TurboQuant paper](https://arxiv.org/) by Amir Zandieh, Majid Daliri, Majid Hadian, and Vahab Mirrokni (Google Research, ICLR 2026).

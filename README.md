# T-VectorDB 🚀

**The Zero-Latency, Configurable, LLM-Native Vector Database**

T-VectorDB is an ultra-fast vector database built in **pure Rust** with zero C/C++ dependencies. It implements the [TurboQuant paper (Google Research, ICLR 2026)](https://arxiv.org/) — a data-oblivious quantization algorithm that compresses float32 vectors to 2/3/4 bits per dimension with zero training.

Unlike traditional databases that require slow global locks or K-Means warmups, T-VectorDB compresses and indexes vectors **the millisecond they arrive**.

## ✨ Feature Comparison

| Feature | T-VectorDB V3 | turboqvec | Qdrant | Pinecone |
|---------|:---:|:---:|:---:|:---:|
| TurboQuant 2/3/4-bit | ✅ | ✅ | ❌ | ❌ |
| **HNSW Graph Index** | ✅ Compressed | ❌ | ✅ | ✅ |
| **Hybrid Re-Rank (100% Accuracy)** | ✅ | ❌ | ❌ | ❌ |
| **Binary gRPC Protocol** | ✅ Zero-Copy | ❌ | ✅ | ✅ |
| **Lock-Free Concurrency** | ✅ | ❌ | ✅ | ✅ |
| **O(1) Metadata Filtering** | ✅ | ❌ | ✅ | ✅ |
| **Python SDK** | ✅ Native gRPC | ❌ | ✅ | ✅ |
| **REST API + CORS** | ✅ | ❌ | ✅ | ✅ |
| **Zero-Downtime Tombstone Deletes** | ✅ | ❌ | ❌ | ✅ |
| **Graceful Shutdown (Sled/HNSW)** | ✅ | ❌ | ❌ | ❌ |
| Zero C/C++ Dependencies | ✅ | ✅ | ❌ | N/A |

## 🧠 How It Works

1. **Orthogonal Rotation (Π):** Rotate incoming vectors using a random orthogonal matrix. The Central Limit Theorem forces coordinates into a predictable Gaussian shape.
2. **Lloyd-Max Quantization:** Snap each coordinate to the mathematically optimal centroid for 2/3/4-bit precision. No training needed — buckets are predetermined.
3. **QJL Residual (1-bit):** Project quantization error through a Gaussian matrix and store only the signs. Corrects inner product bias.
4. **HNSW Graph on Compressed Data:** Vectors are placed into a Hierarchical Navigable Small World graph. At query time, we walk the graph using *compressed* approximate distances (LUTs). This shifts search time from $O(n)$ to **$O(\log n)$**, enabling millisecond searches on millions of vectors natively in RAM.

## 🎯 Hybrid Two-Tier Search — 100% Accuracy Mode

This is our **unique advantage** that no other vector database offers:

```
 ┌──────────────────────────────────────────────────────────────────┐
 │  Client sends: {"exact": true}                                  │
 │                                                                  │
 │                                                                  │
 │  Stage 1: RAM Approximate Scan (HNSW)                          │
 │  ├─ Traverses the HNSW graph using LUTs (sub-millisecond)      │
 │  ├─ Produces Top-100 shortlist in O(log n) time                │
 │  └─ Uses 2/3/4-bit packed memory (10x less RAM)                │
 │                                                                  │
 │  Stage 2: Disk Exact Re-Rank                                    │
 │  ├─ Loads ONLY the Top-100 full Float32 vectors from sled       │
 │  ├─ Computes exact cosine similarity dot products               │
 │  └─ Returns the mathematically perfect Top-K                    │
 │                                                                  │
 │  Result: Speed of compression + Accuracy of brute-force = 100%  │
 └──────────────────────────────────────────────────────────────────┘
```

Use `"exact": false` (default) for blazing-fast approximate results, or `"exact": true` when you need guaranteed correctness.

## 🎛️ Choosing Your Bit-Width

Switch bit-widths on the command line. Each tier serves a different production use case:

| Tier | CLI Flag | Compression | Memory (1M × 384d) | Recall@10 | Best For |
|------|----------|-------------|---------------------|-----------|----------|
| **4-bit** | `--bits 4` | 6.1x | 2.4 GB → 390 MB | **88.4%** | RAG chatbots, document search, production retrieval |
| **3-bit** | `--bits 3` | 7.5x | 2.4 GB → 320 MB | **79.1%** | Agent memory, context windows, balanced workloads |
| **2-bit** | `--bits 2` | 9.8x | 2.4 GB → 245 MB | **63.2%** | Analytics, clustering, weak supervision at scale |

> **Pro tip:** Use 4-bit with `"exact": true` for 100% accurate retrieval at 6x compression. This is the sweet spot most production teams should start with.

## 🚀 Quick Start

```bash
git clone https://github.com/kavinkarthicknattuthurai/T-VectorDB.git
cd T-VectorDB
cargo run --release
```

**Configuration:**
```bash
# Default: 1536-dim, 3-bit, REST on 3000, gRPC on 50051
cargo run --release

# OpenAI embeddings (1536-dim, 4-bit for near-lossless)
cargo run --release -- --dim 1536 --bits 4

# BGE-small embeddings (384-dim, 2-bit for max compression)
cargo run --release -- --dim 384 --bits 2 --port 8080
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dim` | 1536 | Vector dimensionality |
| `--bits` | 3 | Quantization precision (2, 3, or 4) |
| `--port` | 3000 | REST API port |
| `--grpc-port` | 50051 | Binary gRPC port |
| `--data` | ./data | Persistent storage directory |

## 🐳 Docker Deployment (Production)

For production environments (AWS, GCP, Kubernetes), you don't need to install Rust. You can run T-VectorDB as an isolated, hyper-optimized microservice container.

**1. Build the container:**
```bash
docker build -t tvectordb .
```

**2. Run with Persistent Storage:**
```bash
# Maps local ./data folder to the container to ensure vectors survive reboots
# Exposes both REST (3000) and gRPC (50051) ports
docker run -d \
  -p 3000:3000 -p 50051:50051 \
  -v $(pwd)/data:/app/data \
  tvectordb
```

**3. Advanced Configuration Override:**
We highly recommend using Environment Variables (`-e`) to configure the database in Docker, as this prevents clashes with the internal volumes:
```bash
docker run -p 3000:3000 -p 50051:50051 \
  -e TVECTORDB_DIM=384 \
  -e TVECTORDB_BITS=4 \
  tvectordb
```
Alternatively, you can pass CLI flags which append to the entrypoint:
```bash
docker run -p 3000:3000 -p 50051:50051 tvectordb --dim 384 --bits 4
```

## 🐍 Python SDK

```bash
cd clients/python && pip install -e .
```

```python
from tvectordb import TVectorClient

client = TVectorClient("localhost:50051")  # Binary gRPC, not REST

# Insert with metadata tags
client.insert(id=1, vector=embedding, metadata={"category": "finance"})

# Fast approximate search
results = client.search(vector=query, top_k=5)

# 100% accurate hybrid search with metadata filter
results = client.search(
    vector=query, 
    top_k=5, 
    exact=True,              # Activates Two-Tier Re-Rank
    filter={"category": "finance"}  # O(1) pre-filter
)

# Advanced: Tune HNSW search beam width (higher = more accurate but slower)
results = client.search(
    vector=query, 
    top_k=5, 
    ef_search=200            # Default is 100
)

# Batch operations
client.insert_batch([(1, vec1, {"tag": "a"}), (2, vec2, None)])
client.delete(id=1)
```

## 📡 REST API

### Insert with metadata
```bash
curl -X POST http://localhost:3000/insert \
  -H "Content-Type: application/json" \
  -d '{"id": 1, "vector": [0.1, 0.2, ...], "metadata": {"status": "active"}}'
```

### Search with filter
```bash
curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, ...], "top_k": 5, "filter": {"status": "active"}}'
```

### Exact hybrid search (100% accuracy)
```bash
curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, ...], "top_k": 5, "exact": true}'
```

### Other endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/stats` | Database stats + memory usage |
| POST | `/insert_batch` | Insert multiple vectors |
| POST | `/search_batch` | Multiple queries at once |
| DELETE | `/vectors/{id}` | Delete by ID |

## 🏗️ Architecture

```
                  ┌─────────────────────────────┐
                  │      Client Applications     │
                  │  Python SDK · curl · Go · JS │
                  └──────┬──────────────┬────────┘
                   gRPC TCP        JSON HTTP/1.1
                  ┌──────▼──────┐ ┌─────▼────────┐
                  │ Tonic (PB)  │ │  Axum (REST) │
                  │ Port 50051  │ │  Port 3000   │
                  └──────┬──────┘ └─────┬────────┘
                         │              │
                  ┌──────▼──────────────▼────────┐
                  │     Lock-Free TurboQuant      │
                  │   O(1) Metadata Pre-Filter    │
                  │   Rotate → Quantize → Pack    │
                  │     Insert into HNSW Graph    │
                  └──────┬──────────────┬────────┘
                         │              │
                  ┌──────▼──────┐ ┌─────▼────────┐
                  │  RAM Store  │ │  Sled (Disk)  │
                  │ HNSW Index  │ │ Float32 + Meta│
                  │ (O(log n))  │ │ (exact vals)  │
                  └─────────────┘ └──────────────┘
```

## 📂 Project Structure

```
T-VectorDB/
├── Cargo.toml                  # Rust dependencies
├── build.rs                    # Protobuf compiler (vendored)
├── proto/tvectordb.proto       # gRPC schema definition
├── src/
│   ├── main.rs                 # Server entry + CLI
│   ├── lib.rs                  # Module declarations
│   ├── turbo_math.rs           # TurboIndex: QR, Lloyd-Max, centroids
│   ├── storage_engine.rs       # PackedVector, Database, inverted index
│   ├── execution_engine.rs     # ADC search, hybrid re-rank, batch
│   ├── api_server.rs           # Axum REST endpoints
│   ├── grpc_server.rs          # Tonic gRPC binary endpoints
│   └── bin/
│       ├── benchmark.rs        # Performance benchmark suite
│       └── validate_recall.rs  # Scientific accuracy validation
├── clients/
│   └── python/                 # Python SDK (pip install)
│       ├── tvectordb/          # Package with gRPC client
│       ├── examples/           # Quickstart demo
│       └── setup.py
├── README.md
└── LICENSE                     # MIT
```

## 🔬 Scientific Validation

Real numbers from our test suite — not marketing claims:

```bash
cargo run --release --bin validate_recall
cargo run --release --bin benchmark
```

**Recall Accuracy** (d=384, 1000 vectors, 200 queries):

| Bits | Recall@1 | Recall@5 | Recall@10 | Cosine Error |
|------|----------|----------|-----------|--------------|
| 4-bit | 80.5% | 86.5% | 88.4% | 0.0043 |
| 3-bit | 73.5% | 76.3% | 79.1% | 0.0073 |
| 2-bit | 54.5% | 59.1% | 63.2% | 0.0144 |

**Search Performance** (d=384, 10,000 vectors):

| Bits | Avg Search | QPS | Self-Recall | Compression |
|------|-----------|-----|-------------|-------------|
| 4-bit | 10.0 ms | 100 | 100% | 6.1x |
| 3-bit | 10.7 ms | 93 | 100% | 7.5x |
| 2-bit | 9.5 ms | 106 | 100% | 9.8x |

## 📜 License

MIT License. See [LICENSE](LICENSE).

Built on the [TurboQuant paper](https://arxiv.org/) by Vahab Mirrokni et al. (Google Research, ICLR 2026).

# T-VectorDB Python SDK

Native gRPC client for [T-VectorDB](https://github.com/kavinkarthicknattuthurai/T-VectorDB) — the zero-latency TurboQuant vector database.

This SDK communicates over the **binary Protobuf TCP protocol** (not REST). It's the fastest way to interact with T-VectorDB from Python.

## Install

```bash
cd clients/python
pip install -e .
```

## Usage

```python
from tvectordb import TVectorClient

client = TVectorClient("localhost:50051")

# Insert with metadata
client.insert(id=1, vector=[0.1, 0.2, ...], metadata={"type": "invoice"})

# Fast approximate search
results = client.search(vector=[0.1, ...], top_k=5)

# 100% accurate hybrid search + metadata filter
results = client.search(vector=[0.1, ...], top_k=5, exact=True, filter={"type": "invoice"})

# Batch insert
client.insert_batch([(1, [0.1, ...], {"tag": "a"}), (2, [0.2, ...], None)])

# Delete
client.delete(id=1)
```

## Requirements

- Python 3.8+
- `grpcio >= 1.50.0`
- A running T-VectorDB server with gRPC enabled (port 50051)

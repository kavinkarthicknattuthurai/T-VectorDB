"""
T-VectorDB Python SDK — Quickstart Example

Prerequisites:
  1. Start the T-VectorDB server:
     cd T-VectorDB && cargo run --release -- --dim 384

  2. Install the SDK:
     cd clients/python && pip install -e .

  3. Run this script:
     python examples/quickstart.py
"""

import random
from tvectordb import TVectorClient

DIM = 384  # Must match the server's --dim flag

def random_vector(d: int) -> list:
    """Generate a random vector (simulating an embedding)."""
    return [random.gauss(0, 1) for _ in range(d)]

def main():
    # Connect to the Rust server over binary gRPC (port 50051)
    client = TVectorClient("localhost:50051")
    print("✅ Connected to T-VectorDB\n")

    # ─── 1. Insert vectors with metadata ────────────────────────────
    print("📥 Inserting vectors with metadata...")
    
    categories = ["finance", "health", "tech", "science", "sports"]
    for i in range(100):
        client.insert(
            id=i,
            vector=random_vector(DIM),
            metadata={
                "category": categories[i % len(categories)],
                "source": "quickstart",
            }
        )
    print(f"   Inserted 100 vectors across {len(categories)} categories\n")

    # ─── 2. Approximate Search (Fast, RAM-only) ─────────────────────
    print("🔍 Approximate search (RAM-only, fastest):")
    query = random_vector(DIM)
    results = client.search(vector=query, top_k=5)
    for r in results:
        print(f"   ID: {r['id']:>4}  Score: {r['score']:.4f}")

    # ─── 3. Exact Hybrid Search (100% Accuracy) ─────────────────────
    print("\n🎯 Exact hybrid search (Two-Tier Re-Rank, 100% accuracy):")
    results = client.search(vector=query, top_k=5, exact=True)
    for r in results:
        print(f"   ID: {r['id']:>4}  Score: {r['score']:.4f}")

    # ─── 4. Filtered Search (Metadata) ──────────────────────────────
    print("\n🏷️  Filtered search (category='finance' only):")
    results = client.search(
        vector=query,
        top_k=5,
        filter={"category": "finance"}
    )
    for r in results:
        print(f"   ID: {r['id']:>4}  Score: {r['score']:.4f}")

    # ─── 5. Batch Insert ────────────────────────────────────────────
    print("\n📦 Batch inserting 50 more vectors...")
    batch = [
        (1000 + i, random_vector(DIM), {"category": "batch"})
        for i in range(50)
    ]
    result = client.insert_batch(batch)
    print(f"   Inserted: {result['inserted']}, Total: {result['total_vectors']}")

    # ─── 6. Delete ──────────────────────────────────────────────────
    print("\n🗑️  Deleting vector ID 0...")
    result = client.delete(id=0)
    print(f"   Deleted: {result['deleted']}")

    print("\n✅ Quickstart complete! T-VectorDB is working perfectly. 🚀")
    client.close()


if __name__ == "__main__":
    main()

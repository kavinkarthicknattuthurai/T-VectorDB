"""
T-VectorDB Python Client — High-Performance gRPC Interface

Connects to the T-VectorDB Rust server over the binary gRPC protocol (port 50051).
This is NOT a REST wrapper — it speaks raw Protobuf TCP for maximum throughput.

Usage:
    from tvectordb import TVectorClient

    client = TVectorClient("localhost:50051")
    client.insert(id=1, vector=[0.1, 0.2, ...], metadata={"type": "invoice"})
    results = client.search(vector=[0.1, ...], top_k=5, filter={"type": "invoice"})
"""

import grpc
from typing import List, Dict, Optional, Tuple

# Import the generated protobuf stubs
from . import tvectordb_pb2
from . import tvectordb_pb2_grpc


class TVectorClient:
    """Native gRPC client for T-VectorDB.
    
    Communicates over binary Protobuf TCP — no JSON overhead.
    Thread-safe: a single client can be shared across threads.
    
    Args:
        host: Server address in "host:port" format (default: "localhost:50051")
        secure: Use TLS encryption (default: False for local development)
    """

    def __init__(self, host: str = "localhost:50051", secure: bool = False):
        self.host = host
        if secure:
            self.channel = grpc.secure_channel(host, grpc.ssl_channel_credentials())
        else:
            self.channel = grpc.insecure_channel(host)
        self.stub = tvectordb_pb2_grpc.TVectorStub(self.channel)

    def close(self):
        """Close the gRPC channel."""
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # =========================================================================
    # Insert Operations
    # =========================================================================

    def insert(
        self,
        id: int,
        vector: List[float],
        metadata: Optional[Dict[str, str]] = None,
    ) -> dict:
        """Insert a single vector with optional metadata tags.
        
        Args:
            id: Unique integer identifier for this vector
            vector: List of float values (must match server dimension)
            metadata: Optional key-value string tags for filtering
            
        Returns:
            dict with 'success', 'message', and 'id' fields
            
        Example:
            client.insert(id=1, vector=[0.1, 0.2, ...], metadata={"category": "finance"})
        """
        vec = tvectordb_pb2.Vector(
            id=id,
            values=vector,
            metadata=metadata or {},
        )
        request = tvectordb_pb2.InsertRequest(vector=vec)
        response = self.stub.Insert(request)
        return {
            "success": response.success,
            "message": response.message,
            "id": response.id,
        }

    def insert_batch(
        self,
        vectors: List[Tuple[int, List[float], Optional[Dict[str, str]]]],
    ) -> dict:
        """Insert multiple vectors at once for maximum throughput.
        
        Args:
            vectors: List of (id, vector, metadata) tuples.
                     metadata can be None for any entry.
                     
        Returns:
            dict with 'success', 'inserted', and 'total_vectors' fields
            
        Example:
            client.insert_batch([
                (1, [0.1, 0.2, ...], {"type": "doc"}),
                (2, [0.3, 0.4, ...], None),
            ])
        """
        pb_vectors = []
        for item in vectors:
            vid, vals, meta = item[0], item[1], item[2] if len(item) > 2 else None
            pb_vectors.append(tvectordb_pb2.Vector(
                id=vid,
                values=vals,
                metadata=meta or {},
            ))
        request = tvectordb_pb2.BatchInsertRequest(vectors=pb_vectors)
        response = self.stub.InsertBatch(request)
        return {
            "success": response.success,
            "inserted": response.inserted,
            "total_vectors": response.total_vectors,
        }

    # =========================================================================
    # Search Operations
    # =========================================================================

    def search(
        self,
        vector: List[float],
        top_k: int = 5,
        exact: bool = False,
        filter: Optional[Dict[str, str]] = None,
        ef_search: int = 100,
    ) -> List[dict]:
        """Search for the closest vectors to the query.
        
        Args:
            vector: Query vector (must match server dimension)
            top_k: Number of results to return (default: 5)
            exact: If True, uses Hybrid Two-Tier Re-Ranking for 100% accuracy.
                   If False, uses fast approximate RAM-only search.
            filter: Optional metadata filter. Only vectors matching ALL
                    key-value pairs will be considered.
            ef_search: Beam width for HNSW approximate search. Higher = more accurate
                       but slower (default: 100). Ignored if exact=True.
                    
        Returns:
            List of dicts with 'id' and 'score' fields, sorted by relevance
            
        Example:
            # Fast approximate search
            results = client.search(vector=[0.1, ...], top_k=10)
            
            # 100% accurate hybrid search with metadata filter
            results = client.search(
                vector=[0.1, ...], 
                top_k=5, 
                exact=True,
                filter={"category": "finance"}
            )
        """
        request = tvectordb_pb2.SearchRequest(
            query=vector,
            exact=exact,
            top_k=top_k,
            filter=filter or {},
            ef_search=ef_search,
        )
        response = self.stub.Search(request)
        return [
            {"id": r.id, "score": r.score}
            for r in response.results
        ]

    def search_batch(
        self,
        vectors: List[List[float]],
        top_k: int = 5,
    ) -> List[List[dict]]:
        """Run multiple search queries in a single RPC call.
        
        Args:
            vectors: List of query vectors
            top_k: Number of results per query
            
        Returns:
            List of result lists, one per query
        """
        pb_queries = [
            tvectordb_pb2.SearchQuery(values=v) for v in vectors
        ]
        request = tvectordb_pb2.BatchSearchRequest(
            queries=pb_queries,
            top_k=top_k,
        )
        response = self.stub.SearchBatch(request)
        return [
            [{"id": r.id, "score": r.score} for r in batch.results]
            for batch in response.batch_results
        ]

    # =========================================================================
    # Delete Operations
    # =========================================================================

    def delete(self, id: int) -> dict:
        """Delete a vector by ID.
        
        Args:
            id: The vector ID to delete
            
        Returns:
            dict with 'success', 'deleted', and 'id' fields
        """
        request = tvectordb_pb2.DeleteRequest(id=id)
        response = self.stub.Delete(request)
        return {
            "success": response.success,
            "deleted": response.deleted,
            "id": response.id,
        }

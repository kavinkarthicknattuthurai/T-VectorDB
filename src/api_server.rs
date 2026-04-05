//! # REST API Server V3 — Full-Featured HTTP Endpoints with CORS
//!
//! Endpoints:
//! - `GET  /`          — Health check
//! - `GET  /stats`     — Database statistics + memory + HNSW graph info
//! - `POST /insert`    — Insert a single vector
//! - `POST /insert_batch` — Insert multiple vectors
//! - `POST /search`    — Search (approximate or exact hybrid)
//! - `POST /search_batch` — Multiple queries at once
//! - `DELETE /vectors/{id}` — Delete a vector

use crate::execution_engine::{batch_search, hybrid_search, search_ram_store};
use crate::storage_engine::Database;
use crate::turbo_math::TurboIndex;

use axum::{
    extract::{Path, State},
    http::{header, Method, StatusCode},
    routing::{delete, get, post},
    Json, Router,
};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

/// Type alias for batch insert tuples: (id, vector, optional metadata)
type VectorTuple = (u64, Vec<f32>, Option<std::collections::HashMap<String, String>>);

// ============================================================================
// Application State
// ============================================================================

pub struct AppState {
    pub db: Database,
    pub index: TurboIndex,
}

// ============================================================================
// Request / Response Types
// ============================================================================

#[derive(serde::Deserialize)]
pub struct InsertRequest {
    pub id: u64,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

#[derive(serde::Serialize)]
pub struct InsertResponse {
    pub success: bool,
    pub message: String,
    pub id: u64,
}

#[derive(serde::Deserialize)]
pub struct BatchInsertRequest {
    pub vectors: Vec<InsertRequest>,
}

#[derive(serde::Serialize)]
pub struct BatchInsertResponse {
    pub success: bool,
    pub inserted: usize,
    pub total_vectors: usize,
}

#[derive(serde::Deserialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    #[serde(default)]
    pub exact: bool,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub filter: std::collections::HashMap<String, String>,
    #[serde(default = "default_ef_search")]
    pub ef_search: usize,
}

fn default_top_k() -> usize { 5 }
fn default_ef_search() -> usize { 100 }

#[derive(serde::Deserialize)]
pub struct BatchSearchRequest {
    pub vectors: Vec<Vec<f32>>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

#[derive(serde::Serialize)]
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
}

#[derive(serde::Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub total_vectors: usize,
    pub mode: String,
}

#[derive(serde::Serialize)]
pub struct BatchSearchResponse {
    pub results: Vec<Vec<SearchResult>>,
    pub total_vectors: usize,
}

#[derive(serde::Serialize)]
pub struct StatsResponse {
    pub total_vectors: usize,
    pub dimension: usize,
    pub bit_width: usize,
    pub compression_ratio: String,
    pub ram_memory_bytes: usize,
    pub ram_memory_mb: f64,
    pub hnsw_nodes: usize,
    pub hnsw_layers: usize,
    pub hnsw_deleted: usize,
    pub version: String,
}

#[derive(serde::Serialize)]
pub struct DeleteResponse {
    pub success: bool,
    pub deleted: bool,
    pub id: u64,
}

// ============================================================================
// Handlers
// ============================================================================

async fn health() -> &'static str {
    "T-VectorDB is running 🚀"
}

async fn stats(State(state): State<Arc<AppState>>) -> Json<StatsResponse> {
    let mem = state.db.memory_bytes();
    let graph = state.db.hnsw.read().unwrap();
    Json(StatsResponse {
        total_vectors: state.db.len(),
        dimension: state.index.d,
        bit_width: state.index.bits.bits(),
        compression_ratio: format!("{:.1}x", state.index.bits.compression_ratio()),
        ram_memory_bytes: mem,
        ram_memory_mb: mem as f64 / 1_048_576.0,
        hnsw_nodes: graph.len(),
        hnsw_layers: graph.max_layer_count(),
        hnsw_deleted: graph.deleted_count(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

async fn insert_one(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<InsertRequest>,
) -> Result<Json<InsertResponse>, (StatusCode, String)> {
    if payload.vector.len() != state.index.d {
        return Err((StatusCode::BAD_REQUEST, format!(
            "Dimension mismatch: expected {}, got {}", state.index.d, payload.vector.len()
        )));
    }

    let meta_opt = if payload.metadata.is_empty() { None } else { Some(&payload.metadata) };

    match state.db.insert(&state.index, &payload.vector, payload.id, meta_opt) {
        Ok(()) => {
            Ok(Json(InsertResponse {
                success: true,
                message: format!("Vector {} inserted ({})", payload.id, state.index.bits.bits()),
                id: payload.id,
            }))
        },
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, format!("Insert failed: {}", e))),
    }
}

async fn insert_batch(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<BatchInsertRequest>,
) -> Result<Json<BatchInsertResponse>, (StatusCode, String)> {
    for req in &payload.vectors {
        if req.vector.len() != state.index.d {
            return Err((StatusCode::BAD_REQUEST, format!(
                "Vector {} dimension mismatch: expected {}, got {}",
                req.id, state.index.d, req.vector.len()
            )));
        }
    }

    let pairs: Vec<VectorTuple> = payload.vectors
        .into_iter()
        .map(|r| {
            let meta = if r.metadata.is_empty() { None } else { Some(r.metadata) };
            (r.id, r.vector, meta)
        })
        .collect();

    match state.db.insert_batch(&state.index, &pairs) {
        Ok(count) => {
            // No need to rebuild HNSW — batch insert now does incremental HNSW updates
            Ok(Json(BatchInsertResponse {
                success: true,
                inserted: count,
                total_vectors: state.db.len(),
            }))
        },
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, format!("Batch insert failed: {}", e))),
    }
}

async fn search(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    if payload.vector.len() != state.index.d {
        return Err((StatusCode::BAD_REQUEST, format!(
            "Dimension mismatch: expected {}, got {}", state.index.d, payload.vector.len()
        )));
    }

    if state.db.is_empty() {
        return Ok(Json(SearchResponse { results: vec![], total_vectors: 0, mode: "empty".into() }));
    }

    let valid_ids = state.db.get_filtered_ids(&payload.filter);

    let (results, mode) = if payload.exact {
        // Exact hybrid: HNSW shortlist → disk re-rank → 100% accuracy
        (hybrid_search(&state.db, &state.index, &payload.vector, payload.top_k, valid_ids.as_ref()), "exact (hybrid)")
    } else {
        // Default: HNSW graph search — O(log n)
        let graph = state.db.hnsw.read().unwrap();
        if graph.is_empty() {
            // Fallback to linear scan if graph not built
            drop(graph);
            let ram_guard = state.db.ram.read().unwrap();
            let res = search_ram_store(&state.index, &ram_guard, &payload.vector, payload.top_k, valid_ids.as_ref());
            (res, "approximate (linear)")
        } else {
            let ram_guard = state.db.ram.read().unwrap();
            let res = graph.search(&payload.vector, &state.index, &ram_guard, payload.top_k, payload.ef_search, valid_ids.as_ref());
            (res, "approximate (hnsw)")
        }
    };

    Ok(Json(SearchResponse {
        results: results.into_iter().map(|(id, score)| SearchResult { id, score }).collect(),
        total_vectors: state.db.len(),
        mode: mode.to_string(),
    }))
}

async fn search_batch_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<BatchSearchRequest>,
) -> Result<Json<BatchSearchResponse>, (StatusCode, String)> {
    for (i, v) in payload.vectors.iter().enumerate() {
        if v.len() != state.index.d {
            return Err((StatusCode::BAD_REQUEST, format!(
                "Query {} dimension mismatch: expected {}, got {}", i, state.index.d, v.len()
            )));
        }
    }

    let ram_guard = state.db.ram.read().unwrap();
    let all_results = batch_search(&state.index, &ram_guard, &payload.vectors, payload.top_k, None);
    drop(ram_guard);

    Ok(Json(BatchSearchResponse {
        results: all_results.into_iter().map(|results| {
            results.into_iter().map(|(id, score)| SearchResult { id, score }).collect()
        }).collect(),
        total_vectors: state.db.len(),
    }))
}

async fn delete_vector(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<DeleteResponse>, (StatusCode, String)> {
    match state.db.delete(id) {
        Ok(deleted) => Ok(Json(DeleteResponse { success: true, deleted, id })),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, format!("Delete failed: {}", e))),
    }
}

// ============================================================================
// Router
// ============================================================================

pub fn create_router(state: Arc<AppState>) -> Router {
    // CORS: Allow all origins for open-source use
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION]);

    Router::new()
        .route("/", get(health))
        .route("/stats", get(stats))
        .route("/insert", post(insert_one))
        .route("/insert_batch", post(insert_batch))
        .route("/search", post(search))
        .route("/search_batch", post(search_batch_handler))
        .route("/vectors/{id}", delete(delete_vector))
        .layer(cors)
        .with_state(state)
}

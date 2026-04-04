//! # REST API Server — Axum HTTP Endpoints for T-VectorDB
//!
//! Provides two endpoints:
//! - `POST /insert` — Insert a vector with an ID
//! - `POST /search` — Search for similar vectors (approximate or exact)

use crate::execution_engine::{hybrid_search, search_ram_store};
use crate::storage_engine::Database;
use crate::turbo_math::TurboIndex;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

// ============================================================================
// Application State
// ============================================================================

/// Shared application state, safe for concurrent access.
pub struct AppState {
    pub db: RwLock<Database>,
    pub index: TurboIndex,
}

// ============================================================================
// Request / Response Types
// ============================================================================

#[derive(Deserialize)]
pub struct InsertRequest {
    pub id: u64,
    pub vector: Vec<f32>,
}

#[derive(Serialize)]
pub struct InsertResponse {
    pub success: bool,
    pub message: String,
    pub id: u64,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    #[serde(default)]
    pub exact: bool,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_top_k() -> usize {
    5
}

#[derive(Serialize)]
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub total_vectors: usize,
    pub mode: String,
}

#[derive(Serialize)]
pub struct StatsResponse {
    pub total_vectors: usize,
    pub dimension: usize,
    pub compression_ratio: String,
    pub version: String,
}

// ============================================================================
// Handlers
// ============================================================================

/// Health check endpoint.
async fn health() -> &'static str {
    "T-VectorDB is running 🚀"
}

/// GET /stats — Database statistics.
async fn stats(State(state): State<Arc<AppState>>) -> Json<StatsResponse> {
    let db = state.db.read().unwrap();
    Json(StatsResponse {
        total_vectors: db.len(),
        dimension: state.index.d,
        compression_ratio: "16:1 (Float32 → 3-bit)".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// POST /insert — Insert a vector into the database.
async fn insert(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<InsertRequest>,
) -> Result<Json<InsertResponse>, (StatusCode, String)> {
    // Validate dimension
    if payload.vector.len() != state.index.d {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Dimension mismatch: expected {}, got {}",
                state.index.d,
                payload.vector.len()
            ),
        ));
    }

    // Acquire write lock and insert
    let mut db = state.db.write().unwrap();
    match db.insert(&state.index, &payload.vector, payload.id) {
        Ok(()) => {
            tracing::info!("Inserted vector id={}, total={}", payload.id, db.len());
            Ok(Json(InsertResponse {
                success: true,
                message: format!("Vector {} inserted successfully", payload.id),
                id: payload.id,
            }))
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to insert: {}", e),
        )),
    }
}

/// POST /search — Search for similar vectors.
async fn search(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    // Validate dimension
    if payload.vector.len() != state.index.d {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Dimension mismatch: expected {}, got {}",
                state.index.d,
                payload.vector.len()
            ),
        ));
    }

    let db = state.db.read().unwrap();

    if db.is_empty() {
        return Ok(Json(SearchResponse {
            results: vec![],
            total_vectors: 0,
            mode: "empty".to_string(),
        }));
    }

    let (results, mode) = if payload.exact {
        // Two-tier hybrid search: RAM shortlist → exact disk re-rank
        let r = hybrid_search(&db, &state.index, &payload.vector, payload.top_k);
        (r, "exact (hybrid)")
    } else {
        // Fast approximate search over compressed RAM store only
        let r = search_ram_store(&state.index, &db.ram, &payload.vector, payload.top_k);
        (r, "approximate (RAM-only)")
    };

    let search_results: Vec<SearchResult> = results
        .into_iter()
        .map(|(id, score)| SearchResult { id, score })
        .collect();

    Ok(Json(SearchResponse {
        results: search_results,
        total_vectors: db.len(),
        mode: mode.to_string(),
    }))
}

// ============================================================================
// Router
// ============================================================================

/// Build the Axum router with all endpoints.
pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(health))
        .route("/stats", get(stats))
        .route("/insert", post(insert))
        .route("/search", post(search))
        .with_state(state)
}

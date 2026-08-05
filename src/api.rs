//! REST API for an exodus node (axum).  Mirrors the reference FastAPI router.

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

use crate::coordinator::ExodusCoordinator;

type Coord = Arc<ExodusCoordinator>;

fn default_work() -> String {
    "text_generation".to_string()
}

#[derive(Deserialize)]
struct ClaimPayload {
    model_id: String,
    params_b: f64,
    precision: String,
    prompt_tokens: i64,
    completion_tokens: i64,
    compute_seconds: f64,
    flops_estimate: f64,
    #[serde(default)]
    device_tier: Option<String>,
    #[serde(default = "default_work")]
    work_type: String,
    started_at: Option<String>,
    ended_at: Option<String>,
}

#[derive(Deserialize)]
struct LedgerQuery {
    limit: Option<usize>,
}

#[derive(Deserialize)]
struct ClaimsQuery {
    node_id: Option<String>,
}

#[derive(Deserialize)]
struct PeerConnect {
    addr: String,
}

#[derive(Deserialize)]
struct UploadQuery {
    name: Option<String>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
}

// ------------------------------------------------------------------- handlers

async fn status(State(c): State<Coord>) -> Json<Value> {
    Json(c.status())
}

async fn credits(State(c): State<Coord>) -> Json<Value> {
    Json(c.entitlement())
}

async fn network(State(c): State<Coord>) -> Json<Value> {
    Json(c.network_report())
}

async fn ledger(State(c): State<Coord>, Query(q): Query<LedgerQuery>) -> Json<Value> {
    let limit = q.limit.unwrap_or(20).min(500);
    Json(c.ledger_summary(limit))
}

async fn ledger_verify(State(c): State<Coord>) -> Json<Value> {
    let (ok, detail) = c.store.verify_chain();
    Json(json!({ "ok": ok, "detail": detail }))
}

async fn claims(State(c): State<Coord>, Query(q): Query<ClaimsQuery>) -> Json<Value> {
    Json(c.claims_for(q.node_id.as_deref()))
}

async fn consensus(State(c): State<Coord>) -> Json<Value> {
    let mut peers: Vec<String> = c.consensus.active_peers();
    peers.sort();
    Json(json!({
        "node_id": c.identity.node_id,
        "view": c.consensus.view(),
        "sealer": c.consensus.sealer_node(),
        "is_sealer": c.consensus.is_sealer(),
        "quorum_size": c.consensus.quorum_size(),
        "committee": c.consensus.active_peers(),
        "peers": peers,
        "pending_claims": c.consensus.pending_claims_count(),
        "ledger_height": c.store.height(),
        "ledger_head": c.store.head().map(|h| h.block_hash()),
    }))
}

async fn nodes(State(c): State<Coord>) -> Json<Value> {
    let report = c.network_report();
    let participants = report.get("participants").cloned().unwrap_or(Value::Null);
    Json(json!({ "nodes": participants }))
}

/// Ask the node to connect to a peer at `host:port`.
async fn network_connect(State(c): State<Coord>, Json(p): Json<PeerConnect>) -> impl IntoResponse {
    match c.connect_peer(&p.addr) {
        Ok(msg) => (StatusCode::OK, Json(json!({ "message": msg }))),
        Err(msg) => (StatusCode::CONFLICT, Json(json!({ "error": msg }))),
    }
}

async fn rewards(State(c): State<Coord>) -> Json<Value> {
    let report = c.network_report();
    Json(report.get("reward_parameters").cloned().unwrap_or(Value::Null))
}

async fn healthz(State(c): State<Coord>) -> Json<Value> {
    let (ok, _) = c.store.verify_chain();
    Json(json!({ "status": if ok { "ok" } else { "degraded" }, "detail": ok }))
}

async fn submit_claim(State(c): State<Coord>, Json(p): Json<ClaimPayload>) -> impl IntoResponse {
    let device_tier = p.device_tier.unwrap_or_else(|| c.gpu_info().tier_string());
    match c.submit_contribution(
        p.model_id,
        p.params_b,
        p.precision,
        p.prompt_tokens,
        p.completion_tokens,
        p.compute_seconds,
        p.flops_estimate,
        device_tier,
        p.work_type,
        p.started_at,
        p.ended_at,
    ) {
        Ok(claim_id) => (
            StatusCode::OK,
            Json(json!({ "claim_id": claim_id, "message": "contribution submitted" })),
        ),
        Err(msg) => (StatusCode::BAD_REQUEST, Json(json!({ "error": msg }))),
    }
}

async fn models_list(State(c): State<Coord>) -> Json<Value> {
    let gpu = c.gpu_info();
    let mut gpu_json = gpu.to_value();
    gpu_json["gpu_layers"] = c
        .config
        .gpu_layers
        .map(|v| json!(v))
        .unwrap_or(Value::Null);
    Json(json!({
        "gpu": gpu_json,
        "models": available_models(&c),
    }))
}

fn available_models(c: &ExodusCoordinator) -> Vec<Value> {
    let dir = c.config.models_dir();
    let mut items = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            items.push(json!({ "name": name, "size_bytes": size }));
        }
    }
    items.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    items
}

/// Upper bound for a single model upload (512 GiB) — effectively unlimited for
/// real model files while still guarding against unbounded disk usage.
const MAX_UPLOAD_BYTES: u64 = 512 * 1024 * 1024 * 1024;

/// Store an uploaded model file (raw `application/octet-stream` body) under
/// the models directory as `name`.  The body is streamed straight to disk so
/// multi-GB model files are not buffered in memory and do not hit axum's
/// default request-body limit.
async fn models_upload(
    State(c): State<Coord>,
    Query(q): Query<UploadQuery>,
    body: Body,
) -> impl IntoResponse {
    let name = match q.name {
        Some(n) if safe_model_name(&n) => n,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "invalid or missing model name" })),
            )
        }
    };
    let dir = c.config.models_dir();
    if std::fs::create_dir_all(&dir).is_err() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "cannot create models directory" })),
        );
    }
    let path = dir.join(&name);
    let tmp = dir.join(format!(".{name}.{}.uploading", uuid::Uuid::new_v4().simple()));
    let mut file = match tokio::fs::File::create(&tmp).await {
        Ok(f) => f,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": "write failed" })),
            )
        }
    };

    let mut stream = body.into_data_stream();
    let mut total: u64 = 0;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) => {
                total += bytes.len() as u64;
                if total > MAX_UPLOAD_BYTES {
                    let _ = tokio::fs::remove_file(&tmp).await;
                    return (
                        StatusCode::PAYLOAD_TOO_LARGE,
                        Json(json!({ "error": "upload exceeds maximum size" })),
                    );
                }
                if file.write_all(&bytes).await.is_err() {
                    let _ = tokio::fs::remove_file(&tmp).await;
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({ "error": "write failed" })),
                    );
                }
            }
            Err(_) => {
                let _ = tokio::fs::remove_file(&tmp).await;
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({ "error": "upload read failed" })),
                );
            }
        }
    }
    let _ = file.flush().await;
    drop(file);

    if total == 0 {
        let _ = tokio::fs::remove_file(&tmp).await;
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "empty upload" })),
        );
    }
    if tokio::fs::rename(&tmp, &path).await.is_err() {
        let _ = tokio::fs::remove_file(&tmp).await;
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "write failed" })),
        );
    }
    (
        StatusCode::CREATED,
        Json(json!({
            "message": "uploaded",
            "name": name,
            "size_bytes": total,
        })),
    )
}

/// Delete a model file by name.
async fn models_delete(State(c): State<Coord>, Path(name): Path<String>) -> impl IntoResponse {
    if !safe_model_name(&name) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "invalid model name" })),
        );
    }
    let path: PathBuf = c.config.models_dir().join(&name);
    match std::fs::remove_file(&path) {
        Ok(()) => (StatusCode::OK, Json(json!({ "deleted": name }))),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "model not found" })),
        ),
    }
}

/// A model name must be a plain filename: non-empty, no path separators and
/// not a hidden/dot entry, to keep deletes from escaping the models directory.
fn safe_model_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.starts_with('.')
        && !name.contains('/')
        && !name.contains('\\')
}

async fn events(State(c): State<Coord>) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, _rx) = broadcast::channel::<(i64, String)>(64);
    let tx2 = tx.clone();
    c.add_commit_hook("sse", Box::new(move |height, hash| {
        let _ = tx2.send((height, hash));
    }));
    let hello = Event::default().data(
        json!({
            "type": "hello",
            "height": c.store.height(),
            "block_hash": c.store.head().map(|h| h.block_hash()),
            "node_id": c.identity.node_id,
        })
        .to_string(),
    );
    let stream = tokio_stream::once(Ok(hello))
        .chain(BroadcastStream::new(tx.subscribe()).map(|item| {
            let (height, hash) = item.unwrap_or((0, String::new()));
            Ok(Event::default().data(
                json!({ "height": height, "block_hash": hash }).to_string(),
            ))
        }));
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Chat with the distributed model.  This node does not bundle an inference
/// runtime yet, so the reply is a structured stub reporting the model, GPU
/// state and available model files; swap `chat_reply` for a real inference
/// call once a runtime is attached.
async fn chat(State(c): State<Coord>, Json(req): Json<ChatRequest>) -> impl IntoResponse {
    if req.messages.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "no messages" })),
        );
    }
    let model = match req.model.as_deref() {
        Some(m) if safe_model_name(m) => m.to_string(),
        Some(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "invalid model name" })),
            )
        }
        None => model_files(&c).into_iter().next().unwrap_or_else(|| "auto".to_string()),
    };
    let prompt = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();
    (StatusCode::OK, Json(chat_reply(&c, &model, &prompt)))
}

/// Names of the model files present in the models directory, sorted.
fn model_files(c: &ExodusCoordinator) -> Vec<String> {
    let dir = c.config.models_dir();
    let mut files: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    safe_model_name(&name).then_some(name)
                })
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

fn chat_reply(c: &ExodusCoordinator, model: &str, prompt: &str) -> Value {
    let gpu = c.gpu_info();
    let files = model_files(c);
    let present = files.iter().any(|f| f == model);
    let absent = if present {
        String::new()
    } else {
        format!(" (selected '{model}' is not present)")
    };
    json!({
        "runtime": "stub",
        "model": model,
        "model_present": present,
        "models": files,
        "gpu": gpu.to_value(),
        "reply": format!(
            "[{model}] No inference runtime is bundled in this node yet, so I can't generate a real answer.\n\nNode state: {} ({}) with {} model file(s){absent}.\nReceived {} char(s) from the last message.",
            if gpu.available { "GPU-ready" } else { "CPU-only" },
            gpu.tier_string(),
            files.len(),
            prompt.chars().count(),
        ),
    })
}

async fn root() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("static/index.html"))
}

async fn dash() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("static/dash.html"))
}

// ------------------------------------------------------------------- server

/// Serve the node API on `host:port` until shutdown.
pub async fn serve(c: Coord, host: String, port: u16) {
    let app = Router::new()
        .route("/", get(root))
        .route("/exodus/dash.html", get(dash))
        .route("/exodus/chat", post(chat))
        .route("/exodus/status", get(status))
        .route("/exodus/credits", get(credits))
        .route("/exodus/network", get(network))
        .route("/exodus/ledger", get(ledger))
        .route("/exodus/ledger/verify", get(ledger_verify))
        .route("/exodus/claims", get(claims).post(submit_claim))
        .route("/exodus/consensus", get(consensus))
        .route("/exodus/nodes", get(nodes))
        .route("/exodus/network/peers", post(network_connect))
        .route("/exodus/rewards", get(rewards))
        .route("/exodus/healthz", get(healthz))
        .route("/exodus/models", get(models_list))
        .route("/exodus/models/upload", post(models_upload))
        .route("/exodus/models/:name", delete(models_delete))
        .route("/exodus/events", get(events))
        .with_state(c);
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await;
    match listener {
        Ok(l) => {
            if let Err(e) = axum::serve(l, app).await {
                eprintln!("api server error: {e}");
            }
        }
        Err(e) => eprintln!("api bind failed on {host}:{port}: {e}"),
    }
}
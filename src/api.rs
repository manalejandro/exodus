//! REST API for an exodus node (axum).  Mirrors the reference FastAPI router.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};
use tokio::io::AsyncWriteExt;
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::Stream as TokioStream;
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

/// Wraps an SSE stream and removes a coordinator commit-hook by name when the
/// stream is dropped (i.e. when the SSE client disconnects), so repeated
/// dashboard polls do not leak one handler per connection.
struct HookCleanup<S> {
    inner: S,
    coordinator: Coord,
    hook_name: String,
    armed: bool,
}

impl<S: TokioStream + Unpin> TokioStream for HookCleanup<S> {
    type Item = S::Item;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl<S> Drop for HookCleanup<S> {
    fn drop(&mut self) {
        if self.armed {
            self.coordinator.remove_commit_hook(&self.hook_name);
        }
    }
}

async fn events(State(c): State<Coord>) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, _rx) = broadcast::channel::<(i64, String)>(64);
    let tx2 = tx.clone();
    let hook_name = format!("sse-{}", uuid::Uuid::new_v4().simple());
    c.add_commit_hook(&hook_name, Box::new(move |height, hash| {
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
    let inner = tokio_stream::once(Ok(hello))
        .chain(BroadcastStream::new(tx.subscribe()).map(|item| {
            let (height, hash) = item.unwrap_or((0, String::new()));
            Ok(Event::default().data(
                json!({ "height": height, "block_hash": hash }).to_string(),
            ))
        }));
    // Remove the hook when the connection drops so SSE polls do not leak
    // handlers (each one held a broadcast::Sender forever).
    Sse::new(HookCleanup {
        inner,
        coordinator: c.clone(),
        hook_name,
        armed: true,
    })
    .keep_alive(KeepAlive::default())
}

/// Chat with the distributed model.  When a llama.cpp runtime is available
/// (`EXODUS_LLAMA_BIN`, default `llama-cli`) and a model file is present, the
/// node runs a real completion; otherwise it returns a truthful stub
/// describing the node state.
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
        eprintln!(
            "[chat] node={} model={} turns={} inference={} distributed={}",
            c.identity.node_id,
            model,
            req.messages.len(),
            c.config.inference,
            c.config.distributed_inference,
        );
    let turns: Vec<crate::inference::ChatTurn> = req
        .messages
        .iter()
        .map(|m| crate::inference::ChatTurn {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();
    (StatusCode::OK, Json(chat_response(&c, &model, &turns).await))
}

async fn chat_response(c: &Coord, model: &str, turns: &[crate::inference::ChatTurn]) -> Value {
    let gpu = c.gpu_info();
    let files = model_files(c);
    let stub = |reason: &str| {
        eprintln!("[chat] node={} stub: {reason}", c.identity.node_id);
        chat_stub(c, &gpu, &files, model, reason)
    };
    let Some(model_path) = model_path_for(c, model) else {
        return stub("model file not present");
    };
    if !c.config.inference {
        return stub("inference disabled (EXODUS_INFERENCE=0)");
    }

    // 1. Local completion (the node that owns the chat always runs too).
    let config = c.config.clone();
    let path = model_path.clone();
    let turns_for_task = turns.to_vec();
    let gpu_for_task = gpu.clone();
    let local_started = std::time::Instant::now();
    let local = tokio::task::spawn_blocking(move || {
        crate::inference::complete(&config, &gpu_for_task, &path, &turns_for_task)
    })
    .await
    .unwrap_or_else(|e| Err(e.to_string()));
    let local_elapsed = local_started.elapsed().as_secs_f64();
    let (local_reply, local_error) = match local {
        Ok(reply) => (Some(reply), None),
        Err(e) => (None, Some(e)),
    };
    if let Some(ref reply) = local_reply {
        let prompt_chars: usize = turns.iter().map(|t| t.content.chars().count()).sum();
        c.record_inference_claim(model, prompt_chars, reply.chars().count(), local_elapsed);
    }

    // 2. Distributed fan-out: every peer runs the same prompt and replies on
    //    `topics::INFER_RESPONSES`; we collect until we have heard from the
    //    expected peers or the deadline elapses.
    let mut peers_asked = 0usize;
    let mut peers_responded = 0usize;
    let mut peer_replies: Vec<(String, String)> = Vec::new();
    let mut peer_errors: Vec<(String, String)> = Vec::new();
    if c.config.distributed_inference {
        let request_id = uuid::Uuid::new_v4().to_string();
        let messages: Vec<crate::models::InferenceTurn> = turns
            .iter()
            .map(|t| crate::models::InferenceTurn {
                role: t.role.clone(),
                content: t.content.clone(),
            })
            .collect();
        let mut rx = c.request_inference(
            request_id.clone(),
            model.to_string(),
            c.config.max_tokens,
            messages,
        );
        // Wait for the peers to answer.  A peer counts if it is known to the
        // committee *or* the transport currently has a live TCP connection, so
        // a fresh node whose heartbeats have not propagated yet still waits
        // long enough for a slow llama-cli on the remote to reply.  A truly
        // solo node (no committee member, no connected peer) gets a short
        // grace window so its chat stays snappy.
        let known_peers = c.consensus.active_peers().len().saturating_sub(1);
        let connected_peers = c.transport.peer_count();
        let expected = if known_peers > 0 {
            known_peers
        } else if connected_peers > 0 {
            1
        } else {
            0
        };
        let deadline = if expected > 0 {
            Duration::from_secs_f64(c.config.distributed_timeout_seconds.max(1.0))
        } else {
            Duration::from_secs_f64(c.config.distributed_timeout_seconds.min(3.0))
        };
        let collect = async {
            let mut seen: HashSet<String> = HashSet::new();
            while let Some(response) = rx.recv().await {
                if response.node_id == c.identity.node_id {
                    continue;
                }
                if !seen.insert(response.node_id.clone()) {
                    continue;
                }
                peers_responded += 1;
                if let Some(err) = response.error {
                    peer_errors.push((response.node_id, err));
                } else {
                    peer_replies.push((response.node_id, response.reply));
                }
                if expected > 0 && seen.len() >= expected {
                    break;
                }
            }
        };
        let _ = tokio::time::timeout(deadline, collect).await;
        peers_asked = expected;
        c.drop_inference(&request_id);
    }

    // 3. Aggregate across local + peer completions: the reply with the most
    //    agreement wins (ties broken by length, then the local node).
    let (selected, reply, agreement) = pick_agreed_reply(local_reply.as_deref(), &peer_replies);

    json!({
        "runtime": "distributed",
        "model": model,
        "model_present": true,
        "models": files,
        "gpu": gpu.to_value(),
        "reply": reply,
        "local_reply": local_reply,
        "local_error": local_error,
        "distributed": {
            "enabled": c.config.distributed_inference,
            "peers_asked": peers_asked,
            "peers_responded": peers_responded,
            "selected": selected,
            "agreement": agreement,
            "responses": peer_replies
                .iter()
                .map(|(node_id, reply)| json!({ "node_id": node_id, "reply": reply }))
                .collect::<Vec<_>>(),
            "peer_errors": peer_errors
                .iter()
                .map(|(node_id, err)| json!({ "node_id": node_id, "error": err }))
                .collect::<Vec<_>>(),
        },
    })
}

/// Pick the completion with the most support among the local reply and the
/// peer replies.  Returns `(selected_node, reply, agreement_votes)`.
fn pick_agreed_reply(local: Option<&str>, peers: &[(String, String)]) -> (String, String, usize) {
    let mut candidates: Vec<(String, String)> = Vec::new();
    if let Some(l) = local {
        if !l.trim().is_empty() {
            candidates.push(("local".to_string(), l.to_string()));
        }
    }
    for (node, reply) in peers {
        if !reply.trim().is_empty() {
            candidates.push((node.clone(), reply.clone()));
        }
    }
    if candidates.is_empty() {
        return (String::new(), String::new(), 0);
    }
    if candidates.len() == 1 {
        let (node, reply) = candidates.pop().unwrap();
        return (node, reply, 1);
    }
    let sims: Vec<Vec<f64>> = (0..candidates.len())
        .map(|i| {
            (0..candidates.len())
                .map(|j| {
                    if i == j {
                        1.0
                    } else {
                        dice_similarity(&candidates[i].1, &candidates[j].1)
                    }
                })
                .collect()
        })
        .collect();
    let mut best = (0usize, 0usize, 0usize); // (index, votes, len)
    for i in 0..candidates.len() {
        let votes = 1 + (0..candidates.len())
            .filter(|&j| j != i && sims[i][j] >= 0.5)
            .count();
        let len = candidates[i].1.chars().count();
        if votes > best.1 || (votes == best.1 && len > best.2) {
            best = (i, votes, len);
        }
    }
    (candidates[best.0].0.clone(), candidates[best.0].1.clone(), best.1)
}

fn normalized_chars(s: &str) -> Vec<char> {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

fn bigrams(chars: &[char]) -> HashSet<(char, char)> {
    chars.windows(2).map(|w| (w[0], w[1])).collect()
}

/// Dice coefficient on character bigrams, `1.0` for identical text.
fn dice_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    let ca = normalized_chars(a);
    let cb = normalized_chars(b);
    if ca.len() < 2 || cb.len() < 2 {
        return if ca == cb { 1.0 } else { 0.0 };
    }
    let ga = bigrams(&ca);
    let gb = bigrams(&cb);
    let common = ga.intersection(&gb).count() as f64;
    2.0 * common / (ga.len() as f64 + gb.len() as f64)
}

/// Resolve a model name to a file on disk; `auto` picks the first file in the
/// models directory.
fn model_path_for(c: &ExodusCoordinator, model: &str) -> Option<PathBuf> {
    let name = if model == "auto" {
        model_files(c).into_iter().next()?
    } else {
        model.to_string()
    };
    if !safe_model_name(&name) {
        return None;
    }
    let path = c.config.models_dir().join(&name);
    path.is_file().then_some(path)
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

fn chat_stub(
    _c: &ExodusCoordinator,
    gpu: &crate::gpu::GpuInfo,
    files: &[String],
    model: &str,
    reason: &str,
) -> Value {
    json!({
        "runtime": "stub",
        "model": model,
        "model_present": files.iter().any(|f| f == model),
        "models": files,
        "gpu": gpu.to_value(),
        "reply": format!(
            "[{model}] Couldn't generate a real answer ({reason}), so this is a state stub.\n\nNode state: {} ({}) with {} model file(s).",
            if gpu.available { "GPU-ready" } else { "CPU-only" },
            gpu.tier_string(),
            files.len(),
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

fn app(c: Coord) -> Router {
    Router::new()
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
        .with_state(c)
}

/// Serve the node API on `host:port` until shutdown.
pub async fn serve(c: Coord, host: String, port: u16) {
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await;
    match listener {
        Ok(l) => {
            if let Err(e) = axum::serve(l, app(c)).await {
                eprintln!("api server error: {e}");
            }
        }
        Err(e) => eprintln!("api bind failed on {host}:{port}: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dice_similarity_matches_and_ignores_case_punct() {
        assert!((dice_similarity("The cat sat", "The cat sat") - 1.0).abs() < 1e-9);
        let similar = dice_similarity("Hello, world!", "hello world");
        assert!(similar > 0.9, "similar={similar}");
        let unrelated = dice_similarity("hello world", "quantum chromodynamics");
        assert!(unrelated < 0.3, "unrelated={unrelated}");
        assert_eq!(dice_similarity("a", "a"), 1.0);
        assert_eq!(dice_similarity("a", "b"), 0.0);
    }

    #[test]
    fn aggregation_picks_majority_then_longest_then_local() {
        // Local disagrees; two peers agree with each other -> a peer wins.
        let peers = vec![
            ("b".to_string(), "the sky is blue today".to_string()),
            ("c".to_string(), "the sky is blue today".to_string()),
        ];
        let (selected, reply, votes) = pick_agreed_reply(Some("completely different answer"), &peers);
        assert_eq!(selected, "b");
        assert_eq!(reply, "the sky is blue today");
        assert_eq!(votes, 2);

        // No agreement: tie on votes, longer reply wins.
        let peers = vec![
            ("b".to_string(), "short".to_string()),
            ("c".to_string(), "a much longer different reply".to_string()),
        ];
        let (selected, reply, votes) = pick_agreed_reply(None, &peers);
        assert_eq!(selected, "c");
        assert_eq!(reply, "a much longer different reply");
        assert_eq!(votes, 1);

        // Identical local + peer -> local wins and agreement is maximal.
        let peers = vec![("b".to_string(), "same answer here".to_string())];
        let (selected, reply, votes) = pick_agreed_reply(Some("same answer here"), &peers);
        assert_eq!(selected, "local");
        assert_eq!(reply, "same answer here");
        assert_eq!(votes, 2);

        // No candidates at all.
        let (selected, reply, votes) = pick_agreed_reply(None, &[]);
        assert_eq!(selected, "");
        assert_eq!(reply, "");
        assert_eq!(votes, 0);
    }

    fn e2e_dir(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "exodus-api-test-{}-{n}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join(name);
        std::fs::create_dir_all(&out).unwrap();
        out
    }

    fn fake_llama(bin: &std::path::Path) {
        std::fs::write(bin, "#!/bin/sh\nprintf 'Assistant: distributed answer\\n'\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perm = std::fs::metadata(bin).unwrap().permissions();
            perm.set_mode(0o755);
            std::fs::set_permissions(bin, perm).unwrap();
        }
    }

    fn find_subsequence(hay: &[u8], needle: &[u8]) -> Option<usize> {
        hay.windows(needle.len()).position(|w| w == needle)
    }

    async fn raw_post(addr: &str, body: Value) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let payload = serde_json::to_vec(&body).unwrap();
        let head = format!(
            "POST /exodus/chat HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            payload.len()
        );
        stream.write_all(head.as_bytes()).await.unwrap();
        stream.write_all(&payload).await.unwrap();
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            let n = stream.read(&mut tmp).await.unwrap();
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
            if let Some(pos) = find_subsequence(&buf, b"\r\n\r\n") {
                let header = String::from_utf8_lossy(&buf[..pos]).to_string();
                let body_start = pos + 4;
                let clen = header
                    .lines()
                    .find_map(|line| {
                        let line = line.to_lowercase();
                        line.strip_prefix("content-length:")
                            .map(|v| v.trim().to_string())
                    })
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(buf.len() - body_start);
                let mut body = buf[body_start..].to_vec();
                while body.len() < clen {
                    let n = stream.read(&mut tmp).await.unwrap();
                    if n == 0 {
                        break;
                    }
                    body.extend_from_slice(&tmp[..n]);
                }
                return String::from_utf8_lossy(&body[..clen.min(body.len())]).into_owned();
            }
        }
        String::from_utf8_lossy(&buf).into_owned()
    }

    #[tokio::test]
    async fn chat_endpoint_fans_out_to_peers() {
        let base = e2e_dir("run");
        let llama = base.join("llama-cli");
        fake_llama(&llama);
        let models = base.join("models");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::write(models.join("demo-model.gguf"), "fake model bytes").unwrap();
        let mut cfg = crate::config::config_from_env();
        cfg.inference = true;
        cfg.max_tokens = 32;
        cfg.llama_bin = llama.to_string_lossy().into_owned();
        cfg.model_dir = Some(models);
        cfg.distributed_timeout_seconds = 5.0;

        let transport: Arc<dyn crate::network::Transport> =
            Arc::new(crate::network::LocalTransport::new());
        let mut coords = Vec::new();
        for i in 0..3 {
            let identity = crate::simulation::make_identity(&format!("e2e-{i}"));
            let store = Arc::new(
                crate::ledger::ChainStore::open(&base.join(format!("node-{i}/ledger.sqlite3")))
                    .unwrap(),
            );
            let c = ExodusCoordinator::new(identity, store, transport.clone(), cfg.clone(), None);
            c.connect();
            coords.push(c);
        }
        for _ in 0..3 {
            for c in &coords {
                c.consensus.tick();
            }
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = coords[0].clone();
        let addr = format!("127.0.0.1:{port}");
        tokio::spawn(async move {
            let _ = axum::serve(listener, app(server)).await;
        });
        std::thread::sleep(Duration::from_millis(150));

        let body = json!({
            "model": "demo-model.gguf",
            "messages": [ { "role": "user", "content": "hello" } ],
        });
        let response = raw_post(&addr, body).await;
        let v: Value = serde_json::from_str(&response).unwrap_or(Value::Null);

        assert_eq!(v["model_present"], json!(true), "response: {v}");
        assert_eq!(v["runtime"], "distributed");
        assert_eq!(v["local_reply"], "distributed answer");
        assert_eq!(v["distributed"]["enabled"], true);
        assert_eq!(
            v["distributed"]["peers_responded"], 2,
            "the two peers must have run the completion: {v}"
        );
        let responses = v["distributed"]["responses"].as_array().unwrap();
        assert_eq!(responses.len(), 2);
        let mut nodes: Vec<String> = responses
            .iter()
            .map(|r| r["node_id"].as_str().unwrap().to_string())
            .collect();
        nodes.sort();
        nodes.dedup();
        assert_eq!(nodes.len(), 2, "replies must come from two distinct peers: {v}");

        for c in &coords {
            c.close();
        }
    }
}
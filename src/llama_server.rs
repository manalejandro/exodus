//! Long-lived `llama-server` backend for chat completions.
//!
//! Recent llama.cpp builds turned `llama-cli` into an interactive chat client
//! (refactor PR #17824): it ignores `-p` whenever the model ships a chat
//! template, enters a REPL, waits on stdin and never produces clean output when
//! stdout is piped — so one-shot shelling out hangs until the timeout.  The
//! supported way to script llama.cpp is `llama-server`'s OpenAI-compatible
//! HTTP API (`/v1/chat/completions`), which also applies the model's chat
//! template correctly and serves concurrent slots with `--parallel N`.
//!
//! One `llama-server` process is kept alive for the lifetime of the node and
//! reused across turns.  Switching to a different model restarts it.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::inference::ChatTurn;

struct ServerState {
    child: Option<Child>,
    model: Option<std::path::PathBuf>,
}

pub struct LlamaServer {
    bin: String,
    host: String,
    port: u16,
    parallel: usize,
    layers: i64,
    timeout: Duration,
    state: Mutex<ServerState>,
}

impl LlamaServer {
    pub fn new(
        bin: String,
        host: String,
        port: u16,
        parallel: usize,
        layers: i64,
        timeout: Duration,
    ) -> Arc<Self> {
        Arc::new(LlamaServer {
            bin,
            host,
            port,
            parallel: parallel.max(1),
            layers,
            timeout,
            state: Mutex::new(ServerState {
                child: None,
                model: None,
            }),
        })
    }

    /// Run one chat completion through the local llama-server.  The server is
    /// started (or restarted with the requested model) on first use.
    pub fn chat(
        &self,
        model_path: &Path,
        messages: &[ChatTurn],
        max_tokens: i64,
    ) -> Result<String, String> {
        self.ensure_ready(model_path)?;
        let body = json!({
            "messages": messages
                .iter()
                .map(|m| json!({ "role": m.role, "content": m.content }))
                .collect::<Vec<_>>(),
            "max_tokens": max_tokens.max(1),
            "temperature": 0.6,
            "stream": false,
        })
        .to_string();
        let resp = http_post_json(
            &self.host,
            self.port,
            "/v1/chat/completions",
            &body,
            self.timeout,
        )?;
        parse_chat_completion(&resp)
    }

    /// Ensure a llama-server is running and healthy for `model_path`, killing
    /// and relaunching it if the model changed or the process died.  The lock
    /// is held across a cold model load so concurrent calls wait for readiness
    /// instead of double-starting the server.
    fn ensure_ready(&self, model_path: &Path) -> Result<(), String> {
        if !model_path.is_file() {
            return Err(format!("model file not found: {}", model_path.display()));
        }
        let mut st = self.state.lock().unwrap();
        if st.model.as_deref() == Some(model_path) && self.health_ok() {
            return Ok(());
        }
        if st.child.is_some() {
            let _ = st.child.take().map(|mut c| {
                let _ = c.kill();
                let _ = c.wait();
            });
            st.model = None;
        }
        if !crate::inference::binary_exists(&self.bin) {
            return Err(format!(
                "llama-server runtime '{}' not found on PATH (set EXODUS_LLAMA_SERVER_BIN, or EXODUS_INFERENCE_BACKEND=cli for llama-cli one-shot)",
                self.bin
            ));
        }
        let child = Command::new(&self.bin)
            .arg("-m")
            .arg(model_path)
            .arg("--host")
            .arg(&self.host)
            .arg("--port")
            .arg(self.port.to_string())
            .arg("--parallel")
            .arg(self.parallel.to_string())
            .arg("--gpu-layers")
            .arg(self.layers.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .stdin(Stdio::null())
            .spawn()
            .map_err(|e| format!("failed to start {}: {e}", self.bin))?;
        st.child = Some(child);
        // Wait for the server to become ready (model load can take a while).
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            if self.health_ok() {
                break;
            }
            if Instant::now() >= deadline {
                let _ = st.child.take().map(|mut c| {
                    let _ = c.kill();
                    let _ = c.wait();
                });
                return Err(format!(
                    "llama-server did not become ready on {}:{} within 120s",
                    self.host, self.port
                ));
            }
            std::thread::sleep(Duration::from_millis(400));
        }
        st.model = Some(model_path.to_path_buf());
        Ok(())
    }

    fn health_ok(&self) -> bool {
        http_get(&self.host, self.port, "/health", Duration::from_secs(2))
            .map(|b| b.trim() == "OK")
            .unwrap_or(false)
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        let mut st = self.state.lock().unwrap();
        if let Some(mut child) = st.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Parse the `choices[0].message.content` out of a `/v1/chat/completions`
/// response, cleaning it like the llama-cli path and reporting a descriptive
/// error when the model returned no text.
fn parse_chat_completion(resp: &str) -> Result<String, String> {
    let v: Value = serde_json::from_str(resp)
        .map_err(|e| format!("bad completion JSON: {e}: {}", first_chars(resp, 300)))?;
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    let cleaned = crate::inference::clean(content);
    if cleaned.is_empty() {
        let finish = v["choices"][0]["finish_reason"].as_str().unwrap_or("?");
        return Err(format!(
            "model returned an empty completion (finish_reason={finish}); check the chat template / context size"
        ));
    }
    Ok(cleaned)
}

fn first_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn connect(host: &str, port: u16, timeout: Duration) -> Result<TcpStream, String> {
    let addr = (host, port)
        .to_socket_addrs()
        .map_err(|e| format!("resolve {host}:{port}: {e}"))?
        .next()
        .ok_or_else(|| format!("no address for {host}:{port}"))?;
    let stream = TcpStream::connect_timeout(&addr, timeout)
        .map_err(|e| format!("connect to llama-server at {host}:{port}: {e}"))?;
    Ok(stream)
}

fn http_get(host: &str, port: u16, path: &str, timeout: Duration) -> Result<String, String> {
    let mut stream = connect(host, port, timeout)?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    let req = format!("GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n");
    stream
        .write_all(req.as_bytes())
        .map_err(|e| format!("GET {path}: {e}"))?;
    read_response(stream)
}

fn http_post_json(
    host: &str,
    port: u16,
    path: &str,
    body: &str,
    timeout: Duration,
) -> Result<String, String> {
    let mut stream = connect(host, port, timeout)?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream
        .write_all(req.as_bytes())
        .map_err(|e| format!("POST {path}: {e}"))?;
    read_response(stream)
}

/// Read an HTTP/1.1 response and return the body, handling both
/// `Content-Length` and `Transfer-Encoding: chunked` framing.
fn read_response(stream: TcpStream) -> Result<String, String> {
    let mut reader = BufReader::new(stream);
    let mut status = String::new();
    reader
        .read_line(&mut status)
        .map_err(|e| format!("read status line: {e}"))?;
    let code: u16 = status
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut content_length: Option<usize> = None;
    let mut chunked = false;
    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| format!("read header: {e}"))?;
        if n == 0 {
            break;
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        let lower = line.to_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            content_length = v.trim().parse().ok();
        } else if lower.starts_with("transfer-encoding:") && lower.contains("chunked") {
            chunked = true;
        }
    }
    let mut body = Vec::new();
    if chunked {
        loop {
            let mut size_line = String::new();
            let n = reader
                .read_line(&mut size_line)
                .map_err(|e| format!("read chunk size: {e}"))?;
            if n == 0 {
                break;
            }
            let size = usize::from_str_radix(
                size_line.trim().split(';').next().unwrap_or("0").trim(),
                16,
            )
            .unwrap_or(0);
            if size == 0 {
                let _ = reader.read_line(&mut String::new());
                break;
            }
            let mut chunk = vec![0u8; size];
            reader
                .read_exact(&mut chunk)
                .map_err(|e| format!("read chunk body: {e}"))?;
            body.extend_from_slice(&chunk);
            let _ = reader.read_line(&mut String::new());
        }
    } else if let Some(len) = content_length {
        body.resize(len, 0);
        reader
            .read_exact(&mut body)
            .map_err(|e| format!("read body: {e}"))?;
    } else {
        reader
            .read_to_end(&mut body)
            .map_err(|e| format!("read body: {e}"))?;
    }
    let text = String::from_utf8_lossy(&body).into_owned();
    if code != 200 {
        return Err(format!("llama-server HTTP {code}: {}", first_chars(&text, 500)));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;

    #[test]
    fn parse_extracts_message_content() {
        let resp = r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"Hola! Como estas?"},"finish_reason":"stop"}]}"#;
        assert_eq!(parse_chat_completion(resp).unwrap(), "Hola! Como estas?");
    }

    #[test]
    fn parse_strips_assistant_marker() {
        let resp = r#"{"choices":[{"message":{"content":"Assistant: hey"}}]}"#;
        assert_eq!(parse_chat_completion(resp).unwrap(), "hey");
    }

    #[test]
    fn parse_empty_completion_is_error() {
        let resp = r#"{"choices":[{"message":{"content":""},"finish_reason":"length"}]}"#;
        let err = parse_chat_completion(resp).unwrap_err();
        assert!(err.contains("empty completion"), "unexpected: {err}");
    }

    #[test]
    fn parse_bad_json_is_error() {
        assert!(parse_chat_completion("not json").is_err());
    }

    fn serve_one(responder: impl FnOnce(&mut TcpStream) + Send + 'static) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            responder(&mut stream);
        });
        port
    }

    #[test]
    fn http_client_reads_content_length_body() {
        let port = serve_one(|s| {
            let body = r#"{"choices":[{"message":{"content":"hola"}}]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = s.write_all(resp.as_bytes());
        });
        let body = http_post_json(
            "127.0.0.1",
            port,
            "/v1/chat/completions",
            "{}",
            Duration::from_secs(2),
        )
        .unwrap();
        assert!(body.contains("hola"), "unexpected: {body}");
    }

    #[test]
    fn http_client_reads_chunked_body() {
        let port = serve_one(|s| {
            let resp = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhola!\r\n0\r\n\r\n";
            let _ = s.write_all(resp.as_bytes());
        });
        let body =
            http_post_json("127.0.0.1", port, "/v1/chat/completions", "{}", Duration::from_secs(2))
                .unwrap();
        assert_eq!(body, "hola!");
    }

    #[test]
    fn http_error_status_is_reported() {
        let port = serve_one(|s| {
            let body = "not ready";
            let resp = format!(
                "HTTP/1.1 503 Service Unavailable\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = s.write_all(resp.as_bytes());
        });
        let err = http_get("127.0.0.1", port, "/health", Duration::from_secs(2)).unwrap_err();
        assert!(err.contains("503"), "unexpected: {err}");
    }

    #[test]
    fn connection_refused_is_reported() {
        let err = http_get("127.0.0.1", 1, "/health", Duration::from_secs(2)).unwrap_err();
        assert!(err.contains("connect"), "unexpected: {err}");
    }
}

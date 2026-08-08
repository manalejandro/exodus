//! Run llama.cpp as a subprocess to produce chat completions.
//!
//! exodus does not embed an inference runtime; when `EXODUS_LLAMA_BIN`
//! (default `llama-cli`) is on `PATH`, the node shells out to it for each
//! chat turn.  Generations are stateless one-shot runs built from the
//! conversation transcript.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::config::ExodusConfig;
use crate::gpu::GpuInfo;

/// A single chat turn fed into the prompt.
#[derive(Debug, Clone)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

/// Build a plain-text conversation transcript ending in an `Assistant:`
/// continuation marker (llama.cpp fills in the next assistant turn).
pub fn build_prompt(messages: &[ChatTurn]) -> String {
    let mut out = String::new();
    for m in messages {
        let role = match m.role.as_str() {
            "system" => "System",
            "assistant" => "Assistant",
            _ => "User",
        };
        out.push_str(role);
        out.push_str(": ");
        out.push_str(m.content.trim());
        out.push('\n');
    }
    out.push_str("Assistant:");
    out
}

/// Run one completion against `model_path`.  Returns the generated text, or
/// an error describing why the runtime could not run.
pub fn complete(
    config: &ExodusConfig,
    gpu: &GpuInfo,
    model_path: &Path,
    messages: &[ChatTurn],
) -> Result<String, String> {
    let bin = &config.llama_bin;
    if !binary_exists(bin) {
        return Err(format!(
            "inference runtime '{bin}' not found on PATH (checked model {}; set EXODUS_LLAMA_BIN)",
            model_path.display()
        ));
    }
    if !model_path.is_file() {
        return Err(format!("model file not found: {}", model_path.display()));
    }
    let layers = config
        .gpu_layers
        .unwrap_or(if gpu.available { 99 } else { 0 });

    let mut cmd = Command::new(bin);
    cmd.arg("-m").arg(model_path);
    cmd.arg("-p").arg(build_prompt(messages));
    cmd.arg("-n").arg(config.max_tokens.to_string());
    cmd.arg("--n-gpu-layers").arg(layers.to_string());
    cmd.arg("--temp").arg("0.6");
    cmd.arg("--no-display-prompt");
    // Recent llama-cli builds enable conversation mode by default whenever the
    // model ships a chat template: the `-p` prompt is ignored, the process
    // switches to interactive input and waits (and never exits) when stdout is
    // piped, so a normal chat call hangs until the timeout.  `-no-cnv` (older
    // builds) and `-st`/`--single-turn` (newer builds) both force a single-turn
    // completion that prints to stdout and exits.
    cmd.arg("-no-cnv");
    cmd.arg("-st");
    // Never leave stdin inherited: a daemonised exodus often has an open stdin
    // (docker/systemd), and llama-cli in conversational mode blocks reading it
    // forever, with no stderr output at all until it times out.
    cmd.stdin(Stdio::null());

    let timeout = Duration::from_secs_f64(config.inference_timeout_seconds.max(1.0));
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to start {bin}: {e}"))?;

    // Drain stdout and stderr on background threads while we poll the child.
    // If the pipes are never read until the process exits, a child that fills
    // the OS pipe buffer (~64 KB) blocks forever on write, the parent only
    // ever sees try_wait()==None, and the chat hangs "thinking" until the
    // timeout kills it instead of streaming a reply.
    let out_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    let err_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    let mut drainers = Vec::new();
    if let Some(mut out) = child.stdout.take() {
        let buf = out_buf.clone();
        drainers.push(thread::spawn(move || {
            let mut tmp = Vec::new();
            let _ = out.read_to_end(&mut tmp);
            *buf.lock().unwrap() = tmp;
        }));
    }
    if let Some(mut err) = child.stderr.take() {
        let buf = err_buf.clone();
        drainers.push(thread::spawn(move || {
            let mut tmp = Vec::new();
            let _ = err.read_to_end(&mut tmp);
            *buf.lock().unwrap() = tmp;
        }));
    }

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    for d in drainers {
                        let _ = d.join();
                    }
                    let guard = err_buf.lock().unwrap();
                    let stderr = String::from_utf8_lossy(&guard);
                    let mut tail: Vec<char> = stderr.trim().chars().rev().take(300).collect();
                    tail.reverse();
                    let tail: String = tail.into_iter().collect();
                    return Err(format!(
                        "inference timed out after {}s and was killed ({bin}, model {}){}",
                        config.inference_timeout_seconds,
                        model_path.display(),
                        if tail.trim().is_empty() {
                            String::new()
                        } else {
                            format!("; llama-cli stderr tail: {}", tail.trim())
                        }
                    ));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                for d in drainers {
                    let _ = d.join();
                }
                return Err(format!("waiting on {bin}: {e}"));
            }
        }
    };
    for d in drainers {
        let _ = d.join();
    }
    if !status.success() {
        let guard = err_buf.lock().unwrap();
        let stderr = String::from_utf8_lossy(&guard);
        return Err(format!(
            "{bin} exited with {}: {}",
            status,
            stderr.trim()
        ));
    }
    let guard = out_buf.lock().unwrap();
    let text = String::from_utf8_lossy(&guard);
    let cleaned = clean(&text);
    if cleaned.is_empty() {
        // A successful run that emitted no tokens almost always means the
        // model stopped immediately (e.g. the plain prompt is missing the
        // model's chat-template tokens).  Report the llama stderr tail so the
        // failure is visible in the UI instead of an empty reply.
        let guard = err_buf.lock().unwrap();
        let stderr = String::from_utf8_lossy(&guard);
        return Err(format!(
            "{bin} produced no output for {} (check the model's chat template / context size); stderr tail: {}",
            model_path.display(),
            stderr.trim().chars().rev().take(300).collect::<String>().chars().rev().collect::<String>()
        ));
    }
    Ok(cleaned)
}

/// Trim surrounding whitespace and a leading `Assistant:` continuation marker
/// that some runtimes echo back.
pub(crate) fn clean(text: &str) -> String {
    let out = text.trim();
    let out = out.strip_prefix("Assistant:").map(str::trim).unwrap_or(out);
    out.to_string()
}

/// Whether the runtime binary exists (absolute path or on `PATH`).
pub(crate) fn binary_exists(bin: &str) -> bool {
    if bin.contains('/') || bin.contains('\\') {
        return Path::new(bin).is_file();
    }
    std::env::var("PATH")
        .unwrap_or_default()
        .split(':')
        .map(|dir| Path::new(dir).join(bin))
        .any(|p| p.is_file())
}

/// Parse a model's parameter count from a name like
/// `Llama-3.2-1B-Instruct-4bit`, returning the value of the *last* `N.NB`
/// token that is not a `…bit` precision tag (`4bit` is not 4 billion params).
pub fn model_params_b(model: &str) -> Option<f64> {
    let chars: Vec<char> = model.to_lowercase().chars().collect();
    let n = chars.len();
    let mut best: Option<f64> = None;
    let mut i = 0;
    while i < n {
        if chars[i].is_ascii_digit() {
            let start = i;
            while i < n && chars[i].is_ascii_digit() {
                i += 1;
            }
            if i < n && chars[i] == '.' {
                i += 1;
                while i < n && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let num: String = chars[start..i].iter().collect();
            let mut j = i;
            while j < n && chars[j] == ' ' {
                j += 1;
            }
            let is_bit = j + 2 < n && chars[j] == 'b' && chars[j + 1] == 'i' && chars[j + 2] == 't';
            if j < n && chars[j] == 'b' && !is_bit {
                if let Ok(v) = num.parse::<f64>() {
                    best = Some(v);
                }
            }
        } else {
            i += 1;
        }
    }
    best
}

/// Pick a [`crate::models::Precision`] from a model file name (`4bit`, `8bit`,
/// `2bit`, `int4`, …), defaulting to `fp16`.
pub fn model_precision(model: &str) -> crate::models::Precision {
    let s = model.to_lowercase();
    if s.contains("2bit") || s.contains("2-bit") || s.contains("int2") {
        crate::models::Precision::Int2
    } else if s.contains("8bit") || s.contains("8-bit") || s.contains("int8") {
        crate::models::Precision::Int8
    } else if s.contains("4bit") || s.contains("4-bit") || s.contains("int4") {
        crate::models::Precision::Int4
    } else {
        crate::models::Precision::Fp16
    }
}

/// Very rough `tokens ≈ chars / 4` estimate used when the runtime cannot
/// report exact token counts (llama-cli one-shot runs).
pub fn estimate_tokens(chars: usize) -> i64 {
    ((chars as f64) / 4.0).round().max(1.0) as i64
}

/// Reference FLOPS for a completion, mirroring [`crate::accounting::expected_flops`].
pub fn estimated_flops(
    params_b: f64,
    precision: crate::models::Precision,
    prompt_tokens: i64,
    completion_tokens: i64,
) -> f64 {
    2.0 * params_b * 1e9
        * (prompt_tokens as f64 + completion_tokens as f64 * 2.0)
        * precision.factor()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_b_parsing_ignores_precision_tags() {
        assert_eq!(model_params_b("Llama-3.2-1B-Instruct-4bit"), Some(1.0));
        assert_eq!(model_params_b("Mistral-7B-Instruct-v0.3-4bit"), Some(7.0));
        assert_eq!(model_params_b("Qwen2.5-14B-Instruct-4bit"), Some(14.0));
        assert_eq!(model_params_b("Mixtral-8x7B-Instruct-4bit"), Some(7.0));
        assert_eq!(model_params_b("no model name"), None);
    }

    #[test]
    fn precision_parses_from_name() {
        assert_eq!(model_precision("x-4bit"), crate::models::Precision::Int4);
        assert_eq!(model_precision("x-8bit"), crate::models::Precision::Int8);
        assert_eq!(model_precision("x-2bit"), crate::models::Precision::Int2);
        assert_eq!(model_precision("plain"), crate::models::Precision::Fp16);
    }

    #[test]
    fn prompt_builds_transcript() {
        let turns = vec![
            ChatTurn { role: "user".into(), content: "hi".into() },
            ChatTurn { role: "assistant".into(), content: "hello".into() },
            ChatTurn { role: "user".into(), content: "how are you".into() },
        ];
        let p = build_prompt(&turns);
        assert!(p.contains("User: hi"));
        assert!(p.contains("Assistant: hello"));
        assert!(p.ends_with("Assistant:"));
    }

    #[test]
    fn clean_strips_marker_and_prompt_echo() {
        assert_eq!(clean("  Hello there!  "), "Hello there!");
        assert_eq!(
            clean("User: hi\nAssistant: Hello!"),
            "User: hi\nAssistant: Hello!"
        );
        assert_eq!(clean("Assistant: 42"), "42");
    }

    #[test]
    fn binary_missing_reported() {
        let cfg = ExodusConfig {
            llama_bin: "exodus-definitely-missing-bin-xyz".into(),
            max_tokens: 16,
            ..crate::config::config_from_env()
        };
        let gpu = GpuInfo::default();
        let err = complete(
            &cfg,
            &gpu,
            Path::new("/nonexistent/model.gguf"),
            &[],
        )
        .unwrap_err();
        assert!(err.contains("not found"), "unexpected: {err}");
    }

    fn scratch(name: &str, body: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "exodus-infer-test-{}-{n}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, body).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perm = std::fs::metadata(&path).unwrap().permissions();
            perm.set_mode(0o755);
            std::fs::set_permissions(&path, perm).unwrap();
        }
        path
    }

    #[test]
    fn timed_out_inference_is_killed() {
        let bin = scratch("slow.sh", "#!/bin/sh\nsleep 5\n");
        let model = scratch("model.gguf", "fake model bytes");
        let cfg = ExodusConfig {
            llama_bin: bin.to_string_lossy().into_owned(),
            inference_timeout_seconds: 1.0,
            ..crate::config::config_from_env()
        };
        let err = complete(&cfg, &GpuInfo::default(), &model, &[]).unwrap_err();
        assert!(err.contains("timed out"), "unexpected: {err}");
    }

    #[test]
    fn quick_inference_succeeds() {
        let bin = scratch("fast.sh", "#!/bin/sh\nprintf 'Assistant: hello there\\n'\n");
        let model = scratch("model2.gguf", "bytes");
        let cfg = ExodusConfig {
            llama_bin: bin.to_string_lossy().into_owned(),
            inference_timeout_seconds: 5.0,
            ..crate::config::config_from_env()
        };
        let out = complete(&cfg, &GpuInfo::default(), &model, &[]).unwrap();
        assert_eq!(out, "hello there");
    }
}

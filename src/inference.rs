//! Run llama.cpp as a subprocess to produce chat completions.
//!
//! exodus does not embed an inference runtime; when `EXODUS_LLAMA_BIN`
//! (default `llama-cli`) is on `PATH`, the node shells out to it for each
//! chat turn.  Generations are stateless one-shot runs built from the
//! conversation transcript.

use std::path::Path;
use std::process::Command;

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

    let output = cmd
        .output()
        .map_err(|e| format!("failed to start {bin}: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "{bin} exited with {}: {}",
            output.status,
            stderr.trim()
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    Ok(clean(&text))
}

/// Trim surrounding whitespace and a leading `Assistant:` continuation marker
/// that some runtimes echo back.
fn clean(text: &str) -> String {
    let out = text.trim();
    let out = out.strip_prefix("Assistant:").map(str::trim).unwrap_or(out);
    out.to_string()
}

/// Whether the runtime binary exists (absolute path or on `PATH`).
fn binary_exists(bin: &str) -> bool {
    if bin.contains('/') || bin.contains('\\') {
        return Path::new(bin).is_file();
    }
    std::env::var("PATH")
        .unwrap_or_default()
        .split(':')
        .map(|dir| Path::new(dir).join(bin))
        .any(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

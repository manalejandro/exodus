//! GPU detection and capability reporting.
//!
//! exodus ships without an inference runtime, so the node is CPU-only, but it
//! must be able to see and use an NVIDIA GPU exposed through the Docker
//! NVIDIA Container Toolkit: contributions are tagged with the detected device
//! tier and the API reports the real GPU state instead of a hardcoded
//! "not available".

use std::path::Path;
use std::process::Command;

use serde_json::{json, Value};

use crate::models::DeviceTier;

#[derive(Debug, Clone, Default)]
pub struct GpuDevice {
    pub name: String,
    pub memory_total_mb: u64,
}

/// A process currently running on a detected GPU (from `nvidia-smi`).
#[derive(Debug, Clone, Default)]
pub struct GpuProcess {
    pub pid: u64,
    pub memory_used_mb: u64,
}

#[derive(Debug, Clone, Default)]
pub struct GpuInfo {
    pub available: bool,
    pub driver: Option<String>,
    pub devices: Vec<GpuDevice>,
}

impl GpuInfo {
    /// The device tier this node should advertise for local work.
    pub fn tier(&self) -> DeviceTier {
        if self.available {
            DeviceTier::GpuNvidia
        } else {
            DeviceTier::Cpu
        }
    }

    pub fn tier_string(&self) -> String {
        serde_json::to_value(self.tier())
            .ok()
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_else(|| "cpu".to_string())
    }

    pub fn to_value(&self) -> Value {
        json!({
            "available": self.available,
            "gpu_supported": self.available,
            "tier": self.tier_string(),
            "driver": self.driver,
            "devices": self.devices.iter().map(|d| json!({
                "name": d.name,
                "memory_total_mb": d.memory_total_mb,
            })).collect::<Vec<_>>(),
        })
    }
}

/// Live GPU usage: memory in use, utilization and running processes as
/// reported by `nvidia-smi`.  Returns an empty report when no GPU is visible.
pub fn live_usage(info: &GpuInfo) -> Value {
    if !info.available {
        return json!({
            "available": false,
            "devices": [],
            "processes": [],
        });
    }
    let (used, utilization) = query_gpu_usage();
    let processes = query_gpu_processes();
    json!({
        "available": true,
        "devices": info
            .devices
            .iter()
            .zip(used.iter().zip(utilization.iter()))
            .map(|(d, (used_mb, util))| json!({
                "name": d.name,
                "memory_total_mb": d.memory_total_mb,
                "memory_used_mb": used_mb,
                "utilization_percent": util,
            }))
            .collect::<Vec<_>>(),
        "processes": processes
            .iter()
            .map(|p| json!({ "pid": p.pid, "memory_used_mb": p.memory_used_mb }))
            .collect::<Vec<_>>(),
    })
}

/// Query per-device used memory (MB) and utilization (%) from `nvidia-smi`.
fn query_gpu_usage() -> (Vec<u64>, Vec<u64>) {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(out) = out else {
        return (Vec::new(), Vec::new());
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let mut used = Vec::new();
    let mut util = Vec::new();
    for line in text.lines() {
        let mut fields = line.split(',');
        used.push(fields.next().unwrap_or("0").trim().parse().unwrap_or(0));
        util.push(fields.next().unwrap_or("0").trim().parse().unwrap_or(0));
    }
    (used, util)
}

/// List of processes running on the GPU from `nvidia-smi`.
fn query_gpu_processes() -> Vec<GpuProcess> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(out) = out else {
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines()
        .map(|line| {
            let mut fields = line.split(',').map(|s| s.trim());
            GpuProcess {
                pid: fields.next().and_then(|s| s.parse().ok()).unwrap_or(0),
                memory_used_mb: fields.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            }
        })
        .collect()
}

/// Detect the GPU exposed to this process.
///
/// Detection order: `nvidia-smi` (mounted by the NVIDIA Container Toolkit),
/// then container-runtime hints (`NVIDIA_VISIBLE_DEVICES` / `/dev/nvidiactl`),
/// then an explicit `EXODUS_GPU_LAYERS > 0` request.
pub fn detect(gpu_layers: Option<i64>) -> GpuInfo {
    if let Some(info) = detect_nvidia_smi() {
        if info.available {
            return info;
        }
        // nvidia-smi is present but reported no device; fall through to the
        // container hints before giving up.
    }
    if gpu_visible_in_container() {
        return GpuInfo {
            available: true,
            ..Default::default()
        };
    }
    if gpu_layers.unwrap_or(0) > 0 {
        return GpuInfo {
            available: true,
            ..Default::default()
        };
    }
    GpuInfo::default()
}

/// Query `nvidia-smi` for every visible device.  Returns `Some` whenever the
/// binary exists (so a host with no usable GPU still yields a decisive answer);
/// the info is `available` only if at least one device was reported.
fn detect_nvidia_smi() -> Option<GpuInfo> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let mut info = GpuInfo::default();
    for line in text.lines() {
        let mut fields = line.split(',');
        let name = fields.next().unwrap_or("").trim().to_string();
        let mem = fields.next().unwrap_or("").trim().to_string();
        let driver = fields.next().unwrap_or("").trim().to_string();
        if name.is_empty() {
            continue;
        }
        info.devices.push(GpuDevice {
            name,
            memory_total_mb: mem.parse().unwrap_or(0),
        });
        if info.driver.is_none() && !driver.is_empty() {
            info.driver = Some(driver);
        }
    }
    if !info.devices.is_empty() {
        info.available = true;
    }
    Some(info)
}

/// Container-runtime hints that a GPU is wired into this container.
fn gpu_visible_in_container() -> bool {
    let visible = std::env::var("NVIDIA_VISIBLE_DEVICES").unwrap_or_default();
    let visible = visible.trim();
    if !visible.is_empty() && visible != "void" && visible != "none" {
        return true;
    }
    Path::new("/dev/nvidiactl").exists() || Path::new("/dev/nvidia0").exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_gpu_is_cpu_tier() {
        let info = GpuInfo::default();
        assert_eq!(info.tier(), DeviceTier::Cpu);
        assert!(!info.available);
    }

    #[test]
    fn available_gpu_is_nvidia_tier() {
        let info = GpuInfo {
            available: true,
            driver: Some("560.35.03".to_string()),
            devices: vec![GpuDevice {
                name: "NVIDIA GeForce RTX 4090".to_string(),
                memory_total_mb: 24564,
            }],
        };
        assert_eq!(info.tier(), DeviceTier::GpuNvidia);
        assert_eq!(info.tier_string(), "gpu_nvidia");
        let value = info.to_value();
        assert_eq!(value["available"], true);
        assert_eq!(value["devices"][0]["name"], "NVIDIA GeForce RTX 4090");
    }

    #[test]
    fn gpu_layers_request_treats_node_as_gpu() {
        assert!(detect(Some(35)).available);
        assert!(!detect(None).available);
    }
}

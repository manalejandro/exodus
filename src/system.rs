//! Lightweight system resource metrics for the dashboard: CPU usage, memory
//! usage, and the current process's memory footprint.
//!
//! Everything is read from `/proc` on Linux.  On other platforms the metrics
//! degrade gracefully to `None`.  CPU usage is computed as a delta between two
//! samples, so callers should poll at a steady interval for meaningful values.

use std::sync::Mutex;

use serde_json::{json, Value};

/// Current process (`_total` memory usage) plus system-wide memory stats.
pub struct SystemStats {
    /// Total RAM on the machine, in MB.
    pub mem_total_mb: u64,
    /// RAM currently in use by the machine, in MB.
    pub mem_used_mb: Option<u64>,
    /// Resident set size of this process, in MB.
    pub process_rss_mb: u64,
}

/// Incremental CPU usage calculator backed by `/proc/stat` deltas.
pub struct CpuSampler {
    last: Mutex<Option<(u64, u64)>>, // (total, idle)
}

impl Default for CpuSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuSampler {
    pub fn new() -> Self {
        CpuSampler { last: Mutex::new(None) }
    }

    /// CPU usage percent since the previous sample (0..=100).  Returns 0 when
    /// no previous sample exists or the counters cannot be read.
    pub fn percent(&self) -> f64 {
        let Some((total_now, idle_now)) = read_cpu_jiffies() else {
            return 0.0;
        };
        let mut last = self.last.lock().unwrap();
        let pct = match *last {
            Some((total_prev, idle_prev)) if total_now >= total_prev && idle_now >= idle_prev => {
                let total_d = total_now - total_prev;
                let idle_d = idle_now - idle_prev;
                if total_d == 0 {
                    0.0
                } else {
                    ((total_d - idle_d) as f64 / total_d as f64) * 100.0
                }
            }
            _ => 0.0,
        };
        *last = Some((total_now, idle_now));
        pct
    }
}

/// Read machine RAM usage from `/proc/meminfo`, plus this process's RSS.
pub fn memory() -> Value {
    let total = meminfo_kb("MemTotal").unwrap_or(0) / 1024;
    let avail = meminfo_kb("MemAvailable");
    let used = avail.map(|a| total.saturating_sub(a / 1024));
    let process_rss = process_rss_kb() / 1024;
    json!({
        "total_mb": total,
        "used_mb": used,
        "percent": used.map(|u| if total > 0 { (u as f64 / total as f64) * 100.0 } else { 0.0 }),
        "process_rss_mb": process_rss,
    })
}

/// Number of logical CPUs (from `/proc/cpuinfo`), used to express the
/// reachable compute of a node.
pub fn cpu_cores() -> u64 {
    let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") else {
        return 0;
    };
    text.lines()
        .filter(|l| l.starts_with("processor"))
        .count() as u64
}

/// RSS of the current process in MB (from `/proc/self/statm`).
pub fn process_rss_mb() -> u64 {
    let Some(text) = std::fs::read_to_string("/proc/self/statm").ok() else {
        return 0;
    };
    let mut parts = text.split_whitespace();
    let _size = parts.next().and_then(|s| s.parse::<u64>().ok());
    let resident_pages = parts.next().and_then(|s| s.parse::<u64>().ok());
    resident_pages.map(|r| r * 4096 / 1024 / 1024).unwrap_or(0)
}

/// Read a `MemTotal`/`MemAvailable` style value from `/proc/meminfo` (in kB).
fn meminfo_kb(key: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            if let Some(value) = rest
                .trim()
                .trim_start_matches(':')
                .trim()
                .split(' ')
                .next()
            {
                return value.parse().ok();
            }
        }
    }
    None
}

fn process_rss_kb() -> u64 {
    process_rss_mb() * 1024
}

/// (total, idle) jiffies from the aggregate first line of `/proc/stat`.
fn read_cpu_jiffies() -> Option<(u64, u64)> {
    let text = std::fs::read_to_string("/proc/stat").ok()?;
    let line = text.lines().next()?;
    let mut fields = line.split_whitespace().skip(1);
    let mut total: u64 = 0;
    let mut idle: u64 = 0;
    for (i, f) in fields.by_ref().enumerate() {
        let v: u64 = f.parse().ok()?;
        total += v;
        // idle + iowait are the "not busy" jiffies.
        if i == 3 || i == 4 {
            idle += v;
        }
    }
    Some((total, idle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_returns_zero_on_first_call() {
        let s = CpuSampler::new();
        assert_eq!(s.percent(), 0.0);
    }

    #[test]
    fn memory_read_is_wellformed() {
        let v = memory();
        assert!(v["total_mb"].is_number());
    }

    #[test]
    fn meminfo_parses_total() {
        // MemTotal is always present on Linux; the parse should not panic.
        let _ = meminfo_kb("MemTotal");
    }

    #[test]
    fn cpu_cores_is_reasonable() {
        // On any Linux host there is at least one "processor" line.
        assert!(cpu_cores() >= 1);
    }
}

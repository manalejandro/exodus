//! Lightweight system resource metrics for the dashboard: CPU usage, memory
//! usage, and the current process's memory footprint.
//!
//! On Linux everything is read from `/proc`.  On Windows the metrics come from
//! `kernel32` (`GlobalMemoryStatusEx` / `GetSystemTimes`).  On other platforms
//! the metrics degrade gracefully to zeros.  CPU usage is computed as a delta
//! between two samples, so callers should poll at a steady interval for
//! meaningful values.

use std::sync::Mutex;

use serde_json::{json, Value};

/// Incremental CPU usage calculator backed by (total, idle) jiffies deltas.
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

/// Read machine RAM usage plus this process's RSS.
pub fn memory() -> Value {
    let process_rss = process_rss_mb() / 1024;
    let (total, used) = mem_stats();
    json!({
        "total_mb": total,
        "used_mb": used,
        "percent": used.map(|u| if total > 0 { (u as f64 / total as f64) * 100.0 } else { 0.0 }),
        "process_rss_mb": process_rss,
    })
}

/// Number of logical CPUs available to this process, used to express the
/// reachable compute of a node.
pub fn cpu_cores() -> u64 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u64)
        .unwrap_or(0)
}

/// RSS of the current process in MB (Linux: `/proc/self/statm`).
pub fn process_rss_mb() -> u64 {
    process_rss_kb() / 1024
}

/// (total_mb, used_mb) system memory, best-effort.
#[cfg(target_os = "linux")]
fn mem_stats() -> (u64, Option<u64>) {
    let total = meminfo_kb("MemTotal").unwrap_or(0) / 1024;
    let avail = meminfo_kb("MemAvailable");
    let used = avail.map(|a| total.saturating_sub(a / 1024));
    (total, used)
}

/// (total_mb, used_mb) system memory via `GlobalMemoryStatusEx`.
#[cfg(target_os = "windows")]
fn mem_stats() -> (u64, Option<u64>) {
    if let Some((total, avail)) = win_memory_mb() {
        (total, Some(total.saturating_sub(avail)))
    } else {
        (0, None)
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn mem_stats() -> (u64, Option<u64>) {
    (0, None)
}

/// (total, idle) jiffies since boot.  Linux: aggregate first line of
/// `/proc/stat`.  Windows: `GetSystemTimes`.
#[cfg(target_os = "linux")]
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

#[cfg(target_os = "windows")]
fn read_cpu_jiffies() -> Option<(u64, u64)> {
    win::cpu_jiffies()
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn read_cpu_jiffies() -> Option<(u64, u64)> {
    None
}

/// Read a `MemTotal`/`MemAvailable` style value from `/proc/meminfo` (in kB).
#[cfg(target_os = "linux")]
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

/// RSS of the current process in kB (Linux: `/proc/self/statm`).
#[cfg(target_os = "linux")]
fn process_rss_kb() -> u64 {
    let Some(text) = std::fs::read_to_string("/proc/self/statm").ok() else {
        return 0;
    };
    let mut parts = text.split_whitespace();
    let _size = parts.next().and_then(|s| s.parse::<u64>().ok());
    let resident_pages = parts.next().and_then(|s| s.parse::<u64>().ok());
    resident_pages.map(|r| r * 4096 / 1024).unwrap_or(0)
}

#[cfg(target_os = "windows")]
fn process_rss_kb() -> u64 {
    win::process_rss_kb()
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn process_rss_kb() -> u64 {
    0
}

/// Windows API bindings (kernel32), declared without an external crate so the
/// project stays dependency-light.
#[cfg(target_os = "windows")]
mod win {
    use std::mem::size_of;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct FileTime {
        pub low: u32,
        pub high: u32,
    }

    impl FileTime {
        pub fn as_u64(&self) -> u64 {
            ((self.high as u64) << 32) | (self.low as u64)
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct MemoryStatusEx {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }

    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> i32;
        fn GetSystemTimes(
            lp_idle_time: *mut FileTime,
            lp_kernel_time: *mut FileTime,
            lp_user_time: *mut FileTime,
        ) -> i32;
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut std::ffi::c_void,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }

    /// (total_mb, avail_mb) physical memory.
    pub fn win_memory_mb() -> Option<(u64, u64)> {
        let mut ms = MemoryStatusEx {
            dw_length: size_of::<MemoryStatusEx>() as u32,
            dw_memory_load: 0,
            ull_total_phys: 0,
            ull_avail_phys: 0,
            ull_total_page_file: 0,
            ull_avail_page_file: 0,
            ull_total_virtual: 0,
            ull_avail_virtual: 0,
            ull_avail_extended_virtual: 0,
        };
        let ok = unsafe { GlobalMemoryStatusEx(&mut ms) };
        if ok == 0 {
            return None;
        }
        Some((
            ms.ull_total_phys / 1024 / 1024,
            ms.ull_avail_phys / 1024 / 1024,
        ))
    }

    /// (total, idle) 100 ns units from `GetSystemTimes`.
    pub fn cpu_jiffies() -> Option<(u64, u64)> {
        let mut idle = FileTime::default();
        let mut kernel = FileTime::default();
        let mut user = FileTime::default();
        let ok = unsafe { GetSystemTimes(&mut idle, &mut kernel, &mut user) };
        if ok == 0 {
            return None;
        }
        let idle = idle.as_u64();
        let kernel = kernel.as_u64();
        let user = user.as_u64();
        Some((kernel + user, idle))
    }

    /// RSS of the current process in kB via `K32GetProcessMemoryInfo`.
    pub fn process_rss_kb() -> u64 {
        let mut counters = ProcessMemoryCounters {
            cb: size_of::<ProcessMemoryCounters>() as u32,
            page_fault_count: 0,
            peak_working_set_size: 0,
            working_set_size: 0,
            quota_peak_paged_pool_usage: 0,
            quota_paged_pool_usage: 0,
            quota_peak_non_paged_pool_usage: 0,
            quota_non_paged_pool_usage: 0,
            pagefile_usage: 0,
            peak_pagefile_usage: 0,
        };
        let handle = unsafe { GetCurrentProcess() };
        let ok = unsafe {
            K32GetProcessMemoryInfo(handle, &mut counters, size_of::<ProcessMemoryCounters>() as u32)
        };
        if ok == 0 {
            return 0;
        }
        (counters.working_set_size / 1024) as u64
    }
}

/// (total_mb, used_mb) via the platform backend.
#[cfg(target_os = "windows")]
fn win_memory_mb() -> Option<(u64, u64)> {
    win::win_memory_mb()
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
        assert!(v["process_rss_mb"].is_number());
    }

    #[test]
    fn cpu_cores_is_reasonable() {
        // On any host there is at least one available core.
        assert!(cpu_cores() >= 1);
    }
}

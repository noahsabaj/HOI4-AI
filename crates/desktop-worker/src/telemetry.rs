//! What the PC running the worker is doing: CPU, memory, GPU, disks and network, per
//! process where it matters.
//!
//! The question it answers is the one that could not be answered on 2026-09-24: how much
//! of the second PC the game, the worker and the encoder take while a recording runs.
//! Everything here is read-only. A sampler thread starts with the first request, reads
//! everything once a second, and stops a minute after the last request, so a worker that
//! is never asked pays nothing.
//!
//! Sources, all available to an ordinary user:
//! - processes: the tool-help snapshot, then each process's times, memory and I/O counters;
//! - the system: `GetSystemTimes` and `GlobalMemoryStatusEx`;
//! - GPU per process, disks and network: performance counters (PDH), with English
//!   counter names so a localised Windows reads the same. The GPU Engine counters are what
//!   Task Manager shows per process, for any vendor;
//! - the GPU itself: NVML (nvml.dll, part of the NVIDIA driver), loaded at run time and
//!   skipped where there is none: utilisation, the video encoder, memory, temperature,
//!   power, clocks.

#![cfg(windows)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::{size_of, zeroed};
use std::ptr::null_mut;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::{CloseHandle, FILETIME, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Storage::FileSystem::{
    GetDiskFreeSpaceExW, GetDriveTypeW, GetLogicalDrives,
};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS,
};
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows_sys::Win32::System::Performance::{
    PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData, PdhGetFormattedCounterArrayW,
    PdhOpenQueryW, PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_DOUBLE, PDH_MORE_DATA,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX,
};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
use windows_sys::Win32::System::Threading::{
    GetProcessIoCounters, GetProcessTimes, GetSystemTimes, OpenProcess, IO_COUNTERS,
    PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_VM_READ,
};

/// How often the sampler reads everything.
const PERIOD: Duration = Duration::from_secs(1);
/// How long the sampler keeps going after the last request.
const IDLE: Duration = Duration::from_secs(60);
/// Processes always reported, whatever they cost: the game, the worker, the encoder, the
/// bridge, the sessions and jobs (python) and the compositor. A crash reporter (any name with "crash" in
/// it) is too.
const WATCHED: [&str; 6] = [
    "hoi4.exe",
    "hoi4-desktop-worker.exe",
    "ffmpeg.exe",
    "pwsh.exe",
    "python.exe",
    "dwm.exe",
];

fn watched(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    WATCHED.contains(&lower.as_str()) || lower.contains("crash")
}
/// Besides the watched ones, the busiest processes, this many.
const BUSIEST: usize = 8;
/// PDH's flag to report a percentage above 100, as a sum over engines can be.
const PDH_FMT_NOCAP100: u32 = 0x8000;

fn filetime(t: FILETIME) -> u64 {
    ((t.dwHighDateTime as u64) << 32) | t.dwLowDateTime as u64
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(Some(0)).collect()
}

/// One process at one moment.
#[derive(Clone, Debug)]
struct Process {
    pid: u32,
    parent: u32,
    name: String,
    threads: u32,
    /// Creation time, which tells a reused process id from the process it replaced.
    created: u64,
    /// Kernel plus user time, in 100 ns.
    cpu: u64,
    working_set: u64,
    private: u64,
    read: u64,
    written: u64,
}

fn processes() -> Vec<Process> {
    let mut out = Vec::new();
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if snapshot == INVALID_HANDLE_VALUE || snapshot.is_null() {
            return out;
        }
        let mut entry: PROCESSENTRY32W = zeroed();
        entry.dwSize = size_of::<PROCESSENTRY32W>() as u32;
        let mut more = Process32FirstW(snapshot, &mut entry) != 0;
        while more {
            let len = entry.szExeFile.iter().position(|&c| c == 0).unwrap_or(0);
            let name = String::from_utf16_lossy(&entry.szExeFile[..len]);
            let pid = entry.th32ProcessID;
            if pid > 4 {
                let mut p = Process {
                    pid,
                    parent: entry.th32ParentProcessID,
                    name,
                    threads: entry.cntThreads,
                    created: 0,
                    cpu: 0,
                    working_set: 0,
                    private: 0,
                    read: 0,
                    written: 0,
                };
                let mut handle: HANDLE =
                    OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, 0, pid);
                if handle.is_null() {
                    handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
                }
                if !handle.is_null() {
                    let (mut c, mut e, mut k, mut u) = (zeroed(), zeroed(), zeroed(), zeroed());
                    if GetProcessTimes(handle, &mut c, &mut e, &mut k, &mut u) != 0 {
                        p.created = filetime(c);
                        p.cpu = filetime(k) + filetime(u);
                    }
                    let mut memory: PROCESS_MEMORY_COUNTERS_EX = zeroed();
                    memory.cb = size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
                    if K32GetProcessMemoryInfo(handle, &mut memory as *mut _ as *mut _, memory.cb)
                        != 0
                    {
                        p.working_set = memory.WorkingSetSize as u64;
                        p.private = memory.PrivateUsage as u64;
                    }
                    let mut io: IO_COUNTERS = zeroed();
                    if GetProcessIoCounters(handle, &mut io) != 0 {
                        p.read = io.ReadTransferCount;
                        p.written = io.WriteTransferCount;
                    }
                    CloseHandle(handle);
                }
                out.push(p);
            }
            more = Process32NextW(snapshot, &mut entry) != 0;
        }
        CloseHandle(snapshot);
    }
    out
}

/// Idle, kernel (which includes idle) and user time of all CPUs, in 100 ns.
fn system_times() -> Option<(u64, u64, u64)> {
    unsafe {
        let (mut idle, mut kernel, mut user) = (zeroed(), zeroed(), zeroed());
        (GetSystemTimes(&mut idle, &mut kernel, &mut user) != 0)
            .then(|| (filetime(idle), filetime(kernel), filetime(user)))
    }
}

fn memory() -> serde_json::Value {
    unsafe {
        let mut status: MEMORYSTATUSEX = zeroed();
        status.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
        if GlobalMemoryStatusEx(&mut status) == 0 {
            return serde_json::Value::Null;
        }
        serde_json::json!({
            "total_mb": status.ullTotalPhys / (1 << 20),
            "available_mb": status.ullAvailPhys / (1 << 20),
            "load_percent": status.dwMemoryLoad,
            "commit_mb": (status.ullTotalPageFile - status.ullAvailPageFile) / (1 << 20),
            "commit_limit_mb": status.ullTotalPageFile / (1 << 20),
        })
    }
}

/// Free space on every fixed drive.
fn disks() -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    unsafe {
        let mask = GetLogicalDrives();
        for i in 0..26u32 {
            if mask & (1 << i) == 0 {
                continue;
            }
            let letter = (b'A' + i as u8) as char;
            let root = wide(&format!("{letter}:\\"));
            // DRIVE_FIXED
            if GetDriveTypeW(root.as_ptr()) != 3 {
                continue;
            }
            let (mut free, mut total, mut total_free) = (0u64, 0u64, 0u64);
            if GetDiskFreeSpaceExW(root.as_ptr(), &mut free, &mut total, &mut total_free) != 0 {
                out.push(serde_json::json!({
                    "drive": format!("{letter}:"),
                    "free_gb": round1(free as f64 / 1e9),
                    "total_gb": round1(total as f64 / 1e9),
                }));
            }
        }
    }
    out
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// A performance-counter query with wildcard counters, read as (instance, value) pairs.
struct Counters {
    query: isize,
    counters: Vec<(&'static str, isize)>,
}

impl Counters {
    fn open(paths: &[&'static str]) -> Option<Self> {
        unsafe {
            let mut query = 0isize;
            if PdhOpenQueryW(std::ptr::null(), 0, &mut query) != 0 {
                return None;
            }
            let mut counters = Vec::new();
            for &path in paths {
                let mut counter = 0isize;
                let wide_path = wide(path);
                if PdhAddEnglishCounterW(query, wide_path.as_ptr(), 0, &mut counter) == 0 {
                    counters.push((path, counter));
                }
            }
            PdhCollectQueryData(query);
            Some(Self { query, counters })
        }
    }

    fn collect(&self) {
        unsafe {
            PdhCollectQueryData(self.query);
        }
    }

    /// Every instance's value of the counter added as `path`.
    fn read(&self, path: &str) -> Vec<(String, f64)> {
        let Some(&(_, counter)) = self.counters.iter().find(|(p, _)| *p == path) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        unsafe {
            let (mut size, mut count) = (0u32, 0u32);
            let format = PDH_FMT_DOUBLE | PDH_FMT_NOCAP100;
            let status =
                PdhGetFormattedCounterArrayW(counter, format, &mut size, &mut count, null_mut());
            if status != PDH_MORE_DATA || size == 0 {
                return out;
            }
            // u64 words keep the items' f64 values aligned.
            let mut buffer = vec![0u64; (size as usize).div_ceil(8)];
            let items = buffer.as_mut_ptr() as *mut PDH_FMT_COUNTERVALUE_ITEM_W;
            if PdhGetFormattedCounterArrayW(counter, format, &mut size, &mut count, items) != 0 {
                return out;
            }
            for i in 0..count as usize {
                let item = &*items.add(i);
                // PDH_CSTATUS_VALID_DATA and PDH_CSTATUS_NEW_DATA.
                if item.FmtValue.CStatus > 1 || item.szName.is_null() {
                    continue;
                }
                let len = (0..).take_while(|&k| *item.szName.add(k) != 0).count();
                let name = String::from_utf16_lossy(std::slice::from_raw_parts(item.szName, len));
                out.push((name, item.FmtValue.Anonymous.doubleValue));
            }
        }
        out
    }
}

impl Drop for Counters {
    fn drop(&mut self) {
        unsafe {
            PdhCloseQuery(self.query);
        }
    }
}

const GPU_ENGINE: &str = "\\GPU Engine(*)\\Utilization Percentage";
const GPU_PROCESS_MEMORY: &str = "\\GPU Process Memory(*)\\Dedicated Usage";
const GPU_ADAPTER_MEMORY: &str = "\\GPU Adapter Memory(*)\\Dedicated Usage";
const DISK_READ: &str = "\\PhysicalDisk(*)\\Disk Read Bytes/sec";
const DISK_WRITE: &str = "\\PhysicalDisk(*)\\Disk Write Bytes/sec";
const DISK_BUSY: &str = "\\PhysicalDisk(*)\\% Idle Time";
const NET_IN: &str = "\\Network Interface(*)\\Bytes Received/sec";
const NET_OUT: &str = "\\Network Interface(*)\\Bytes Sent/sec";
const NET_LINK: &str = "\\Network Interface(*)\\Current Bandwidth";

/// The process id and engine type in a GPU Engine instance name, such as
/// `pid_1234_luid_0x00000000_0x0000D1A5_phys_0_eng_0_engtype_3D`.
pub fn gpu_instance(name: &str) -> Option<(u32, String)> {
    let rest = name.strip_prefix("pid_")?;
    let pid = rest.split('_').next()?.parse().ok()?;
    let engine = name
        .split_once("engtype_")
        .map(|(_, e)| e.to_string())
        .unwrap_or_default();
    Some((pid, engine))
}

/// The engine family a GPU Engine type belongs to: `3D`, `Compute_0` and `Cuda` all do
/// shader work; the video engines and copies are reported apart.
pub fn engine_family(engine: &str) -> &'static str {
    let e = engine.to_ascii_lowercase();
    if e.starts_with("videoencode") {
        "encode"
    } else if e.starts_with("videodecode") {
        "decode"
    } else if e.starts_with("copy") {
        "copy"
    } else if e.starts_with("3d") || e.starts_with("compute") || e.starts_with("cuda") {
        "shader"
    } else {
        "other"
    }
}

/// NVML, loaded from the driver when there is one.
/// NVML's functions as used here: each returns 0 on success.
type Init = unsafe extern "C" fn() -> u32;
type Count = unsafe extern "C" fn(*mut u32) -> u32;
type Handle = unsafe extern "C" fn(u32, *mut *mut c_void) -> u32;
type Name = unsafe extern "C" fn(*mut c_void, *mut u8, u32) -> u32;
/// nvmlUtilization_t: gpu, memory.
type Utilization = unsafe extern "C" fn(*mut c_void, *mut [u32; 2]) -> u32;
/// Utilisation and its sampling period, in microseconds.
type EngineUse = unsafe extern "C" fn(*mut c_void, *mut u32, *mut u32) -> u32;
/// nvmlMemory_t: total, free, used.
type Memory = unsafe extern "C" fn(*mut c_void, *mut [u64; 3]) -> u32;
/// A sensor or clock by its NVML number, and its reading.
type Reading = unsafe extern "C" fn(*mut c_void, u32, *mut u32) -> u32;
type Power = unsafe extern "C" fn(*mut c_void, *mut u32) -> u32;

struct Nvml {
    devices: Vec<*mut c_void>,
    name: Name,
    utilization: Utilization,
    encoder: EngineUse,
    decoder: EngineUse,
    memory: Memory,
    temperature: Reading,
    power: Power,
    clock: Reading,
}

// The device handles are NVML's own, valid for the process and usable from any thread.
unsafe impl Send for Nvml {}

impl Nvml {
    fn load() -> Option<Self> {
        unsafe {
            let library = LoadLibraryW(wide("nvml.dll").as_ptr());
            if library.is_null() {
                return None;
            }
            macro_rules! function {
                ($name:literal, $kind:ty) => {{
                    let f = GetProcAddress(library, concat!($name, "\0").as_ptr())?;
                    std::mem::transmute::<unsafe extern "system" fn() -> isize, $kind>(f)
                }};
            }
            let init = function!("nvmlInit_v2", Init);
            if init() != 0 {
                return None;
            }
            let count_fn = function!("nvmlDeviceGetCount_v2", Count);
            let handle_fn = function!("nvmlDeviceGetHandleByIndex_v2", Handle);
            let mut count = 0u32;
            if count_fn(&mut count) != 0 {
                return None;
            }
            let mut devices = Vec::new();
            for i in 0..count.min(8) {
                let mut device = null_mut();
                if handle_fn(i, &mut device) == 0 {
                    devices.push(device);
                }
            }
            Some(Self {
                devices,
                name: function!("nvmlDeviceGetName", Name),
                utilization: function!("nvmlDeviceGetUtilizationRates", Utilization),
                encoder: function!("nvmlDeviceGetEncoderUtilization", EngineUse),
                decoder: function!("nvmlDeviceGetDecoderUtilization", EngineUse),
                memory: function!("nvmlDeviceGetMemoryInfo", Memory),
                temperature: function!("nvmlDeviceGetTemperature", Reading),
                power: function!("nvmlDeviceGetPowerUsage", Power),
                clock: function!("nvmlDeviceGetClockInfo", Reading),
            })
        }
    }

    fn read(&self) -> Vec<serde_json::Value> {
        let mut out = Vec::new();
        for &device in &self.devices {
            unsafe {
                let mut name = [0u8; 96];
                let name = if (self.name)(device, name.as_mut_ptr(), name.len() as u32) == 0 {
                    let len = name.iter().position(|&b| b == 0).unwrap_or(0);
                    String::from_utf8_lossy(&name[..len]).into_owned()
                } else {
                    String::new()
                };
                let mut util = [0u32; 2];
                let util = ((self.utilization)(device, &mut util) == 0).then_some(util);
                let (mut enc, mut dec, mut period) = (0u32, 0u32, 0u32);
                let enc = ((self.encoder)(device, &mut enc, &mut period) == 0).then_some(enc);
                let dec = ((self.decoder)(device, &mut dec, &mut period) == 0).then_some(dec);
                let mut memory = [0u64; 3];
                let memory = ((self.memory)(device, &mut memory) == 0).then_some(memory);
                let mut temperature = 0u32;
                let temperature =
                    ((self.temperature)(device, 0, &mut temperature) == 0).then_some(temperature);
                let mut milliwatts = 0u32;
                let power = ((self.power)(device, &mut milliwatts) == 0)
                    .then(|| round1(milliwatts as f64 / 1000.0));
                // NVML_CLOCK_SM
                let mut sm_clock = 0u32;
                let sm_clock = ((self.clock)(device, 1, &mut sm_clock) == 0).then_some(sm_clock);
                out.push(serde_json::json!({
                    "name": name,
                    "busy_percent": util.map(|u| u[0]),
                    "memory_busy_percent": util.map(|u| u[1]),
                    "encoder_percent": enc,
                    "decoder_percent": dec,
                    "vram_used_mb": memory.map(|m| m[2] / (1 << 20)),
                    "vram_total_mb": memory.map(|m| m[0] / (1 << 20)),
                    "temperature_c": temperature,
                    "power_w": power,
                    "sm_clock_mhz": sm_clock,
                }));
            }
        }
        out
    }
}

/// What the sampler keeps between reads.
struct Previous {
    at: Instant,
    system: Option<(u64, u64, u64)>,
    processes: HashMap<(u32, u64), Process>,
}

/// One window of measurements, from the last two reads.
fn window(
    previous: &Previous,
    now: Instant,
    system: Option<(u64, u64, u64)>,
    current: &[Process],
    counters: Option<&Counters>,
    nvml: Option<&Nvml>,
) -> serde_json::Value {
    let seconds = now.duration_since(previous.at).as_secs_f64().max(1e-3);
    let logical = thread::available_parallelism().map_or(1, |n| n.get());
    let cpu = match (previous.system, system) {
        (Some((i0, k0, u0)), Some((i1, k1, u1))) => {
            let total = (k1 - k0) + (u1 - u0);
            let busy = total.saturating_sub(i1 - i0);
            (total > 0).then(|| busy as f64 / total as f64)
        }
        _ => None,
    };
    // GPU use per process, by engine family, and dedicated memory per process.
    let mut gpu: HashMap<u32, HashMap<&'static str, f64>> = HashMap::new();
    let mut vram: HashMap<u32, f64> = HashMap::new();
    let (mut disk, mut net, mut adapters) = (Vec::new(), Vec::new(), Vec::new());
    if let Some(counters) = counters {
        for (name, value) in counters.read(GPU_ENGINE) {
            if let Some((pid, engine)) = gpu_instance(&name) {
                *gpu.entry(pid)
                    .or_default()
                    .entry(engine_family(&engine))
                    .or_default() += value;
            }
        }
        for (name, value) in counters.read(GPU_PROCESS_MEMORY) {
            if let Some((pid, _)) = gpu_instance(&name) {
                *vram.entry(pid).or_default() += value;
            }
        }
        for (name, value) in counters.read(GPU_ADAPTER_MEMORY) {
            adapters.push(serde_json::json!({"adapter": name, "dedicated_mb": round1(value / (1 << 20) as f64)}));
        }
        let reads: HashMap<String, f64> = counters.read(DISK_READ).into_iter().collect();
        let writes: HashMap<String, f64> = counters.read(DISK_WRITE).into_iter().collect();
        let idle: HashMap<String, f64> = counters.read(DISK_BUSY).into_iter().collect();
        for (name, read) in &reads {
            if name == "_Total" {
                continue;
            }
            disk.push(serde_json::json!({
                "disk": name,
                "read_mb_s": round2(read / 1e6),
                "write_mb_s": round2(writes.get(name).copied().unwrap_or(0.0) / 1e6),
                "busy_percent": idle.get(name).map(|i| round1((100.0 - i).clamp(0.0, 100.0))),
            }));
        }
        let sent: HashMap<String, f64> = counters.read(NET_OUT).into_iter().collect();
        let link: HashMap<String, f64> = counters.read(NET_LINK).into_iter().collect();
        for (name, received) in counters.read(NET_IN) {
            let out = sent.get(&name).copied().unwrap_or(0.0);
            // Adapters with no traffic at all (virtual switches, tunnels at rest) are noise.
            if received + out < 1.0 {
                continue;
            }
            net.push(serde_json::json!({
                "interface": name,
                "in_mbit_s": round2(received * 8.0 / 1e6),
                "out_mbit_s": round2(out * 8.0 / 1e6),
                "link_mbit_s": link.get(&name).map(|b| (b / 1e6).round()),
            }));
        }
    }
    let mut rows: Vec<(f64, serde_json::Value, bool)> = Vec::new();
    for p in current {
        let old = previous.processes.get(&(p.pid, p.created));
        let cores = old.map_or(0.0, |o| p.cpu.saturating_sub(o.cpu) as f64 / 1e7 / seconds);
        let watched = watched(&p.name);
        let engines = gpu.get(&p.pid);
        let mut row = serde_json::json!({
            "pid": p.pid,
            "parent": p.parent,
            "name": p.name,
            "cpu_cores": round2(cores),
            "working_set_mb": p.working_set / (1 << 20),
            "private_mb": p.private / (1 << 20),
            "threads": p.threads,
        });
        if let Some(o) = old {
            row["read_mb_s"] =
                serde_json::json!(round2(p.read.saturating_sub(o.read) as f64 / 1e6 / seconds));
            row["write_mb_s"] = serde_json::json!(round2(
                p.written.saturating_sub(o.written) as f64 / 1e6 / seconds
            ));
        }
        if let Some(engines) = engines {
            for (family, value) in engines {
                if *value > 0.0 {
                    row[format!("gpu_{family}_percent")] = serde_json::json!(round1(*value));
                }
            }
        }
        if let Some(mb) = vram.get(&p.pid) {
            row["vram_mb"] = serde_json::json!((mb / (1 << 20) as f64).round());
        }
        let gpu_busy = engines.map_or(0.0, |e| e.values().copied().fold(0.0, f64::max));
        rows.push((cores + gpu_busy / 100.0, row, watched));
    }
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut shown = Vec::new();
    let mut busiest = 0;
    for (_, row, watched) in rows {
        if watched {
            shown.push(row);
        } else if busiest < BUSIEST {
            busiest += 1;
            shown.push(row);
        }
    }
    serde_json::json!({
        "window_s": round2(seconds),
        "cpu": {
            "logical": logical,
            "busy_percent": cpu.map(|c| round1(c * 100.0)),
            "busy_cores": cpu.map(|c| round2(c * logical as f64)),
        },
        "memory": memory(),
        "gpu": nvml.map(|n| n.read()).unwrap_or_default(),
        "gpu_memory": adapters,
        "processes": shown,
        "disks": disks(),
        "disk_io": disk,
        "network": net,
    })
}

struct Shared {
    latest: Mutex<Option<serde_json::Value>>,
    ready: Condvar,
    last_request: Mutex<Instant>,
    running: Mutex<bool>,
}

fn shared() -> &'static Arc<Shared> {
    static SHARED: OnceLock<Arc<Shared>> = OnceLock::new();
    SHARED.get_or_init(|| {
        Arc::new(Shared {
            latest: Mutex::new(None),
            ready: Condvar::new(),
            last_request: Mutex::new(Instant::now()),
            running: Mutex::new(false),
        })
    })
}

fn sampler(shared: Arc<Shared>) {
    let counters = Counters::open(&[
        GPU_ENGINE,
        GPU_PROCESS_MEMORY,
        GPU_ADAPTER_MEMORY,
        DISK_READ,
        DISK_WRITE,
        DISK_BUSY,
        NET_IN,
        NET_OUT,
        NET_LINK,
    ]);
    let nvml = Nvml::load();
    let snapshot = |processes: Vec<Process>| {
        processes
            .into_iter()
            .map(|p| ((p.pid, p.created), p))
            .collect::<HashMap<_, _>>()
    };
    let mut previous = Previous {
        at: Instant::now(),
        system: system_times(),
        processes: snapshot(processes()),
    };
    loop {
        thread::sleep(PERIOD);
        let idle = shared
            .last_request
            .lock()
            .map(|t| t.elapsed() > IDLE)
            .unwrap_or(true);
        if idle {
            break;
        }
        let now = Instant::now();
        let system = system_times();
        let current = processes();
        if let Some(counters) = &counters {
            counters.collect();
        }
        let value = window(
            &previous,
            now,
            system,
            &current,
            counters.as_ref(),
            nvml.as_ref(),
        );
        if let Ok(mut latest) = shared.latest.lock() {
            *latest = Some(value);
        }
        shared.ready.notify_all();
        previous = Previous {
            at: now,
            system,
            processes: snapshot(current),
        };
    }
    if let Ok(mut running) = shared.running.lock() {
        *running = false;
    }
    if let Ok(mut latest) = shared.latest.lock() {
        *latest = None;
    }
}

/// The latest second of measurements. The first request starts the sampler and waits for
/// its first window, about a second; later ones return at once.
pub fn snapshot(wait: Duration) -> Result<serde_json::Value, String> {
    let shared = shared();
    if let Ok(mut last) = shared.last_request.lock() {
        *last = Instant::now();
    }
    {
        let mut running = shared.running.lock().map_err(|_| "telemetry_lock")?;
        if !*running {
            *running = true;
            let s = Arc::clone(shared);
            thread::Builder::new()
                .name("telemetry".into())
                .spawn(move || sampler(s))
                .map_err(|e| format!("telemetry_thread_failed: {e}"))?;
        }
    }
    let latest = shared.latest.lock().map_err(|_| "telemetry_lock")?;
    let (latest, _) = shared
        .ready
        .wait_timeout_while(latest, wait, |l| l.is_none())
        .map_err(|_| "telemetry_lock")?;
    latest.clone().ok_or_else(|| "telemetry_not_ready".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_engine_instances_name_their_process_and_engine() {
        assert_eq!(
            gpu_instance("pid_1234_luid_0x00000000_0x0000D1A5_phys_0_eng_0_engtype_3D"),
            Some((1234, "3D".to_string()))
        );
        assert_eq!(
            gpu_instance("pid_7_luid_0x0_0x1_phys_0_eng_5_engtype_VideoEncode"),
            Some((7, "VideoEncode".to_string()))
        );
        assert_eq!(gpu_instance("luid_0x0_phys_0"), None);
        assert_eq!(engine_family("3D"), "shader");
        assert_eq!(engine_family("Compute_0"), "shader");
        assert_eq!(engine_family("Cuda"), "shader");
        assert_eq!(engine_family("VideoEncode"), "encode");
        assert_eq!(engine_family("VideoDecode"), "decode");
        assert_eq!(engine_family("Copy"), "copy");
    }

    /// The sampler reads this machine: the worker's own process is always among the
    /// watched ones, with a CPU time and memory, and the system has CPUs and memory.
    #[test]
    fn a_snapshot_reports_this_process() {
        let value = snapshot(Duration::from_secs(5)).expect("telemetry");
        assert!(value["cpu"]["logical"].as_u64().unwrap() >= 1);
        assert!(value["memory"]["total_mb"].as_u64().unwrap() > 0);
        let processes = value["processes"].as_array().unwrap();
        assert!(!processes.is_empty());
        for row in processes {
            assert!(row["pid"].as_u64().is_some());
            assert!(row["cpu_cores"].as_f64().unwrap() >= 0.0);
        }
        assert!(watched("hoi4.exe") && watched("HOI4.EXE") && watched("CrashReporter.exe"));
        assert!(!watched("explorer.exe"));
    }
}

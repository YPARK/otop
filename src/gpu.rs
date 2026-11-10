//! GPUs and their processes, sampled through NVML.

use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::{
    Clock, PcieUtilCounter, PerformanceState, TemperatureSensor,
};
use nvml_wrapper::enums::device::UsedGpuMemory;
use std::collections::{HashMap, VecDeque};

/// How many samples of history each GPU keeps.
pub const HISTORY: usize = 600;

/// One GPU, as last sampled. Anything the driver does not report is `None`.
#[derive(Default, Clone)]
pub struct Gpu {
    pub index: u32,
    pub name: String,
    pub util: Option<u32>,
    pub mem_util: Option<u32>,
    pub mem_used: u64,
    pub mem_total: u64,
    pub temp: Option<u32>,
    pub power: Option<u32>,
    pub power_limit: Option<u32>,
    pub fan: Option<u32>,
    pub clock_gr: Option<u32>,
    pub clock_gr_max: Option<u32>,
    pub clock_mem: Option<u32>,
    pub pcie_tx: Option<u32>,
    pub pcie_rx: Option<u32>,
    pub enc: Option<u32>,
    pub dec: Option<u32>,
    pub pstate: Option<String>,
    /// GPU utilisation, and memory used, in percent, oldest first.
    pub hist_util: VecDeque<f64>,
    pub hist_mem: VecDeque<f64>,
}

impl Gpu {
    pub fn mem_frac(&self) -> f64 {
        if self.mem_total == 0 {
            0.0
        } else {
            self.mem_used as f64 / self.mem_total as f64
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Compute,
    Graphics,
    Both,
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Kind::Compute => "C",
            Kind::Graphics => "G",
            Kind::Both => "C+G",
        }
    }
}

/// A process with memory on a GPU.
#[derive(Clone)]
pub struct Proc {
    pub pid: u32,
    pub gpu: u32,
    pub kind: Kind,
    pub user: String,
    pub mem: Option<u64>,
    pub sm: Option<u32>,
    pub enc: Option<u32>,
    pub dec: Option<u32>,
    pub command: String,
}

pub struct Sampler {
    nvml: Nvml,
    pub driver: String,
    pub cuda: String,
    pub gpus: Vec<Gpu>,
    pub procs: Vec<Proc>,
    /// Per GPU, the time of the last process sample read, so each read
    /// returns only what is newer.
    last_seen: Vec<u64>,
    users: HashMap<u32, String>,
}

impl Sampler {
    pub fn new() -> anyhow::Result<Self> {
        let nvml = Nvml::init().map_err(|e| {
            anyhow::anyhow!("cannot load NVML ({e}); is NVIDIA's driver installed?")
        })?;
        let driver = nvml.sys_driver_version().unwrap_or_else(|_| "?".into());
        let cuda = nvml
            .sys_cuda_driver_version()
            .map(|v| format!("{}.{}", v / 1000, (v % 1000) / 10))
            .unwrap_or_else(|_| "?".into());
        let n = nvml.device_count()?;
        let mut gpus = Vec::with_capacity(n as usize);
        for i in 0..n {
            let name = nvml
                .device_by_index(i)
                .and_then(|d| d.name())
                .unwrap_or_else(|_| "unknown".into());
            gpus.push(Gpu {
                index: i,
                name,
                ..Default::default()
            });
        }
        let mut s = Sampler {
            nvml,
            driver,
            cuda,
            gpus,
            procs: Vec::new(),
            last_seen: vec![0; n as usize],
            users: read_passwd(),
        };
        s.sample();
        Ok(s)
    }

    /// Reads every GPU, and the processes on them, once.
    pub fn sample(&mut self) {
        let mut procs: Vec<Proc> = Vec::new();
        for (i, g) in self.gpus.iter_mut().enumerate() {
            let Ok(d) = self.nvml.device_by_index(i as u32) else {
                continue;
            };
            let u = d.utilization_rates().ok();
            g.util = u.as_ref().map(|u| u.gpu);
            g.mem_util = u.as_ref().map(|u| u.memory);
            if let Ok(m) = d.memory_info() {
                g.mem_used = m.used;
                g.mem_total = m.total;
            }
            g.temp = d.temperature(TemperatureSensor::Gpu).ok();
            g.power = d.power_usage().ok();
            g.power_limit = d.enforced_power_limit().ok();
            g.fan = d.fan_speed(0).ok();
            g.clock_gr = d.clock_info(Clock::Graphics).ok();
            g.clock_gr_max = d.max_clock_info(Clock::Graphics).ok();
            g.clock_mem = d.clock_info(Clock::Memory).ok();
            g.pcie_tx = d.pcie_throughput(PcieUtilCounter::Send).ok();
            g.pcie_rx = d.pcie_throughput(PcieUtilCounter::Receive).ok();
            g.enc = d.encoder_utilization().ok().map(|e| e.utilization);
            g.dec = d.decoder_utilization().ok().map(|e| e.utilization);
            g.pstate = d.performance_state().ok().map(pstate);

            push(&mut g.hist_util, g.util.unwrap_or(0) as f64);
            let mem = 100.0 * g.mem_frac();
            push(&mut g.hist_mem, mem);

            // Per-process SM, encoder and decoder use since the last read.
            let mut util: HashMap<u32, (u32, u32, u32)> = HashMap::new();
            if let Ok(samples) = d.process_utilization_stats(self.last_seen[i]) {
                for s in samples {
                    self.last_seen[i] = self.last_seen[i].max(s.timestamp);
                    let e = util.entry(s.pid).or_default();
                    e.0 = e.0.max(s.sm_util);
                    e.1 = e.1.max(s.enc_util);
                    e.2 = e.2.max(s.dec_util);
                }
            }

            let mut on_gpu: Vec<Proc> = Vec::new();
            let lists = [
                (Kind::Compute, d.running_compute_processes()),
                (Kind::Graphics, d.running_graphics_processes()),
            ];
            for (kind, list) in lists {
                for p in list.unwrap_or_default() {
                    let mem = match p.used_gpu_memory {
                        UsedGpuMemory::Used(b) => Some(b),
                        UsedGpuMemory::Unavailable => None,
                    };
                    if let Some(q) = on_gpu.iter_mut().find(|q| q.pid == p.pid) {
                        if q.kind != kind {
                            q.kind = Kind::Both;
                        }
                        continue;
                    }
                    let u = util.get(&p.pid);
                    on_gpu.push(Proc {
                        pid: p.pid,
                        gpu: i as u32,
                        kind,
                        user: String::new(),
                        mem,
                        sm: u.map(|u| u.0),
                        enc: u.map(|u| u.1),
                        dec: u.map(|u| u.2),
                        command: String::new(),
                    });
                }
            }
            procs.extend(on_gpu);
        }
        for p in &mut procs {
            p.user = proc_user(p.pid, &self.users);
            p.command = proc_command(p.pid);
        }
        self.procs = procs;
    }
}

/// A performance state as nvidia-smi writes it: P0 is fastest, P8 idle.
fn pstate(p: PerformanceState) -> String {
    use PerformanceState::*;
    let n = match p {
        Zero => 0,
        One => 1,
        Two => 2,
        Three => 3,
        Four => 4,
        Five => 5,
        Six => 6,
        Seven => 7,
        Eight => 8,
        Nine => 9,
        Ten => 10,
        Eleven => 11,
        Twelve => 12,
        Thirteen => 13,
        Fourteen => 14,
        Fifteen => 15,
        Unknown => return "P?".into(),
    };
    format!("P{n}")
}

fn push(h: &mut VecDeque<f64>, v: f64) {
    if h.len() == HISTORY {
        h.pop_front();
    }
    h.push_back(v);
}

/// User names by uid, from /etc/passwd; uids not there are shown as numbers.
fn read_passwd() -> HashMap<u32, String> {
    std::fs::read_to_string("/etc/passwd")
        .unwrap_or_default()
        .lines()
        .filter_map(|l| {
            let mut f = l.split(':');
            let name = f.next()?;
            let uid = f.nth(1)?.parse().ok()?;
            Some((uid, name.to_string()))
        })
        .collect()
}

fn proc_user(pid: u32, users: &HashMap<u32, String>) -> String {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap_or_default();
    let uid = status
        .lines()
        .find_map(|l| l.strip_prefix("Uid:"))
        .and_then(|r| r.split_whitespace().next())
        .and_then(|u| u.parse::<u32>().ok());
    match uid {
        Some(u) => users.get(&u).cloned().unwrap_or_else(|| u.to_string()),
        // In a container, or another user's, the process may be out of sight.
        None => "?".into(),
    }
}

fn proc_command(pid: u32) -> String {
    let raw = std::fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
    let cmd: Vec<String> = raw
        .split(|&b| b == 0)
        .filter(|a| !a.is_empty())
        .map(|a| String::from_utf8_lossy(a).into_owned())
        .collect();
    if cmd.is_empty() {
        std::fs::read_to_string(format!("/proc/{pid}/comm"))
            .map(|s| format!("[{}]", s.trim()))
            .unwrap_or_else(|_| "?".into())
    } else {
        cmd.join(" ")
    }
}

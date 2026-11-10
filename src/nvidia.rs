//! NVIDIA's GPUs, and the processes on them, through NVML.

use crate::gpu::{Gpu, GpuUse};
use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::{
    Clock, PcieUtilCounter, PerformanceState, TemperatureSensor,
};
use nvml_wrapper::enums::device::UsedGpuMemory;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// NVML says which processes use each GPU, and how much.
pub const PER_PROCESS: bool = true;

pub struct Source {
    nvml: Nvml,
    /// Per GPU, the time of the last process sample read, so each read
    /// returns only what is newer.
    last_seen: Vec<u64>,
    /// Per GPU, each process's SM use as last read. NVML has no new
    /// sample when asked faster than it samples, or for an idle process;
    /// then the last is kept rather than shown as unknown.
    last_sm: Vec<HashMap<u32, u32>>,
    /// Per GPU, its PCIe throughput as last read, and when: each read
    /// takes NVML 20 ms, so it is read every `PCIE` at most.
    last_pcie: Vec<(Option<u64>, Option<u64>)>,
    pcie_at: Option<Instant>,
}

/// How often PCIe throughput is read.
const PCIE: Duration = Duration::from_secs(2);

impl Source {
    /// NVML, its driver's versions, and its GPUs; or why there are none.
    pub fn open() -> Result<(Self, String, Vec<Gpu>), String> {
        let nvml = Nvml::init().map_err(|_| "no NVIDIA driver".to_string())?;
        let driver = nvml.sys_driver_version().unwrap_or_else(|_| "?".into());
        let cuda = nvml
            .sys_cuda_driver_version()
            .map(|v| format!("{}.{}", v / 1000, (v % 1000) / 10))
            .unwrap_or_else(|_| "?".into());
        let n = nvml.device_count().map_err(|e| e.to_string())?;
        // What does not change, read once.
        let gpus = (0..n)
            .map(|i| {
                let d = nvml.device_by_index(i);
                let d = d.as_ref().ok();
                Gpu {
                    index: i,
                    name: d
                        .and_then(|d| d.name().ok())
                        .unwrap_or_else(|| "unknown".into()),
                    power_limit: d
                        .and_then(|d| d.enforced_power_limit().ok())
                        .map(|mw| mw / 1000),
                    clock_gr_max: d.and_then(|d| d.max_clock_info(Clock::Graphics).ok()),
                    ..Default::default()
                }
            })
            .collect();
        let source = Source {
            nvml,
            last_seen: vec![0; n as usize],
            last_sm: vec![HashMap::new(); n as usize],
            last_pcie: vec![(None, None); n as usize],
            pcie_at: None,
        };
        Ok((source, format!("Driver {driver}  CUDA {cuda}"), gpus))
    }

    pub fn sample(&mut self, gpus: &mut [Gpu], procs: &mut HashMap<u32, GpuUse>) {
        let pcie_due = self.pcie_at.is_none_or(|t| t.elapsed() >= PCIE);
        if pcie_due {
            self.pcie_at = Some(Instant::now());
        }
        for (i, g) in gpus.iter_mut().enumerate() {
            let Ok(d) = self.nvml.device_by_index(i as u32) else {
                continue;
            };
            g.util = d.utilization_rates().ok().map(|u| u.gpu);
            if let Ok(m) = d.memory_info() {
                g.mem_used = m.used;
                g.mem_total = m.total;
            }
            g.temp = d.temperature(TemperatureSensor::Gpu).ok();
            // NVML gives milliwatts and KB/s.
            g.power = d.power_usage().ok().map(|mw| mw / 1000);
            g.fan = d.fan_speed(0).ok();
            g.clock_gr = d.clock_info(Clock::Graphics).ok();
            g.clock_mem = d.clock_info(Clock::Memory).ok();
            if pcie_due {
                let pcie = |c| d.pcie_throughput(c).ok().map(|kb| kb as u64 * 1024);
                self.last_pcie[i] = (pcie(PcieUtilCounter::Send), pcie(PcieUtilCounter::Receive));
            }
            (g.pcie_tx, g.pcie_rx) = self.last_pcie[i];
            g.enc = d.encoder_utilization().ok().map(|e| e.utilization);
            g.dec = d.decoder_utilization().ok().map(|e| e.utilization);
            g.pstate = d.performance_state().ok().map(pstate);

            // Each process's SM use since the last read, if NVML has
            // sampled since.
            if let Ok(samples) = d.process_utilization_stats(self.last_seen[i])
                && !samples.is_empty()
            {
                let sm = &mut self.last_sm[i];
                sm.clear();
                for s in samples {
                    self.last_seen[i] = self.last_seen[i].max(s.timestamp);
                    let e = sm.entry(s.pid).or_default();
                    *e = (*e).max(s.sm_util);
                }
            }
            let sm = &self.last_sm[i];

            let lists = [
                d.running_compute_processes(),
                d.running_graphics_processes(),
            ];
            for p in lists.into_iter().flat_map(Result::unwrap_or_default) {
                let u = procs.entry(p.pid).or_default();
                // A process both computing and drawing is listed twice,
                // with the same memory; count it once on this GPU.
                if u.gpus.last() == Some(&(i as u32)) {
                    continue;
                }
                u.gpus.push(i as u32);
                if let UsedGpuMemory::Used(b) = p.used_gpu_memory {
                    u.mem = Some(u.mem.unwrap_or(0) + b);
                }
                if let Some(&s) = sm.get(&p.pid) {
                    u.sm = Some(u.sm.unwrap_or(0).max(s));
                }
            }
        }
    }
}

/// A performance state as nvidia-smi writes it: P0 is fastest, P8 idle.
fn pstate(p: PerformanceState) -> String {
    match p {
        PerformanceState::Unknown => "P?".into(),
        p => format!("P{}", p.as_c()),
    }
}

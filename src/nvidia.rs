//! NVIDIA's GPUs, and the processes on them, through NVML.

use crate::gpu::{Gpu, GpuUse};
use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::{
    Clock, PcieUtilCounter, PerformanceState, TemperatureSensor,
};
use nvml_wrapper::enums::device::UsedGpuMemory;
use std::collections::HashMap;

pub struct Source {
    nvml: Nvml,
    /// Per GPU, the time of the last process sample read, so each read
    /// returns only what is newer.
    last_seen: Vec<u64>,
}

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
        let gpus = (0..n)
            .map(|i| Gpu {
                index: i,
                name: nvml
                    .device_by_index(i)
                    .and_then(|d| d.name())
                    .unwrap_or_else(|_| "unknown".into()),
                ..Default::default()
            })
            .collect();
        let source = Source {
            nvml,
            last_seen: vec![0; n as usize],
        };
        Ok((source, format!("Driver {driver}  CUDA {cuda}"), gpus))
    }

    pub fn sample(&mut self, gpus: &mut [Gpu], procs: &mut HashMap<u32, GpuUse>) {
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

            // Each process's SM use since the last read.
            let mut sm: HashMap<u32, u32> = HashMap::new();
            if let Ok(samples) = d.process_utilization_stats(self.last_seen[i]) {
                for s in samples {
                    self.last_seen[i] = self.last_seen[i].max(s.timestamp);
                    let e = sm.entry(s.pid).or_default();
                    *e = (*e).max(s.sm_util);
                }
            }

            // A process both computing and drawing is listed twice, with
            // the same memory; count it once on this GPU.
            let mut seen: Vec<u32> = Vec::new();
            let lists = [
                d.running_compute_processes(),
                d.running_graphics_processes(),
            ];
            for p in lists.into_iter().flat_map(Result::unwrap_or_default) {
                if seen.contains(&p.pid) {
                    continue;
                }
                seen.push(p.pid);
                let u = procs.entry(p.pid).or_default();
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

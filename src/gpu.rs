//! GPUs, from whichever source this platform has: NVIDIA's NVML, or on
//! macOS the IORegistry, for Apple's GPUs. Without one, there are none.

use crate::hist::History;
use std::collections::HashMap;

#[cfg(target_os = "macos")]
use crate::apple as source;
#[cfg(not(target_os = "macos"))]
use crate::nvidia as source;

/// One GPU, as last sampled. Anything its source does not report is `None`.
#[derive(Default, Clone)]
pub struct Gpu {
    pub index: u32,
    pub name: String,
    pub util: Option<u32>,
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
    /// Apple's GPUs: how many cores.
    pub cores: Option<u32>,
    pub hist_util: History,
    pub hist_mem: History,
}

impl Gpu {
    pub fn mem_frac(&self) -> f64 {
        crate::sys::frac(self.mem_used, self.mem_total)
    }

    /// Forgets the last sample's readings, keeping what does not change.
    fn clear(&mut self) {
        *self = Gpu {
            index: self.index,
            name: std::mem::take(&mut self.name),
            mem_total: self.mem_total,
            cores: self.cores,
            hist_util: std::mem::take(&mut self.hist_util),
            hist_mem: std::mem::take(&mut self.hist_mem),
            ..Default::default()
        };
    }

    /// Records this sample in the history.
    pub fn record(&mut self) {
        self.hist_util.push(self.util.unwrap_or(0) as f64);
        self.hist_mem.push(100.0 * self.mem_frac());
    }
}

/// A process's use of the GPUs.
#[derive(Default, Clone)]
pub struct GpuUse {
    /// The GPUs it is on, in order.
    pub gpus: Vec<u32>,
    /// GPU memory, summed over them, where the driver reports it.
    pub mem: Option<u64>,
    /// SM use, the most on any of them.
    pub sm: Option<u32>,
}

pub struct Gpus {
    source: Option<source::Source>,
    /// The driver, or why there are no GPUs, for the header.
    pub info: String,
    pub gpus: Vec<Gpu>,
    pub procs: HashMap<u32, GpuUse>,
    /// Whether the source says which processes use the GPUs; Apple's does
    /// not.
    pub per_process: bool,
}

impl Gpus {
    pub fn new() -> Self {
        let (source, info, gpus) = match source::Source::open() {
            Ok((s, info, gpus)) => (Some(s), info, gpus),
            Err(why) => (None, why, Vec::new()),
        };
        let mut g = Gpus {
            source,
            info,
            gpus,
            procs: HashMap::new(),
            per_process: source::PER_PROCESS,
        };
        g.sample();
        g
    }

    pub fn sample(&mut self) {
        if let Some(s) = &mut self.source {
            self.procs.clear();
            // What a GPU's read fails to give is shown as unknown, not as
            // what it was last time.
            for g in &mut self.gpus {
                g.clear();
            }
            s.sample(&mut self.gpus, &mut self.procs);
            for g in &mut self.gpus {
                g.record();
            }
        }
    }
}

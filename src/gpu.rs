//! GPUs, from whichever source this platform has: NVIDIA's NVML, or on
//! macOS the IORegistry, for Apple's GPUs. Without one, there are none.
//!
//! Each source is a module with a `Source` that has
//! `open() -> Result<(Source, String, Vec<Gpu>), String>`, giving the
//! driver's name for the header and the GPUs, or why there are none, and
//! `sample(&mut self, &mut [Gpu], &mut HashMap<u32, GpuUse>)`; and with
//! `PER_PROCESS`, whether it fills in the processes' use. The source is
//! picked here, once, for the platform.

use crate::hist::History;
use std::collections::HashMap;

#[cfg(target_os = "macos")]
use crate::apple as source;
#[cfg(not(target_os = "macos"))]
use crate::nvidia as source;

/// One GPU, as last sampled, in plain units. Anything its source does not
/// report is `None`.
#[derive(Default, Clone)]
pub struct Gpu {
    pub index: u32,
    pub name: String,
    /// Percent.
    pub util: Option<u32>,
    /// Bytes.
    pub mem_used: u64,
    pub mem_total: u64,
    /// Degrees Celsius.
    pub temp: Option<u32>,
    /// Watts.
    pub power: Option<u32>,
    pub power_limit: Option<u32>,
    /// Percent.
    pub fan: Option<u32>,
    /// MHz.
    pub clock_gr: Option<u32>,
    pub clock_gr_max: Option<u32>,
    pub clock_mem: Option<u32>,
    /// PCIe throughput, in bytes per second.
    pub pcie_tx: Option<u64>,
    pub pcie_rx: Option<u64>,
    /// Encoder and decoder use, in percent.
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
            power_limit: self.power_limit,
            clock_gr_max: self.clock_gr_max,
            cores: self.cores,
            hist_util: std::mem::take(&mut self.hist_util),
            hist_mem: std::mem::take(&mut self.hist_mem),
            ..Default::default()
        };
    }

    /// Records this sample in the history.
    fn record(&mut self) {
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
        };
        g.sample();
        g
    }

    /// Whether there are GPUs, and their source says which processes use
    /// them: Apple's does not.
    pub fn per_process(&self) -> bool {
        source::PER_PROCESS && !self.gpus.is_empty()
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

//! GPUs, from whichever source this platform has: NVIDIA's NVML, or on
//! macOS the IORegistry, for Apple's GPUs. Without one, there are none.
//!
//! Each source is a module with a `Source` that has
//! `open() -> Result<(Source, String, Vec<Gpu>), String>`, giving the
//! driver's name for the header and the GPUs, or why there are none, and
//! `sample(&mut self, &mut [Gpu], &mut HashMap<u32, GpuUse>)`; and with
//! `PER_PROCESS`, whether it fills in the processes' use, and `MOST_MS`, the
//! longest it should go between reads, if it has one. The source is picked here, once,
//! for the platform.

use crate::hist::History;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::time::Duration;

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

    /// What does not change, to read a sample into: what a read fails to
    /// give is then unknown, not what it was last time.
    fn fresh(&self) -> Gpu {
        Gpu {
            index: self.index,
            name: self.name.clone(),
            mem_total: self.mem_total,
            power_limit: self.power_limit,
            clock_gr_max: self.clock_gr_max,
            cores: self.cores,
            ..Default::default()
        }
    }

    /// Takes a sample's readings, keeping this GPU's history.
    fn take(&mut self, mut reading: Gpu) {
        reading.hist_util = std::mem::take(&mut self.hist_util);
        reading.hist_mem = std::mem::take(&mut self.hist_mem);
        *self = reading;
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

/// One sample of the GPUs, and of the processes on them.
type Sample = (Vec<Gpu>, HashMap<u32, GpuUse>);

fn read(source: &mut source::Source, gpus: &[Gpu]) -> Sample {
    let mut sample: Sample = (gpus.iter().map(Gpu::fresh).collect(), HashMap::new());
    source.sample(&mut sample.0, &mut sample.1);
    sample
}

/// The GPUs' source, read on a thread of its own, so the screen does not
/// wait for reads that take tens of milliseconds (NVML's PCIe throughput,
/// `ioreg`); the screen takes the newest sample at each refresh.
struct Reader {
    samples: Receiver<Sample>,
    /// How often to read, in milliseconds, as the screen's refresh.
    every: Arc<AtomicU64>,
}

impl Reader {
    fn spawn(mut source: source::Source, gpus: Vec<Gpu>, every: Duration) -> Self {
        let (tx, samples) = channel();
        let every = Arc::new(AtomicU64::new(every.as_millis() as u64));
        let period = every.clone();
        std::thread::spawn(move || {
            let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
            loop {
                let ms = period.load(Ordering::Relaxed);
                let ms = source::MOST_MS.map_or(ms, |most| ms.min(most));
                std::thread::sleep(jittered(ms, &mut seed));
                if tx.send(read(&mut source, &gpus)).is_err() {
                    break;
                }
            }
        });
        Reader { samples, every }
    }
}

/// `ms` give or take 15%, so two programs reading the same GPUs at the
/// same rate drift apart instead of colliding at every read.
fn jittered(ms: u64, seed: &mut u64) -> Duration {
    // xorshift: plenty, for spreading reads.
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    let spread = ms * 3 / 10;
    Duration::from_millis(ms - spread / 2 + *seed % (spread + 1))
}

pub struct Gpus {
    reader: Option<Reader>,
    /// The driver, or why there are no GPUs, for the header.
    pub info: String,
    pub gpus: Vec<Gpu>,
    pub procs: HashMap<u32, GpuUse>,
}

impl Gpus {
    /// The GPUs, read every `every`, or more often where their source
    /// needs it.
    pub fn new(every: Duration) -> Self {
        let mut g = Gpus {
            reader: None,
            info: String::new(),
            gpus: Vec::new(),
            procs: HashMap::new(),
        };
        match source::Source::open() {
            Ok((mut source, info, gpus)) => {
                // The first sample at once, so there is one to show.
                let first = read(&mut source, &gpus);
                g.reader = Some(Reader::spawn(source, gpus.clone(), every));
                g.info = info;
                g.gpus = gpus;
                g.take(first);
            }
            Err(why) => g.info = why,
        }
        g
    }

    /// Whether there are GPUs, and their source says which processes use
    /// them: Apple's does not.
    pub fn per_process(&self) -> bool {
        source::PER_PROCESS && !self.gpus.is_empty()
    }

    fn take(&mut self, (gpus, procs): Sample) {
        for (g, r) in self.gpus.iter_mut().zip(gpus) {
            g.take(r);
            g.record();
        }
        self.procs = procs;
    }

    /// Reads the GPUs every `every` from now on.
    pub fn set_interval(&self, every: Duration) {
        if let Some(r) = &self.reader {
            r.every.store(every.as_millis() as u64, Ordering::Relaxed);
        }
    }

    /// Takes the newest sample the reader has.
    pub fn sample(&mut self) {
        let Some(r) = &self.reader else { return };
        match r.samples.try_iter().last() {
            Some(s) => self.take(s),
            // None yet: the last readings again, so the charts keep time.
            None => self.gpus.iter_mut().for_each(Gpu::record),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::jittered;
    use std::time::Duration;

    #[test]
    fn jitter_stays_within_15_percent() {
        let mut seed = 1;
        for _ in 0..1000 {
            let d = jittered(1000, &mut seed);
            assert!(d >= Duration::from_millis(850) && d <= Duration::from_millis(1150));
        }
    }
}

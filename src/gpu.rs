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
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, channel, sync_channel};
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

/// The GPUs' source, read on a thread of its own when the screen asks, at
/// each refresh: a read that takes long (NVML's PCIe throughput every 2 s)
/// is shown at the next refresh instead of holding up this one.
struct Reader {
    /// Asks for a sample; at most one is asked for at a time.
    ask: SyncSender<()>,
    samples: Receiver<Sample>,
}

/// How long the screen waits for the sample it asked for.
const WAIT: Duration = Duration::from_millis(40);

impl Reader {
    fn spawn(mut source: source::Source, gpus: Vec<Gpu>) -> Self {
        let (ask, asked) = sync_channel(1);
        let (tx, samples) = channel();
        std::thread::spawn(move || {
            while asked.recv().is_ok() {
                if tx.send(read(&mut source, &gpus)).is_err() {
                    break;
                }
            }
        });
        Reader { ask, samples }
    }
}

pub struct Gpus {
    reader: Option<Reader>,
    /// The driver, or why there are no GPUs, for the header.
    pub info: String,
    pub gpus: Vec<Gpu>,
    pub procs: HashMap<u32, GpuUse>,
}

impl Gpus {
    pub fn new() -> Self {
        match source::Source::open() {
            Ok((mut source, info, gpus)) => {
                // The first sample at once, so there is one to show.
                let first = read(&mut source, &gpus);
                let mut g = Gpus {
                    reader: Some(Reader::spawn(source, gpus.clone())),
                    info,
                    gpus,
                    procs: HashMap::new(),
                };
                g.take(first);
                g
            }
            Err(info) => Gpus {
                reader: None,
                info,
                gpus: Vec::new(),
                procs: HashMap::new(),
            },
        }
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

    /// Asks for a sample and takes it, if it comes soon, or else the
    /// newest that has come since; if none, the last readings again, so the
    /// charts keep time.
    pub fn sample(&mut self) {
        let Some(r) = &self.reader else { return };
        let mut newest = r.samples.try_iter().last();
        let mut stopped = false;
        // Asked and answered in time, the sample is in step with the screen.
        if r.ask.try_send(()).is_ok() {
            match r.samples.recv_timeout(WAIT) {
                Ok(s) => newest = Some(s),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => stopped = true,
            }
        }
        if stopped {
            self.stop();
        } else if let Some(s) = newest {
            self.take(s);
        } else {
            self.gpus.iter_mut().for_each(Gpu::record);
        }
    }

    /// The reader has stopped: its last readings are no longer true, so
    /// the GPUs are shown as unknown.
    fn stop(&mut self) {
        self.reader = None;
        self.info += "  (GPUs no longer read)";
        let unknown = self.gpus.iter().map(Gpu::fresh).collect();
        self.take((unknown, HashMap::new()));
    }
}

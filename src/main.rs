//! gputop: monitor NVIDIA GPUs, and the processes on them, in the terminal.
//!
//! GPUs are read through NVML, which is loaded at run time from NVIDIA's
//! driver, and drawn with ratatui.

mod app;
mod gpu;
mod ui;

use clap::Parser;
use std::time::Duration;

/// Monitor NVIDIA GPUs and their processes, like nvtop.
///
/// Keys: q or Esc quits, up/down (or j/k) pick a process, s changes what
/// processes are sorted by, r reverses it, +/- refresh faster or slower,
/// g shows the processes of one GPU at a time, or all, h hides or shows
/// the history charts, and v charts each GPU's utilisation and memory
/// apart, together, or every GPU's in one chart.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Milliseconds between samples.
    #[arg(short = 'd', long, default_value_t = 1000, value_parser = clap::value_parser!(u64).range(100..=60000))]
    interval: u64,
    /// Print one sample of the GPUs and their processes, and exit.
    #[arg(long)]
    once: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let sampler = gpu::Sampler::new()?;
    if args.once {
        app::print_once(&sampler);
        return Ok(());
    }
    let mut terminal = ratatui::init();
    let result = app::App::new(sampler, Duration::from_millis(args.interval)).run(&mut terminal);
    ratatui::restore();
    result
}

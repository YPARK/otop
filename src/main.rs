//! otop: CPUs, memory, GPUs and processes, in the terminal, like btop.
//!
//! CPUs, memory and processes are read through sysinfo. NVIDIA's GPUs are
//! read through NVML, loaded at run time from NVIDIA's driver, and on macOS
//! Apple's GPUs from the IORegistry. All of it is drawn with ratatui.

mod app;
#[cfg(any(target_os = "macos", test))]
mod apple;
mod gpu;
mod hist;
#[cfg(not(target_os = "macos"))]
mod nvidia;
mod sys;
mod ui;

use clap::Parser;
use std::time::Duration;

/// Monitor CPUs, memory, GPUs and processes, like btop.
///
/// Keys: q quits; up/down, PgUp/PgDn pick a process; / searches processes
/// by command, user or PID, Enter keeps the search, Esc clears it; k (or
/// F9) stops the process picked, after asking; s changes what processes are
/// sorted by, r reverses it; g lists every process, those on a GPU, or on
/// one GPU; h hides or shows the charts, and v charts each GPU's
/// utilisation and memory apart, together, or every GPU's in one; +/-
/// refresh faster or slower.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Milliseconds between samples.
    #[arg(short = 'd', long, default_value_t = 1000, value_parser = clap::value_parser!(u64).range(100..=60000))]
    interval: u64,
    /// Print one sample of the system and its busiest processes, and exit.
    #[arg(long)]
    once: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let app = app::App::new(
        sys::Sys::new(),
        gpu::Gpus::new(),
        Duration::from_millis(args.interval),
    );
    if args.once {
        // Piped into `head`, say, stdout may close early; that is not an error.
        return match app::print_once(&app, &mut std::io::stdout().lock()) {
            Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
            r => Ok(r?),
        };
    }
    let mut app = app;
    let mut terminal = ratatui::init();
    let result = app.run(&mut terminal);
    ratatui::restore();
    result
}

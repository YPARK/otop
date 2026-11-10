//! The app's state, and the keys that change it.

use crate::gpu::{GpuUse, Gpus};
use crate::sys::Sys;
use crate::ui;
use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::widgets::TableState;
use std::io::{self, Write};
use std::time::{Duration, Instant};
use sysinfo::Signal;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SortBy {
    Cpu,
    Mem,
    GpuMem,
    Sm,
    Pid,
    User,
}

impl SortBy {
    pub fn label(self) -> &'static str {
        match self {
            SortBy::Cpu => "CPU",
            SortBy::Mem => "memory",
            SortBy::GpuMem => "GPU memory",
            SortBy::Sm => "SM",
            SortBy::Pid => "PID",
            SortBy::User => "user",
        }
    }

    /// The next to sort by; GPU memory and SM only where they are known.
    fn next(self, gpu: bool) -> Self {
        match self {
            SortBy::Cpu => SortBy::Mem,
            SortBy::Mem if gpu => SortBy::GpuMem,
            SortBy::Mem => SortBy::Pid,
            SortBy::GpuMem => SortBy::Sm,
            SortBy::Sm => SortBy::Pid,
            SortBy::Pid => SortBy::User,
            SortBy::User => SortBy::Cpu,
        }
    }
}

/// How the GPUs' history is charted.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Charts {
    /// Two charts for each GPU: its utilisation, and its memory.
    Split,
    /// One chart for each GPU, of both.
    Together,
    /// One chart of every GPU's utilisation, and one of their memory.
    Combined,
}

impl Charts {
    pub fn next(self) -> Self {
        match self {
            Charts::Split => Charts::Together,
            Charts::Together => Charts::Combined,
            Charts::Combined => Charts::Split,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Charts::Split => "split",
            Charts::Together => "per GPU",
            Charts::Combined => "combined",
        }
    }
}

/// Which processes are listed.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Only {
    All,
    AnyGpu,
    Gpu(u32),
}

/// A process's row: its CPU and memory, and its use of the GPUs.
#[derive(Clone)]
pub struct Row {
    pub pid: u32,
    pub user: String,
    pub cpu: f32,
    pub mem: u64,
    pub gpu: Option<GpuUse>,
    pub command: String,
    pub start: u64,
}

/// What keys do just now.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Normal,
    /// Typing a search, after `/`.
    Search,
    /// Asked whether to stop this process: its PID, and when it started.
    Kill(u32, u64),
}

pub struct App {
    pub sys: Sys,
    pub gpus: Gpus,
    pub interval: Duration,
    pub sort: SortBy,
    /// Largest first, unless reversed.
    pub descending: bool,
    pub only: Only,
    pub show_charts: bool,
    pub charts: Charts,
    pub mode: Mode,
    /// Processes whose command, user or PID contain this are listed.
    pub query: String,
    /// The processes listed, filtered and sorted, as of the last sample.
    pub rows: Vec<Row>,
    /// The process picked, kept by PID as rows are sorted around it.
    pub selected: Option<u32>,
    pub table: TableState,
    /// What the last kill did, shown for a few seconds.
    pub message: Option<(String, Instant)>,
    quit: bool,
}

impl App {
    pub fn new(sys: Sys, gpus: Gpus, interval: Duration) -> Self {
        let mut app = App {
            sys,
            gpus,
            interval,
            sort: SortBy::Cpu,
            descending: true,
            only: Only::All,
            show_charts: true,
            charts: Charts::Split,
            mode: Mode::Normal,
            query: String::new(),
            rows: Vec::new(),
            selected: None,
            table: TableState::default(),
            message: None,
            quit: false,
        };
        app.update_rows();
        app
    }

    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        let mut last = Instant::now();
        while !self.quit {
            terminal.draw(|f| ui::draw(f, self))?;
            let wait = (last + self.interval).saturating_duration_since(Instant::now());
            if event::poll(wait)? {
                // Take every key waiting, as when a search is pasted, and
                // rebuild the rows once for all of them.
                loop {
                    if let Event::Key(k) = event::read()?
                        && k.kind == KeyEventKind::Press
                    {
                        self.key(k);
                    }
                    if self.quit || !event::poll(Duration::ZERO)? {
                        break;
                    }
                }
                self.update_rows();
            }
            // Measured from the last sample, so a new interval applies now.
            if last.elapsed() >= self.interval {
                self.sample();
                last = Instant::now();
            }
        }
        Ok(())
    }

    fn sample(&mut self) {
        self.sys.sample();
        self.gpus.sample();
        self.update_rows();
    }

    fn key(&mut self, k: KeyEvent) {
        if k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL) {
            self.quit = true;
            return;
        }
        match self.mode {
            Mode::Search => match k.code {
                KeyCode::Enter => self.mode = Mode::Normal,
                KeyCode::Esc => {
                    self.query.clear();
                    self.mode = Mode::Normal;
                }
                KeyCode::Backspace => {
                    self.query.pop();
                }
                KeyCode::Up => self.step(-1),
                KeyCode::Down => self.step(1),
                KeyCode::Char(c) => self.query.push(c),
                _ => {}
            },
            Mode::Kill(pid, start) => {
                // Only these two; any other key, Enter too, cancels.
                let signal = match k.code {
                    KeyCode::Char('t') => Some(Signal::Term),
                    KeyCode::Char('9') => Some(Signal::Kill),
                    _ => None,
                };
                if let Some(s) = signal {
                    let text = match self.sys.signal(pid, start, s) {
                        Ok(()) => format!("sent {s:?} to {pid}"),
                        Err(e) => e,
                    };
                    self.message = Some((text, Instant::now()));
                }
                self.mode = Mode::Normal;
            }
            Mode::Normal => self.normal_key(k.code),
        }
    }

    fn normal_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char('q') => self.quit = true,
            KeyCode::Esc if !self.query.is_empty() => self.query.clear(),
            KeyCode::Esc => self.quit = true,
            KeyCode::Char('/') => self.mode = Mode::Search,
            KeyCode::Char('k') | KeyCode::F(9) => {
                if let Some(r) = self.selected_index().map(|i| &self.rows[i]) {
                    self.mode = Mode::Kill(r.pid, r.start);
                }
            }
            KeyCode::Down | KeyCode::Char('j') => self.step(1),
            KeyCode::Up => self.step(-1),
            KeyCode::PageDown => self.step(20),
            KeyCode::PageUp => self.step(-20),
            KeyCode::Home => self.step(isize::MIN / 2),
            KeyCode::End => self.step(isize::MAX / 2),
            KeyCode::Char('s') => self.sort = self.sort.next(self.gpus.per_process),
            KeyCode::Char('r') => self.descending = !self.descending,
            KeyCode::Char('h') => self.show_charts = !self.show_charts,
            KeyCode::Char('v') => self.charts = self.charts.next(),
            KeyCode::Char('g') if self.gpus.per_process => {
                let n = self.gpus.gpus.len() as u32;
                self.only = match self.only {
                    Only::All if n > 0 => Only::AnyGpu,
                    Only::AnyGpu if n > 1 => Only::Gpu(0),
                    Only::Gpu(i) if i + 1 < n => Only::Gpu(i + 1),
                    _ => Only::All,
                };
            }
            KeyCode::Char('+') | KeyCode::Char('=') => {
                self.interval = (self.interval / 2).max(Duration::from_millis(100));
            }
            KeyCode::Char('-') => {
                self.interval = (self.interval * 2).min(Duration::from_secs(60));
            }
            _ => {}
        }
    }

    /// Moves the selection `by` rows, staying within the list.
    fn step(&mut self, by: isize) {
        if self.rows.is_empty() {
            return;
        }
        let i = self
            .selected_index()
            .map_or(0, |i| i.saturating_add_signed(by).min(self.rows.len() - 1));
        self.selected = Some(self.rows[i].pid);
    }

    pub fn selected_index(&self) -> Option<usize> {
        let pid = self.selected?;
        self.rows.iter().position(|r| r.pid == pid)
    }

    /// Rebuilds the rows, from the last sample, as filtered and sorted now.
    fn update_rows(&mut self) {
        let query = self.query.to_lowercase();
        let gpus = &self.gpus.procs;
        let mut rows: Vec<Row> = self
            .sys
            .procs
            .iter()
            .map(|p| Row {
                pid: p.pid,
                user: p.user.clone(),
                cpu: p.cpu,
                mem: p.mem,
                gpu: gpus.get(&p.pid).cloned(),
                command: p.command.clone(),
                start: p.start,
            })
            .filter(|r| match (self.only, &r.gpu) {
                (Only::All, _) => true,
                (Only::AnyGpu, g) => g.is_some(),
                (Only::Gpu(i), Some(g)) => g.gpus.contains(&i),
                (Only::Gpu(_), None) => false,
            })
            .filter(|r| {
                query.is_empty()
                    || r.command.to_lowercase().contains(&query)
                    || r.user.to_lowercase().contains(&query)
                    || r.pid.to_string().contains(&query)
            })
            .collect();
        let gmem = |r: &Row| r.gpu.as_ref().and_then(|g| g.mem);
        let sm = |r: &Row| r.gpu.as_ref().and_then(|g| g.sm);
        rows.sort_by(|a, b| {
            let o = match self.sort {
                SortBy::Cpu => a.cpu.total_cmp(&b.cpu),
                SortBy::Mem => a.mem.cmp(&b.mem),
                SortBy::GpuMem => gmem(a).cmp(&gmem(b)),
                SortBy::Sm => sm(a).cmp(&sm(b)),
                SortBy::Pid => a.pid.cmp(&b.pid),
                SortBy::User => a.user.cmp(&b.user),
            }
            .then(a.pid.cmp(&b.pid));
            if self.descending { o.reverse() } else { o }
        });
        self.rows = rows;
        // Keep the process picked, or pick the first where it is gone.
        if self.selected_index().is_none() {
            self.selected = self.rows.first().map(|r| r.pid);
        }
    }
}

/// `--once`: the CPUs, memory, GPUs and busiest processes, as plain text.
pub fn print_once(app: &App, out: &mut impl Write) -> io::Result<()> {
    let (sys, gpus) = (&app.sys, &app.gpus);
    writeln!(
        out,
        "{}  {}  CPU {:.0}%  MEM {}/{}  SWAP {}/{}",
        sys.host,
        sys.cpu_brand,
        sys.cpu,
        ui::bytes(sys.mem_used),
        ui::bytes(sys.mem_total),
        ui::bytes(sys.swap_used),
        ui::bytes(sys.swap_total),
    )?;
    writeln!(out, "{}", gpus.info)?;
    for g in &gpus.gpus {
        writeln!(
            out,
            "GPU{} {:<28} util {:>3}%  mem {}/{}  {}  {}",
            g.index,
            g.name,
            ui::opt(g.util),
            ui::bytes(g.mem_used),
            ui::bytes(g.mem_total),
            g.temp.map_or("-".into(), |t| format!("{t}°C")),
            g.power.map_or("-".into(), |p| format!("{}W", p / 1000)),
        )?;
    }
    writeln!(out)?;
    writeln!(
        out,
        "{:>8} {:<10} {:>6} {:>8} {:>5} {:>8} {:>4}  COMMAND",
        "PID", "USER", "CPU%", "MEM", "GPU", "GPU MEM", "SM%"
    )?;
    for r in app.rows.iter().take(20) {
        let g = r.gpu.as_ref();
        writeln!(
            out,
            "{:>8} {:<10} {:>6.1} {:>8} {:>5} {:>8} {:>4}  {}",
            r.pid,
            r.user,
            r.cpu,
            ui::bytes(r.mem),
            g.map_or("-".into(), ui::gpu_list),
            g.and_then(|g| g.mem).map_or("-".into(), ui::bytes),
            ui::opt(g.and_then(|g| g.sm)),
            r.command
        )?;
    }
    Ok(())
}

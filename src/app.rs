//! The app's state, and the keys that change it.

use crate::gpu::{Proc, Sampler};
use crate::ui;
use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::widgets::TableState;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SortBy {
    Mem,
    Sm,
    Pid,
    Gpu,
    User,
}

impl SortBy {
    pub fn label(self) -> &'static str {
        match self {
            SortBy::Mem => "memory",
            SortBy::Sm => "SM",
            SortBy::Pid => "PID",
            SortBy::Gpu => "GPU",
            SortBy::User => "user",
        }
    }

    fn next(self) -> Self {
        match self {
            SortBy::Mem => SortBy::Sm,
            SortBy::Sm => SortBy::Pid,
            SortBy::Pid => SortBy::Gpu,
            SortBy::Gpu => SortBy::User,
            SortBy::User => SortBy::Mem,
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

pub struct App {
    pub sampler: Sampler,
    pub interval: Duration,
    pub sort: SortBy,
    /// Largest first, unless reversed.
    pub descending: bool,
    /// Processes of this GPU alone, or of all.
    pub only_gpu: Option<u32>,
    pub show_charts: bool,
    pub charts: Charts,
    pub table: TableState,
    quit: bool,
}

impl App {
    pub fn new(sampler: Sampler, interval: Duration) -> Self {
        App {
            sampler,
            interval,
            sort: SortBy::Mem,
            descending: true,
            only_gpu: None,
            show_charts: true,
            charts: Charts::Split,
            table: TableState::default().with_selected(Some(0)),
            quit: false,
        }
    }

    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        let mut next = Instant::now() + self.interval;
        while !self.quit {
            terminal.draw(|f| ui::draw(f, self))?;
            let wait = next.saturating_duration_since(Instant::now());
            if event::poll(wait)?
                && let Event::Key(k) = event::read()?
                && k.kind == KeyEventKind::Press
            {
                self.key(k.code, k.modifiers);
            }
            if Instant::now() >= next {
                self.sampler.sample();
                next = Instant::now() + self.interval;
            }
        }
        Ok(())
    }

    fn key(&mut self, code: KeyCode, mods: KeyModifiers) {
        let n = self.processes().len();
        match code {
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            KeyCode::Char('c') if mods.contains(KeyModifiers::CONTROL) => self.quit = true,
            KeyCode::Down | KeyCode::Char('j') => self.table.select_next(),
            KeyCode::Up | KeyCode::Char('k') => self.table.select_previous(),
            KeyCode::Home => self.table.select_first(),
            KeyCode::End => self.table.select(Some(n.saturating_sub(1))),
            KeyCode::Char('s') => self.sort = self.sort.next(),
            KeyCode::Char('r') => self.descending = !self.descending,
            KeyCode::Char('h') => self.show_charts = !self.show_charts,
            KeyCode::Char('v') => self.charts = self.charts.next(),
            KeyCode::Char('g') => {
                let count = self.sampler.gpus.len() as u32;
                self.only_gpu = match self.only_gpu {
                    None if count > 0 => Some(0),
                    Some(i) if i + 1 < count => Some(i + 1),
                    _ => None,
                };
                self.table.select(Some(0));
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

    /// The processes shown, filtered and sorted.
    pub fn processes(&self) -> Vec<Proc> {
        let mut ps: Vec<Proc> = self
            .sampler
            .procs
            .iter()
            .filter(|p| self.only_gpu.is_none_or(|g| p.gpu == g))
            .cloned()
            .collect();
        ps.sort_by(|a, b| {
            let o = match self.sort {
                SortBy::Mem => a.mem.cmp(&b.mem),
                SortBy::Sm => a.sm.cmp(&b.sm),
                SortBy::Pid => a.pid.cmp(&b.pid),
                SortBy::Gpu => a.gpu.cmp(&b.gpu),
                SortBy::User => a.user.cmp(&b.user),
            }
            .then(a.pid.cmp(&b.pid));
            if self.descending { o.reverse() } else { o }
        });
        ps
    }
}

/// `--once`: the GPUs and their processes, printed as plain text.
pub fn print_once(s: &Sampler) {
    println!("Driver {}  CUDA {}", s.driver, s.cuda);
    for g in &s.gpus {
        println!(
            "GPU{} {:<28} util {:>3}%  mem {}/{}  {}  {}",
            g.index,
            g.name,
            ui::opt(g.util),
            ui::bytes(g.mem_used),
            ui::bytes(g.mem_total),
            g.temp.map_or("-".into(), |t| format!("{t}°C")),
            g.power.map_or("-".into(), |p| format!("{}W", p / 1000)),
        );
    }
    if !s.procs.is_empty() {
        println!();
        println!(
            "{:>8} {:>3} {:>3} {:<10} {:>9} {:>4}  COMMAND",
            "PID", "GPU", "TYP", "USER", "MEM", "SM%"
        );
        for p in &s.procs {
            println!(
                "{:>8} {:>3} {:>3} {:<10} {:>9} {:>4}  {}",
                p.pid,
                p.gpu,
                p.kind.label(),
                p.user,
                p.mem.map_or("-".into(), ui::bytes),
                ui::opt(p.sm),
                p.command
            );
        }
    }
}

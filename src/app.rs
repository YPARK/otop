//! The app's state, and the keys that change it.

use crate::gpu::{GpuUse, Gpus};
use crate::sys::{Proc, Sys};
use crate::ui;
use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use std::time::{Duration, Instant};
use sysinfo::Signal;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Only {
    All,
    AnyGpu,
    Gpu(u32),
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
    /// Processes whose PID, user or command contain this are listed.
    pub query: String,
    /// The processes listed, as indices into `sys.procs`, filtered and
    /// sorted as of the last sample.
    pub rows: Vec<usize>,
    /// The process picked, kept by PID as rows are sorted around it, and
    /// its place in `rows`.
    selected: Option<(u32, usize)>,
    /// The first row on screen.
    pub offset: usize,
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
            offset: 0,
            message: None,
            quit: false,
        };
        app.update_rows();
        app
    }

    /// The process listed in `row`, and its use of the GPUs.
    pub fn row(&self, row: usize) -> (&Proc, Option<&GpuUse>) {
        let p = &self.sys.procs[self.rows[row]];
        (p, self.gpus.procs.get(&p.pid))
    }

    /// The row picked.
    pub fn selected(&self) -> Option<usize> {
        self.selected.map(|(_, i)| i)
    }

    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        let mut last = Instant::now();
        while !self.quit {
            terminal.draw(|f| ui::draw(f, self))?;
            let wait = (last + self.interval).saturating_duration_since(Instant::now());
            if event::poll(wait)? {
                // Take every key waiting, as when a search is pasted, and
                // rebuild the rows once, if they changed what is listed.
                let before = self.listing();
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
                if self.listing() != before {
                    self.update_rows();
                }
            }
            // Measured from the last sample, so a new interval applies now.
            if last.elapsed() >= self.interval {
                self.sample();
                last = Instant::now();
            }
        }
        Ok(())
    }

    /// What decides which processes are listed, and in what order.
    fn listing(&self) -> (String, SortBy, bool, Only) {
        (self.query.clone(), self.sort, self.descending, self.only)
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
            // Clears the search, and does nothing else, so pressing it
            // once too often to leave a search does not quit.
            KeyCode::Esc => self.query.clear(),
            KeyCode::Char('/') => self.mode = Mode::Search,
            KeyCode::Char('k') | KeyCode::F(9) => {
                if let Some(i) = self.selected() {
                    let p = self.row(i).0;
                    if crate::sys::signalable(p.pid) {
                        self.mode = Mode::Kill(p.pid, p.start);
                    } else {
                        let text = format!("process {} cannot be signalled", p.pid);
                        self.message = Some((text, Instant::now()));
                    }
                }
            }
            KeyCode::Down => self.step(1),
            KeyCode::Up => self.step(-1),
            KeyCode::PageDown => self.step(20),
            KeyCode::PageUp => self.step(-20),
            KeyCode::Home => self.step(isize::MIN / 2),
            KeyCode::End => self.step(isize::MAX / 2),
            KeyCode::Char('s') => self.sort = self.sort.next(self.gpus.per_process()),
            KeyCode::Char('r') => self.descending = !self.descending,
            KeyCode::Char('h') => self.show_charts = !self.show_charts,
            KeyCode::Char('v') => self.charts = self.charts.next(),
            KeyCode::Char('g') if self.gpus.per_process() => {
                let n = self.gpus.gpus.len() as u32;
                self.only = match self.only {
                    Only::All => Only::AnyGpu,
                    Only::AnyGpu if n > 1 => Only::Gpu(0),
                    Only::Gpu(i) if i + 1 < n => Only::Gpu(i + 1),
                    _ => Only::All,
                };
            }
            KeyCode::Char('+') | KeyCode::Char('=') => self.interval = faster(self.interval),
            KeyCode::Char('-') => self.interval = slower(self.interval),
            _ => {}
        }
    }

    /// Picks the row `i`, or none where there are no rows.
    fn select(&mut self, i: usize) {
        self.selected = self.rows.get(i).map(|&p| (self.sys.procs[p].pid, i));
    }

    /// Moves the selection `by` rows, staying within the list.
    fn step(&mut self, by: isize) {
        let last = self.rows.len().saturating_sub(1);
        let i = self
            .selected()
            .map_or(0, |i| i.saturating_add_signed(by).min(last));
        self.select(i);
    }

    /// Rebuilds the rows, from the last sample, as filtered and sorted now.
    fn update_rows(&mut self) {
        let query = self.query.to_lowercase();
        let (procs, gpus) = (&self.sys.procs, &self.gpus.procs);
        let gpu = |p: &Proc| gpus.get(&p.pid);
        let mut rows: Vec<usize> = (0..procs.len())
            .filter(|&i| {
                let p = &procs[i];
                let on = match (self.only, gpu(p)) {
                    (Only::All, _) => true,
                    (Only::AnyGpu, g) => g.is_some(),
                    (Only::Gpu(i), g) => g.is_some_and(|g| g.gpus.contains(&i)),
                };
                on && p.search.contains(&query)
            })
            .collect();
        let gmem = |p: &Proc| gpu(p).and_then(|g| g.mem);
        let sm = |p: &Proc| gpu(p).and_then(|g| g.sm);
        rows.sort_by(|&a, &b| {
            let (a, b) = (&procs[a], &procs[b]);
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
        // Keep the process picked where it still is; where it has ended,
        // the row it was in, so the selection does not jump to the top.
        let (pid, was) = self.selected.unzip();
        let i = rows
            .iter()
            .position(|&r| Some(procs[r].pid) == pid)
            .unwrap_or_else(|| was.unwrap_or(0).min(rows.len().saturating_sub(1)));
        self.rows = rows;
        self.select(i);
    }
}

/// The refresh intervals `+` and `-` step through, in milliseconds.
const STEPS: [u64; 9] = [100, 250, 500, 1000, 2000, 5000, 10000, 30000, 60000];

/// The next step faster than `d`, or `d` where there is none.
fn faster(d: Duration) -> Duration {
    let ms = d.as_millis() as u64;
    let step = STEPS.iter().rev().find(|&&s| s < ms);
    step.map_or(d, |&s| Duration::from_millis(s))
}

/// The next step slower than `d`, or `d` where there is none.
fn slower(d: Duration) -> Duration {
    let ms = d.as_millis() as u64;
    let step = STEPS.iter().find(|&&s| s > ms);
    step.map_or(d, |&s| Duration::from_millis(s))
}

#[cfg(test)]
mod tests {
    use super::{faster, slower};
    use std::time::Duration;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn steps_come_back_to_a_second() {
        assert_eq!(faster(slower(ms(1000))), ms(1000));
        assert_eq!(slower(faster(ms(1000))), ms(1000));
        assert_eq!(slower(slower(faster(faster(ms(1000))))), ms(1000));
    }

    #[test]
    fn steps_stop_at_the_ends() {
        assert_eq!(faster(ms(100)), ms(100));
        assert_eq!(slower(ms(60000)), ms(60000));
    }

    #[test]
    fn steps_never_go_the_wrong_way() {
        assert_eq!(slower(ms(120000)), ms(120000));
        assert_eq!(faster(ms(50)), ms(50));
    }

    #[test]
    fn an_interval_between_steps_goes_to_the_next() {
        assert_eq!(faster(ms(1500)), ms(1000));
        assert_eq!(slower(ms(1500)), ms(2000));
    }
}

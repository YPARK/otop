//! What sysinfo cannot read of a process, from `ps`: on macOS, sysinfo
//! cannot read other users' processes without root, but ps can. Elsewhere
//! sysinfo reads every process, and this does nothing.

use std::sync::Arc;

#[cfg(any(target_os = "macos", test))]
use std::collections::HashMap;

/// `ps`'s view of every process, read on a thread of its own, to fill in
/// what sysinfo leaves out.
pub struct Overlay {
    #[cfg(target_os = "macos")]
    ps: Ps,
    #[cfg(target_os = "macos")]
    rx: std::sync::mpsc::Receiver<Ps>,
}

#[cfg(target_os = "macos")]
impl Overlay {
    /// How often ps is run: it takes tens of milliseconds, on a thread of
    /// its own, and fills in only what sysinfo cannot read.
    const EVERY: std::time::Duration = std::time::Duration::from_secs(5);

    pub fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(Self::EVERY);
                if tx.send(ps()).is_err() {
                    break;
                }
            }
        });
        Overlay { ps: ps(), rx }
    }

    /// Takes the newest of ps's readings.
    pub fn refresh(&mut self) {
        if let Some(ps) = self.rx.try_iter().last() {
            self.ps = ps;
        }
    }

    /// Fills in what sysinfo left out of process `pid`, started at `start`,
    /// from what ps said of it, if ps saw that same process.
    pub fn fill(
        &self,
        pid: u32,
        start: u64,
        user: &mut Option<Arc<str>>,
        mem: &mut u64,
        command: &mut String,
    ) {
        let Some(e) = self.ps.get(&pid).filter(|e| e.start.abs_diff(start) <= 2) else {
            return;
        };
        if user.is_none() {
            *user = Some(e.user.clone());
        }
        if *mem == 0 {
            *mem = e.rss;
        }
        if command.is_empty() {
            command.clone_from(&e.command);
        }
    }
}

#[cfg(not(target_os = "macos"))]
impl Overlay {
    pub fn new() -> Self {
        Overlay {}
    }

    pub fn refresh(&mut self) {}

    pub fn fill(&self, _: u32, _: u64, _: &mut Option<Arc<str>>, _: &mut u64, _: &mut String) {}
}

/// What `ps` says of a process.
#[cfg(any(target_os = "macos", test))]
#[derive(Debug, PartialEq)]
struct PsEntry {
    user: Arc<str>,
    /// Resident memory, in bytes.
    rss: u64,
    command: String,
    /// When it started, in seconds since the epoch, to within a second: so
    /// an entry is not taken for a later process given the same PID.
    start: u64,
}

#[cfg(any(target_os = "macos", test))]
type Ps = HashMap<u32, PsEntry>;

#[cfg(target_os = "macos")]
fn ps() -> Ps {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    std::process::Command::new("ps")
        .args(["-axo", "pid=,etime=,rss=,user=,args="])
        .output()
        .map(|o| parse(&String::from_utf8_lossy(&o.stdout), now))
        .unwrap_or_default()
}

/// Every process, as `ps -axo pid=,etime=,rss=,user=,args=` gives them, by
/// PID, at `now` seconds since the epoch.
#[cfg(any(target_os = "macos", test))]
fn parse(out: &str, now: u64) -> Ps {
    out.lines()
        .filter_map(|l| {
            let mut f = l.split_whitespace();
            let pid = f.next()?.parse().ok()?;
            let start = now.checked_sub(elapsed(f.next()?)?)?;
            let rss: u64 = f.next()?.parse().ok()?;
            let user = f.next()?.into();
            let command = f.collect::<Vec<_>>().join(" ");
            let rss = rss * 1024;
            Some((
                pid,
                PsEntry {
                    user,
                    rss,
                    command,
                    start,
                },
            ))
        })
        .collect()
}

/// Seconds in ps's `[[dd-]hh:]mm:ss`.
#[cfg(any(target_os = "macos", test))]
fn elapsed(etime: &str) -> Option<u64> {
    let (days, rest) = match etime.split_once('-') {
        Some((d, r)) => (d.parse::<u64>().ok()?, r),
        None => (0, etime),
    };
    let mut secs = 0;
    for part in rest.split(':') {
        secs = secs * 60 + part.parse::<u64>().ok()?;
    }
    Some(days * 86400 + secs)
}

#[cfg(test)]
mod tests {
    use super::{elapsed, parse};

    #[test]
    fn reads_ps() {
        let ps = parse(
            "  488 2-03:04:05   1696 root     /usr/sbin/distnoted agent\n 1164       00:07   1584 _rmd     /usr/libexec/rmd\nbad line\n",
            1_000_000,
        );
        let e = &ps[&488];
        assert_eq!(
            (&*e.user, e.rss, e.command.as_str()),
            ("root", 1696 * 1024, "/usr/sbin/distnoted agent")
        );
        assert_eq!(e.start, 1_000_000 - (2 * 86400 + 3 * 3600 + 4 * 60 + 5));
        assert_eq!(&*ps[&1164].user, "_rmd");
        assert_eq!(ps[&1164].start, 1_000_000 - 7);
        assert_eq!(ps.len(), 2);
    }

    #[test]
    fn reads_elapsed_times() {
        assert_eq!(elapsed("00:07"), Some(7));
        assert_eq!(elapsed("01:02:03"), Some(3723));
        assert_eq!(elapsed("3-00:00:01"), Some(3 * 86400 + 1));
        assert_eq!(elapsed("x"), None);
    }
}

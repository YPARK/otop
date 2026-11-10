//! CPUs, memory and processes, sampled through sysinfo.

use crate::hist::History;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};
use sysinfo::{
    CpuRefreshKind, MemoryRefreshKind, Pid, ProcessRefreshKind, ProcessesToUpdate, Signal, System,
    Uid, UpdateKind,
};

/// A process, as last sampled.
pub struct Proc {
    pub pid: u32,
    /// The program's name, short where the command is a long path.
    pub name: String,
    /// The effective user, as top and ps show.
    pub user: Arc<str>,
    /// Percent of one CPU, so above 100 for a process on several.
    pub cpu: f32,
    /// Resident memory, in bytes.
    pub mem: u64,
    pub command: String,
    /// When it started, in seconds since the epoch: with its PID, which
    /// can be reused, this says which process it is.
    pub start: u64,
    /// Its PID, user, name and command, in lower case, for searching.
    pub search: String,
}

/// User names by UID. A UID not in /etc/passwd, such as an LDAP user's, is
/// looked up in the system's user database: at startup at once, and later
/// on a thread of its own, so a slow directory does not hold up the
/// screen; until it answers, the UID stands in for the name.
struct Users {
    names: HashMap<Uid, Arc<str>>,
    asked: HashSet<Uid>,
    /// Whether to wait for a lookup, as at startup, before the screen is up.
    wait: bool,
    tx: Sender<(Uid, Arc<str>)>,
    rx: Receiver<(Uid, Arc<str>)>,
}

impl Users {
    fn new() -> Self {
        let (tx, rx) = channel();
        Users {
            names: sysinfo::Users::new_with_refreshed_list()
                .iter()
                .map(|u| (u.id().clone(), u.name().into()))
                .collect(),
            asked: HashSet::new(),
            wait: true,
            tx,
            rx,
        }
    }

    /// Takes in the names looked up since last time.
    fn receive(&mut self) {
        self.names.extend(self.rx.try_iter());
    }

    fn name(&mut self, uid: &Uid) -> Arc<str> {
        if let Some(n) = self.names.get(uid) {
            return n.clone();
        }
        if self.wait {
            let n = user_name(uid);
            self.names.insert(uid.clone(), n.clone());
            return n;
        }
        if self.asked.insert(uid.clone()) {
            let (uid, tx) = (uid.clone(), self.tx.clone());
            std::thread::spawn(move || {
                let n = user_name(&uid);
                let _ = tx.send((uid, n));
            });
        }
        uid.to_string().into()
    }
}

pub struct Sys {
    sys: System,
    users: Users,
    /// When every process's command and user were last read.
    reread: Instant,
    /// On macOS, what ps says of processes sysinfo cannot read, those of
    /// other users: by PID, their user, resident memory and command.
    #[cfg(target_os = "macos")]
    ps: HashMap<u32, (Arc<str>, u64, String)>,
    pub host: String,
    pub cpu_brand: String,
    /// Each CPU's use, in percent.
    pub cores: Vec<f32>,
    pub cpu: f32,
    pub mem_used: u64,
    pub mem_total: u64,
    pub swap_used: u64,
    pub swap_total: u64,
    pub hist_cpu: History,
    pub hist_mem: History,
    pub procs: Vec<Proc>,
}

/// How often every process's command and user are read again, as exec,
/// setuid and setproctitle change them; between, they are read for new
/// processes alone, which costs a third as much.
const REREAD: Duration = Duration::from_secs(5);

fn proc_refresh(names: UpdateKind) -> ProcessRefreshKind {
    ProcessRefreshKind::nothing()
        .with_cpu()
        .with_memory()
        .with_cmd(names)
        .with_user(names)
        // Processes alone: on Linux each thread would be read too.
        .without_tasks()
}

impl Sys {
    pub fn new() -> Self {
        let mut sys = System::new();
        sys.refresh_cpu_specifics(CpuRefreshKind::nothing().with_cpu_usage());
        sys.refresh_processes_specifics(
            ProcessesToUpdate::All,
            true,
            proc_refresh(UpdateKind::OnlyIfNotSet),
        );
        let cpu_brand = sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .unwrap_or_default();
        let mut s = Sys {
            sys,
            users: Users::new(),
            reread: Instant::now(),
            #[cfg(target_os = "macos")]
            ps: ps(),
            host: System::host_name().unwrap_or_default(),
            cpu_brand,
            cores: Vec::new(),
            cpu: 0.0,
            mem_used: 0,
            mem_total: 0,
            swap_used: 0,
            swap_total: 0,
            hist_cpu: History::default(),
            hist_mem: History::default(),
            procs: Vec::new(),
        };
        // CPU use is measured between two refreshes; the first is above.
        std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
        s.sample();
        s.users.wait = false;
        s
    }

    pub fn sample(&mut self) {
        let sys = &mut self.sys;
        sys.refresh_cpu_specifics(CpuRefreshKind::nothing().with_cpu_usage());
        sys.refresh_memory_specifics(MemoryRefreshKind::everything());
        let names = if self.reread.elapsed() >= REREAD {
            self.reread = Instant::now();
            #[cfg(target_os = "macos")]
            {
                self.ps = ps();
            }
            UpdateKind::Always
        } else {
            UpdateKind::OnlyIfNotSet
        };
        sys.refresh_processes_specifics(ProcessesToUpdate::All, true, proc_refresh(names));

        self.cores = sys.cpus().iter().map(|c| c.cpu_usage()).collect();
        self.cpu = sys.global_cpu_usage();
        self.mem_used = sys.used_memory();
        self.mem_total = sys.total_memory();
        self.swap_used = sys.used_swap();
        self.swap_total = sys.total_swap();
        self.hist_cpu.push(self.cpu as f64);
        self.hist_mem
            .push(100.0 * frac(self.mem_used, self.mem_total));

        let users = &mut self.users;
        users.receive();
        #[cfg(target_os = "macos")]
        let ps = &self.ps;
        self.procs = sys
            .processes()
            .values()
            .filter(|p| p.thread_kind().is_none())
            .map(|p| {
                let pid = p.pid().as_u32();
                let name = p.name().to_string_lossy().into_owned();
                let mut command = String::new();
                for a in p.cmd() {
                    if !command.is_empty() {
                        command.push(' ');
                    }
                    command.push_str(&a.to_string_lossy());
                }
                let user = p.effective_user_id().or(p.user_id()).map(|u| users.name(u));
                // Another user's process, which sysinfo cannot read on
                // macOS: what ps said of it.
                #[cfg(target_os = "macos")]
                let (user, mem, command) = fill(ps.get(&pid), user, p.memory(), command);
                #[cfg(not(target_os = "macos"))]
                let mem = p.memory();
                let command = if command.is_empty() {
                    format!("[{name}]")
                } else {
                    command
                };
                let user = user.unwrap_or_else(|| "?".into());
                let search = format!("{pid} {user} {name} {command}").to_lowercase();
                Proc {
                    pid,
                    name,
                    user,
                    cpu: p.cpu_usage(),
                    mem,
                    command,
                    start: p.start_time(),
                    search,
                }
            })
            .collect();
    }

    /// Sends `signal` to process `pid`, if it is still the one started at
    /// `start`; an error says why it was not sent.
    pub fn signal(&mut self, pid: u32, start: u64, signal: Signal) -> Result<(), String> {
        // Look again, now: the process may have ended since the last sample,
        // and its PID been given to another.
        let id = Pid::from_u32(pid);
        self.sys.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[id]),
            true,
            ProcessRefreshKind::nothing(),
        );
        let p = self
            .sys
            .process(id)
            .filter(|p| p.start_time() == start)
            .ok_or_else(|| format!("process {pid} has ended"))?;
        match p.kill_with(signal) {
            Some(true) => Ok(()),
            Some(false) => Err(format!("cannot signal {pid}: not permitted, or gone")),
            None => Err(format!("{signal:?} is not supported here")),
        }
    }
}

/// A UID's name, from the system's user database, which, unlike the
/// /etc/passwd sysinfo reads, has directory users, such as LDAP's, too.
/// The UID itself where there is no name.
fn user_name(uid: &Uid) -> Arc<str> {
    let uid = uid.to_string();
    std::process::Command::new("id")
        .args(["-nu", &uid])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|n| !n.is_empty())
        .unwrap_or(uid)
        .into()
}

/// Every process's user, resident memory and command, as `ps` gives them,
/// by PID: on macOS ps can read other users' processes, which sysinfo
/// cannot without root.
#[cfg(any(target_os = "macos", test))]
fn parse_ps(out: &str) -> HashMap<u32, (Arc<str>, u64, String)> {
    out.lines()
        .filter_map(|l| {
            let mut f = l.split_whitespace();
            let pid = f.next()?.parse().ok()?;
            let rss: u64 = f.next()?.parse().ok()?;
            let user = f.next()?.into();
            let args = f.collect::<Vec<_>>().join(" ");
            Some((pid, (user, rss * 1024, args)))
        })
        .collect()
}

/// What sysinfo could not read of a process, from what ps said of it.
#[cfg(target_os = "macos")]
fn fill(
    ps: Option<&(Arc<str>, u64, String)>,
    user: Option<Arc<str>>,
    mem: u64,
    command: String,
) -> (Option<Arc<str>>, u64, String) {
    let Some((u, rss, args)) = ps else {
        return (user, mem, command);
    };
    (
        user.or_else(|| Some(u.clone())),
        if mem == 0 { *rss } else { mem },
        if command.is_empty() {
            args.clone()
        } else {
            command
        },
    )
}

#[cfg(target_os = "macos")]
fn ps() -> HashMap<u32, (Arc<str>, u64, String)> {
    std::process::Command::new("ps")
        .args(["-axo", "pid=,rss=,user=,args="])
        .output()
        .map(|o| parse_ps(&String::from_utf8_lossy(&o.stdout)))
        .unwrap_or_default()
}

pub fn frac(used: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        used as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::parse_ps;

    #[test]
    fn reads_ps() {
        let ps = parse_ps(
            "  488   1696 root     /usr/sbin/distnoted agent\n 1164   1584 _rmd     /usr/libexec/rmd\nbad line\n",
        );
        let (user, rss, args) = &ps[&488];
        assert_eq!(
            (&**user, *rss, args.as_str()),
            ("root", 1696 * 1024, "/usr/sbin/distnoted agent")
        );
        assert_eq!(&*ps[&1164].0, "_rmd");
        assert_eq!(ps.len(), 2);
    }
}

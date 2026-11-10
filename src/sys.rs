//! CPUs, memory and processes, sampled through sysinfo.

use crate::hist::History;
use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{Receiver, Sender, channel};
use sysinfo::{
    CpuRefreshKind, MemoryRefreshKind, Pid, ProcessRefreshKind, ProcessesToUpdate, Signal, System,
    UpdateKind,
};

/// A process, as last sampled.
#[derive(Clone)]
pub struct Proc {
    pub pid: u32,
    pub user: String,
    /// Percent of one CPU, so above 100 for a process on several.
    pub cpu: f32,
    /// Resident memory, in bytes.
    pub mem: u64,
    pub command: String,
    /// When it started, in seconds since the epoch: with its PID, which
    /// can be reused, this says which process it is.
    pub start: u64,
}

/// User names by UID. A UID not in /etc/passwd, such as an LDAP user's, is
/// looked up in the system's user database on a thread of its own, so a
/// slow directory does not hold up the screen; until it answers, the UID
/// stands in for the name.
struct Users {
    names: HashMap<String, String>,
    asked: HashSet<String>,
    tx: Sender<(String, String)>,
    rx: Receiver<(String, String)>,
}

impl Users {
    fn new() -> Self {
        let (tx, rx) = channel();
        Users {
            names: sysinfo::Users::new_with_refreshed_list()
                .iter()
                .map(|u| (u.id().to_string(), u.name().to_string()))
                .collect(),
            asked: HashSet::new(),
            tx,
            rx,
        }
    }

    /// `uid`'s name, waiting for it if `wait`, as at startup.
    fn name(&mut self, uid: &str, wait: bool) -> String {
        self.names.extend(self.rx.try_iter());
        if let Some(n) = self.names.get(uid) {
            return n.clone();
        }
        if wait {
            let n = user_name(uid);
            self.names.insert(uid.to_string(), n.clone());
            return n;
        }
        if self.asked.insert(uid.to_string()) {
            let (uid, tx) = (uid.to_string(), self.tx.clone());
            std::thread::spawn(move || {
                let n = user_name(&uid);
                let _ = tx.send((uid, n));
            });
        }
        uid.to_string()
    }
}

pub struct Sys {
    sys: System,
    users: Users,
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

fn proc_refresh() -> ProcessRefreshKind {
    ProcessRefreshKind::nothing()
        .with_cpu()
        .with_memory()
        // Read every time: exec, setuid and setproctitle change them.
        .with_cmd(UpdateKind::Always)
        .with_user(UpdateKind::Always)
}

impl Sys {
    pub fn new() -> Self {
        let mut sys = System::new();
        sys.refresh_cpu_specifics(CpuRefreshKind::nothing().with_cpu_usage());
        sys.refresh_processes_specifics(ProcessesToUpdate::All, true, proc_refresh());
        let cpu_brand = sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .unwrap_or_default();
        let mut s = Sys {
            sys,
            users: Users::new(),
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
        // At startup, before the screen is up, wait for users' names.
        s.read(true);
        s
    }

    pub fn sample(&mut self) {
        self.read(false);
    }

    fn read(&mut self, wait: bool) {
        let sys = &mut self.sys;
        sys.refresh_cpu_specifics(CpuRefreshKind::nothing().with_cpu_usage());
        sys.refresh_memory_specifics(MemoryRefreshKind::everything());
        sys.refresh_processes_specifics(ProcessesToUpdate::All, true, proc_refresh());

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
        self.procs = sys
            .processes()
            .values()
            // Linux lists each thread as well; show processes alone.
            .filter(|p| p.thread_kind().is_none())
            .map(|p| {
                let cmd: Vec<_> = p.cmd().iter().map(|a| a.to_string_lossy()).collect();
                let command = if cmd.is_empty() {
                    format!("[{}]", p.name().to_string_lossy())
                } else {
                    cmd.join(" ")
                };
                let user = p
                    .user_id()
                    .map_or_else(|| "?".into(), |u| users.name(&u.to_string(), wait));
                Proc {
                    pid: p.pid().as_u32(),
                    user,
                    cpu: p.cpu_usage(),
                    mem: p.memory(),
                    command,
                    start: p.start_time(),
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
fn user_name(uid: &str) -> String {
    std::process::Command::new("id")
        .args(["-nu", uid])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| uid.to_string())
}

pub fn frac(used: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        used as f64 / total as f64
    }
}

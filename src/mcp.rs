//! `otop mcp`: the CPUs, memory, GPUs and processes, served to agents over
//! the Model Context Protocol, as JSON-RPC on stdin and stdout.

use crate::app::{Only, SortBy, select};
use crate::gpu::{Gpu, GpuUse, Gpus};
use crate::sys::{Proc, Sys, frac};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::ops::RangeInclusive;
use std::time::{Duration, Instant};

/// The protocol versions spoken, newest first.
const VERSIONS: [&str; 4] = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];
const LATEST: &str = VERSIONS[0];

const INSTRUCTIONS: &str = "This machine's CPUs, memory, GPUs and processes, \
    read afresh at each call. status gives their use now, processes the \
    busiest processes and what they use of the GPUs, and watch the least, \
    mean and most use over a few seconds, to tell a busy GPU or CPU from an \
    idle one before choosing where to run a job.";

const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;

/// How many processes `processes` lists unless asked for more.
const LIMIT: u64 = 20;

/// The machine as last sampled, borrowed from its source.
pub struct View<'a> {
    pub host: &'a str,
    pub cpu_brand: &'a str,
    /// Percent, over every CPU, and of each.
    pub cpu: f32,
    pub cores: &'a [f32],
    /// Used and total, in bytes.
    pub mem: (u64, u64),
    pub swap: (u64, u64),
    pub procs: &'a [Proc],
    /// The GPUs' driver, or why there are none.
    pub gpu_info: &'a str,
    pub gpus: &'a [Gpu],
    pub gpu_use: &'a HashMap<u32, GpuUse>,
    /// Whether the GPUs say which processes use them: Apple's do not.
    pub per_process: bool,
}

/// What the server reads the machine from: sysinfo and the GPUs' source,
/// or a stand-in in tests.
pub trait Source {
    /// Samples the machine again.
    fn refresh(&mut self);
    fn view(&self) -> View<'_>;
    /// Waits, between `watch`'s samples.
    fn sleep(&mut self, d: Duration) {
        std::thread::sleep(d);
    }
}

/// The machine itself.
struct Live {
    sys: Sys,
    gpus: Gpus,
    /// When it was last sampled, and when the GPUs were last looked for.
    last: Instant,
    gpus_tried: Instant,
}

/// CPU use is measured since the last sample: one older than this would
/// give the mean over all that time, so another is taken first.
const STALE: Duration = Duration::from_secs(10);

/// How long to wait for the GPUs' reader, whose reads can take a while
/// (NVML's PCIe throughput takes 20 ms a GPU).
const GPU_WAIT: Duration = Duration::from_secs(2);

impl Live {
    fn new() -> Self {
        Live {
            sys: Sys::new(),
            gpus: Gpus::new(),
            last: Instant::now(),
            gpus_tried: Instant::now(),
        }
    }
}

impl Source for Live {
    fn refresh(&mut self) {
        // No GPUs may be a driver not loaded yet; look again now and then.
        if self.gpus.gpus.is_empty() && self.gpus_tried.elapsed() > STALE {
            self.gpus = Gpus::new();
            self.gpus_tried = Instant::now();
        }
        if self.last.elapsed() > STALE {
            self.sys.sample();
            std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
        }
        self.sys.sample();
        self.gpus.sample_within(GPU_WAIT);
        self.last = Instant::now();
    }

    fn view(&self) -> View<'_> {
        let s = &self.sys;
        View {
            host: &s.host,
            cpu_brand: &s.cpu_brand,
            cpu: s.cpu,
            cores: &s.cores,
            mem: (s.mem_used, s.mem_total),
            swap: (s.swap_used, s.swap_total),
            procs: &s.procs,
            gpu_info: &self.gpus.info,
            gpus: &self.gpus.gpus,
            gpu_use: &self.gpus.procs,
            per_process: self.gpus.per_process(),
        }
    }
}

/// Serves the machine on stdin and stdout until the client hangs up.
pub fn run() -> anyhow::Result<()> {
    Ok(serve(
        &mut Server::new(Live::new()),
        std::io::stdin().lock(),
        std::io::stdout().lock(),
    )?)
}

/// Answers each line read with one line written, until the input ends.
pub fn serve<S: Source>(
    server: &mut Server<S>,
    input: impl BufRead,
    mut output: impl Write,
) -> std::io::Result<()> {
    for line in input.split(b'\n') {
        let reply = match String::from_utf8(line?) {
            Ok(l) if l.trim().is_empty() => continue,
            Ok(l) => server.handle(&l),
            Err(_) => Some(error(Value::Null, PARSE_ERROR, "not UTF-8")),
        };
        if let Some(reply) = reply {
            writeln!(output, "{reply}")?;
            output.flush()?;
        }
    }
    Ok(())
}

pub struct Server<S> {
    source: S,
}

impl<S: Source> Server<S> {
    pub fn new(source: S) -> Self {
        Server { source }
    }

    /// The reply to one line from the client; none to a notification.
    pub fn handle(&mut self, line: &str) -> Option<Value> {
        let msg: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => return Some(error(Value::Null, PARSE_ERROR, &e.to_string())),
        };
        let Some(method) = msg.get("method").and_then(Value::as_str) else {
            // No requests are sent, so a reply from the client is ignored.
            if msg.get("result").is_some() || msg.get("error").is_some() {
                return None;
            }
            let id = msg.get("id").cloned().unwrap_or(Value::Null);
            return Some(error(id, INVALID_REQUEST, "not a JSON-RPC request"));
        };
        let id = match msg.get("id")? {
            Value::Null => return Some(error(Value::Null, INVALID_REQUEST, "id is null")),
            id => id.clone(),
        };
        let params = msg.get("params").unwrap_or(&Value::Null);
        let result = match method {
            "initialize" => Ok(initialize(params)),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": tools() })),
            "tools/call" => self.call(params),
            _ => Err((METHOD_NOT_FOUND, format!("no method {method:?}"))),
        };
        Some(match result {
            Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
            Err((code, message)) => error(id, code, &message),
        })
    }

    fn call(&mut self, params: &Value) -> Result<Value, (i64, String)> {
        let name = params.get("name").and_then(Value::as_str).unwrap_or("");
        let tool = match name {
            "status" => status,
            "processes" => processes,
            "watch" => watch,
            _ => return Err((INVALID_PARAMS, format!("no tool {name:?}"))),
        };
        let args = params.get("arguments").unwrap_or(&Value::Null);
        let out = if args.is_object() || args.is_null() {
            tool(&mut self.source, args)
        } else {
            Err("arguments must be an object".into())
        };
        Ok(match out {
            Ok(v) => json!({
                "content": [{ "type": "text", "text": v.to_string() }],
                "structuredContent": v,
                "isError": false,
            }),
            Err(e) => json!({
                "content": [{ "type": "text", "text": e }],
                "isError": true,
            }),
        })
    }
}

fn error(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// The client's protocol version if it is one spoken here, else the newest.
fn initialize(params: &Value) -> Value {
    let asked = params.get("protocolVersion").and_then(Value::as_str);
    let version = VERSIONS
        .into_iter()
        .find(|v| Some(*v) == asked)
        .unwrap_or(LATEST);
    json!({
        "protocolVersion": version,
        "capabilities": { "tools": {} },
        "serverInfo": {
            "name": "otop",
            "title": "otop: CPU, memory, GPU and process monitor",
            "version": env!("CARGO_PKG_VERSION"),
        },
        "instructions": INSTRUCTIONS,
    })
}

fn sort_names() -> Vec<&'static str> {
    SortBy::ALL.map(SortBy::name).to_vec()
}

fn tools() -> Value {
    let read_only = json!({ "readOnlyHint": true, "openWorldHint": false });
    json!([
        {
            "name": "status",
            "title": "Machine status",
            "description": "This machine now: its CPUs' use, overall and of \
                each; memory and swap used; and each GPU's utilisation, \
                memory, temperature, power, fan, clocks, PCIe throughput, \
                encoder and decoder use, performance state, and how many \
                processes it runs. What is not reported is null.",
            "inputSchema": { "type": "object", "properties": {} },
            "annotations": read_only,
        },
        {
            "name": "processes",
            "title": "Processes",
            "description": "The machine's processes, busiest CPU first by \
                default: PID, name, user, CPU percent (of one CPU, so above \
                100 on several), resident memory, command line, start time, \
                and, for a process on a GPU, which GPUs, its GPU memory and \
                SM use. NVIDIA's GPUs say which processes use them; Apple's \
                do not.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "search": {
                        "type": "string",
                        "description": "Only processes whose PID, user, name or \
                            command contain this, in any case.",
                    },
                    "gpu_only": {
                        "type": "boolean", "default": false,
                        "description": "Only processes on a GPU.",
                    },
                    "gpu": {
                        "type": "integer", "minimum": 0,
                        "description": "Only processes on this GPU, by index.",
                    },
                    "sort": {
                        "type": "string", "enum": sort_names(), "default": "cpu",
                        "description": "What to sort by; ties go by PID.",
                    },
                    "descending": {
                        "type": "boolean", "default": true,
                        "description": "Largest first.",
                    },
                    "limit": {
                        "type": "integer", "minimum": 1, "default": LIMIT,
                        "description": "At most this many processes.",
                    },
                },
            },
            "annotations": read_only,
        },
        {
            "name": "watch",
            "title": "Watch the machine",
            "description": "Samples the machine for a few seconds and gives \
                the least, mean and most CPU use and memory used, and each \
                GPU's utilisation and memory used, in percent. Use it to tell \
                an idle GPU or CPU from a busy one, since one reading can \
                catch it between bursts of work.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer", "minimum": 1, "maximum": 30, "default": 5,
                        "description": "How long to watch.",
                    },
                    "interval_ms": {
                        "type": "integer", "minimum": 250, "maximum": 5000, "default": 500,
                        "description": "Milliseconds between samples.",
                    },
                },
            },
            "annotations": read_only,
        },
    ])
}

/// An argument, or `None` if it is not given or null.
fn arg<'a>(args: &'a Value, name: &str) -> Option<&'a Value> {
    args.get(name).filter(|v| !v.is_null())
}

/// An argument read by `read`, or `None` if it is not given; an error,
/// saying it must be `what`, if `read` cannot read it.
fn arg_as<'a, T>(
    args: &'a Value,
    name: &str,
    what: &str,
    read: impl FnOnce(&'a Value) -> Option<T>,
) -> Result<Option<T>, String> {
    arg(args, name)
        .map(|v| read(v).ok_or_else(|| format!("{name} must be {what}")))
        .transpose()
}

/// A whole-number argument in `range`, or `None` if it is not given.
fn arg_u64(args: &Value, name: &str, range: RangeInclusive<u64>) -> Result<Option<u64>, String> {
    let what = format!("a whole number from {} to {}", range.start(), range.end());
    arg_as(args, name, &what, |v| {
        v.as_u64().filter(|n| range.contains(n))
    })
}

/// One decimal place is all an agent needs of a percentage.
fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn pct(f: f32) -> f64 {
    round1(f as f64)
}

/// Used and total bytes, and the percent used.
fn usage((used, total): (u64, u64)) -> Value {
    json!({
        "used_bytes": used,
        "total_bytes": total,
        "used_pct": round1(100.0 * frac(used, total)),
    })
}

fn status<S: Source>(s: &mut S, _: &Value) -> Result<Value, String> {
    s.refresh();
    let v = s.view();
    let gpus: Vec<Value> = v
        .gpus
        .iter()
        .map(|g| {
            let on = v.gpu_use.values().filter(|u| u.gpus.contains(&g.index));
            json!({
                "index": g.index,
                "name": g.name,
                "util_pct": g.util,
                "mem_used_bytes": g.mem_used,
                "mem_total_bytes": g.mem_total,
                "mem_used_pct": round1(100.0 * g.mem_frac()),
                "temp_c": g.temp,
                "power_w": g.power,
                "power_limit_w": g.power_limit,
                "fan_pct": g.fan,
                "clock_graphics_mhz": g.clock_gr,
                "clock_graphics_max_mhz": g.clock_gr_max,
                "clock_mem_mhz": g.clock_mem,
                "pcie_tx_bytes_per_s": g.pcie_tx,
                "pcie_rx_bytes_per_s": g.pcie_rx,
                "encoder_pct": g.enc,
                "decoder_pct": g.dec,
                "pstate": g.pstate,
                "cores": g.cores,
                "processes": v.per_process.then(|| on.count()),
            })
        })
        .collect();
    Ok(json!({
        "host": v.host,
        "cpu": {
            "brand": v.cpu_brand,
            "util_pct": pct(v.cpu),
            "cores_pct": v.cores.iter().map(|&c| pct(c)).collect::<Vec<_>>(),
        },
        "memory": usage(v.mem),
        "swap": usage(v.swap),
        "gpu_info": v.gpu_info,
        "gpus": gpus,
    }))
}

fn processes<S: Source>(s: &mut S, args: &Value) -> Result<Value, String> {
    let search = arg_as(args, "search", "a string", Value::as_str)?.unwrap_or("");
    let gpu_only = arg_as(args, "gpu_only", "true or false", Value::as_bool)?.unwrap_or(false);
    let gpu = arg_u64(args, "gpu", 0..=u32::MAX as u64)?.map(|g| g as u32);
    let sort = arg_as(
        args,
        "sort",
        &format!("one of {}", sort_names().join(", ")),
        |v| {
            let n = v.as_str()?;
            SortBy::ALL.into_iter().find(|b| b.name() == n)
        },
    )?
    .unwrap_or(SortBy::Cpu);
    let descending = arg_as(args, "descending", "true or false", Value::as_bool)?.unwrap_or(true);
    let limit = arg_u64(args, "limit", 1..=u64::MAX)?.unwrap_or(LIMIT);

    s.refresh();
    let v = s.view();
    let only = match gpu {
        Some(g) => Only::Gpu(g),
        None if gpu_only => Only::AnyGpu,
        None => Only::All,
    };
    if only != Only::All || sort.needs_gpu() {
        if v.gpus.is_empty() {
            return Err(format!("no GPUs: {}", v.gpu_info));
        }
        if !v.per_process {
            return Err("these GPUs do not say which processes use them".into());
        }
    }
    if let Some(g) = gpu.filter(|&g| g as usize >= v.gpus.len()) {
        return Err(format!("no GPU {g}; there are {}", v.gpus.len()));
    }
    let rows = select(v.procs, v.gpu_use, only, search, sort, descending);
    let shown: Vec<Value> = rows
        .iter()
        .take(limit.try_into().unwrap_or(usize::MAX))
        .map(|&i| {
            let p = &v.procs[i];
            let gpu = v
                .gpu_use
                .get(&p.pid)
                .map(|u| json!({ "gpus": u.gpus, "mem_bytes": u.mem, "sm_pct": u.sm }));
            json!({
                "pid": p.pid,
                "name": p.name,
                "user": &*p.user,
                "cpu_pct": pct(p.cpu),
                "mem_bytes": p.mem,
                "command": p.command,
                "start_unix": p.start,
                "gpu": gpu,
            })
        })
        .collect();
    Ok(json!({ "total": rows.len(), "processes": shown }))
}

/// Least, mean and most of `xs`; null if there are none.
fn summary(xs: &[f64]) -> Value {
    if xs.is_empty() {
        return Value::Null;
    }
    let min = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    json!({ "min": round1(min), "mean": round1(mean), "max": round1(max) })
}

fn watch<S: Source>(s: &mut S, args: &Value) -> Result<Value, String> {
    let seconds = arg_u64(args, "seconds", 1..=30)?.unwrap_or(5);
    let window = 1000 * seconds;
    let interval = arg_u64(args, "interval_ms", 250..=5000)?
        .unwrap_or(500)
        .min(window);
    let samples = window / interval + 1;

    let (mut cpu, mut mem) = (Vec::new(), Vec::new());
    // Per GPU, by position: utilisation, and memory used, in percent.
    let mut seen: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    for i in 0..samples {
        // Sampling takes time too: the wait is what is left of the tick.
        let tick = Instant::now();
        s.refresh();
        let v = s.view();
        cpu.push(v.cpu as f64);
        mem.push(100.0 * frac(v.mem.0, v.mem.1));
        seen.resize_with(v.gpus.len(), Default::default);
        for (g, (util, gmem)) in v.gpus.iter().zip(&mut seen) {
            util.extend(g.util.map(f64::from));
            gmem.push(100.0 * g.mem_frac());
        }
        if i + 1 < samples {
            s.sleep(Duration::from_millis(interval).saturating_sub(tick.elapsed()));
        }
    }
    let v = s.view();
    let gpus: Vec<Value> = v
        .gpus
        .iter()
        .zip(&seen)
        .map(|(g, (util, gmem))| {
            json!({
                "index": g.index,
                "name": g.name,
                "util_pct": summary(util),
                "mem_used_pct": summary(gmem),
            })
        })
        .collect();
    Ok(json!({
        "seconds": seconds,
        "interval_ms": interval,
        "samples": samples,
        "cpu_pct": summary(&cpu),
        "mem_used_pct": summary(&mem),
        "gpus": gpus,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sys::search_key;
    use std::sync::Arc;

    /// One sample of a made-up machine.
    #[derive(Default)]
    struct Snap {
        cpu: f32,
        mem_used: u64,
        procs: Vec<Proc>,
        gpus: Vec<Gpu>,
        gpu_use: HashMap<u32, GpuUse>,
    }

    /// A machine that changes, one snapshot to each refresh, staying on
    /// the last.
    struct Fake {
        snaps: Vec<Snap>,
        at: usize,
        per_process: bool,
        /// How long each refresh takes, and how long `watch` has waited.
        busy: Duration,
        slept: Duration,
    }

    impl Source for Fake {
        fn refresh(&mut self) {
            std::thread::sleep(self.busy);
            self.at = (self.at + 1).min(self.snaps.len() - 1);
        }
        fn view(&self) -> View<'_> {
            let s = &self.snaps[self.at];
            View {
                host: "host1",
                cpu_brand: "CPU-A",
                cpu: s.cpu,
                cores: &[12.34, 56.0],
                mem: (s.mem_used, 1000),
                swap: (0, 0),
                procs: &s.procs,
                gpu_info: "driver 1",
                gpus: &s.gpus,
                gpu_use: &s.gpu_use,
                per_process: self.per_process,
            }
        }
        fn sleep(&mut self, d: Duration) {
            self.slept += d;
        }
    }

    fn gpu(index: u32, util: u32, mem_used: u64) -> Gpu {
        Gpu {
            index,
            name: format!("GPU-{index}"),
            util: Some(util),
            mem_used,
            mem_total: 1000,
            power: Some(150),
            pcie_tx: Some(2048),
            ..Default::default()
        }
    }

    fn proc_(pid: u32, cpu: f32, mem: u64) -> Proc {
        let (user, name, command) = (
            format!("user{pid}"),
            format!("prog{pid}"),
            format!("cmd{pid} --flag"),
        );
        Proc {
            pid,
            search: search_key(pid, &user, &name, &command),
            name,
            user: Arc::from(user),
            cpu,
            mem,
            command,
            start: 1_000 + pid as u64,
        }
    }

    fn on(gpus: &[u32], mem: u64, sm: u32) -> GpuUse {
        GpuUse {
            gpus: gpus.to_vec(),
            mem: Some(mem),
            sm: Some(sm),
        }
    }

    fn server(snaps: Vec<Snap>, per_process: bool) -> Server<Fake> {
        Server::new(Fake {
            snaps,
            at: 0,
            per_process,
            busy: Duration::ZERO,
            slept: Duration::ZERO,
        })
    }

    /// A GPU that does not say which processes use it, as Apple's.
    fn silent_gpu() -> Server<Fake> {
        let snap = Snap {
            gpus: vec![gpu(0, 10, 0)],
            ..Default::default()
        };
        server(vec![snap], false)
    }

    /// Two GPUs and four processes, three of them on the GPUs; the first
    /// snapshot is stale, so a tool that does not refresh is caught.
    fn machine() -> Server<Fake> {
        let now = Snap {
            cpu: 37.25,
            mem_used: 250,
            procs: vec![
                proc_(10, 5.0, 300),
                proc_(11, 90.0, 100),
                proc_(12, 20.0, 500),
                proc_(13, 50.0, 50),
            ],
            gpus: vec![gpu(0, 40, 250), gpu(1, 80, 600)],
            gpu_use: HashMap::from([
                (10, on(&[0], 300, 5)),
                (11, on(&[1], 100, 90)),
                (12, on(&[0, 1], 500, 20)),
            ]),
        };
        let stale = Snap {
            gpus: vec![gpu(0, 0, 0), gpu(1, 0, 0)],
            ..Default::default()
        };
        server(vec![stale, now], true)
    }

    fn request(s: &mut Server<Fake>, method: &str, params: Value) -> Value {
        let line = json!({"jsonrpc": "2.0", "id": 7, "method": method, "params": params});
        let reply = s.handle(&line.to_string()).expect("a reply");
        assert_eq!(reply["jsonrpc"], "2.0");
        assert_eq!(reply["id"], 7);
        reply
    }

    fn call(s: &mut Server<Fake>, tool: &str, args: Value) -> Value {
        let reply = request(s, "tools/call", json!({"name": tool, "arguments": args}));
        let result = reply["result"].clone();
        assert!(result.is_object(), "{reply}");
        result
    }

    /// A tool's structured result, checked to match its text.
    fn ok(result: &Value) -> &Value {
        assert_eq!(result["isError"], false, "{result}");
        let text = result["content"][0]["text"].as_str().expect("text");
        let parsed: Value = serde_json::from_str(text).expect("text is JSON");
        assert_eq!(parsed, result["structuredContent"]);
        &result["structuredContent"]
    }

    fn pids(st: &Value) -> Vec<u64> {
        st["processes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["pid"].as_u64().unwrap())
            .collect()
    }

    fn listed(s: &mut Server<Fake>, args: Value) -> Vec<u64> {
        pids(ok(&call(s, "processes", args)))
    }

    /// The replies served to `input`, one to a line.
    fn serve_lines(s: &mut Server<Fake>, input: &[u8]) -> Vec<Value> {
        let mut out = Vec::new();
        serve(s, input, &mut out).unwrap();
        String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    fn error_text(result: &Value) -> String {
        assert_eq!(result["isError"], true, "{result}");
        result["content"][0]["text"].as_str().unwrap().to_string()
    }

    #[test]
    fn initialize_agrees_a_version_and_offers_tools() {
        let mut s = machine();
        let r = request(
            &mut s,
            "initialize",
            json!({"protocolVersion": "2025-06-18"}),
        );
        assert_eq!(r["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(r["result"]["serverInfo"]["name"], "otop");
        assert!(r["result"]["capabilities"]["tools"].is_object());
        assert!(r["result"]["instructions"].is_string());

        let r = request(
            &mut s,
            "initialize",
            json!({"protocolVersion": "1999-01-01"}),
        );
        assert_eq!(r["result"]["protocolVersion"], LATEST);
    }

    #[test]
    fn notifications_get_no_reply_and_ping_an_empty_one() {
        let mut s = machine();
        let n = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
        assert!(s.handle(&n.to_string()).is_none());
        assert_eq!(request(&mut s, "ping", json!({}))["result"], json!({}));
    }

    #[test]
    fn lists_its_tools_with_schemas() {
        let mut s = machine();
        let r = request(&mut s, "tools/list", json!({}));
        let tools = r["result"]["tools"].as_array().unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["status", "processes", "watch"]);
        for t in tools {
            assert!(t["description"].is_string());
            assert_eq!(t["inputSchema"]["type"], "object");
            assert_eq!(t["annotations"]["readOnlyHint"], true);
        }
        let sorts = &tools[1]["inputSchema"]["properties"]["sort"]["enum"];
        assert_eq!(sorts[0], "cpu");
    }

    #[test]
    fn status_reads_the_machine_afresh() {
        let mut s = machine();
        let r = call(&mut s, "status", json!({}));
        let st = ok(&r);
        assert_eq!(st["host"], "host1");
        assert_eq!(st["cpu"]["brand"], "CPU-A");
        assert_eq!(st["cpu"]["util_pct"], 37.3);
        assert_eq!(st["cpu"]["cores_pct"], json!([12.3, 56.0]));
        assert_eq!(st["memory"]["used_bytes"], 250);
        assert_eq!(st["memory"]["total_bytes"], 1000);
        assert_eq!(st["memory"]["used_pct"], 25.0);
        assert_eq!(st["swap"]["used_pct"], 0.0);
        assert_eq!(st["gpu_info"], "driver 1");
        let g = &st["gpus"][1];
        assert_eq!(g["index"], 1);
        assert_eq!(g["name"], "GPU-1");
        assert_eq!(g["util_pct"], 80);
        assert_eq!(g["mem_used_bytes"], 600);
        assert_eq!(g["mem_used_pct"], 60.0);
        assert_eq!(g["power_w"], 150);
        assert_eq!(g["pcie_tx_bytes_per_s"], 2048);
        assert_eq!(g["temp_c"], Value::Null);
        // Process 12 is on both.
        assert_eq!(g["processes"], 2);
        assert_eq!(st["gpus"][0]["processes"], 2);
    }

    #[test]
    fn gpus_that_do_not_say_who_uses_them_count_no_processes() {
        let mut s = silent_gpu();
        let r = call(&mut s, "status", json!({}));
        assert_eq!(ok(&r)["gpus"][0]["processes"], Value::Null);
    }

    #[test]
    fn processes_by_cpu_largest_first_by_default() {
        let mut s = machine();
        let r = call(&mut s, "processes", json!({}));
        let st = ok(&r);
        assert_eq!(pids(st), [11, 13, 12, 10]);
        assert_eq!(st["total"], 4);
        let p = &st["processes"][0];
        assert_eq!(p["name"], "prog11");
        assert_eq!(p["user"], "user11");
        assert_eq!(p["cpu_pct"], 90.0);
        assert_eq!(p["mem_bytes"], 100);
        assert_eq!(p["command"], "cmd11 --flag");
        assert_eq!(p["start_unix"], 1011);
        assert_eq!(
            p["gpu"],
            json!({"gpus": [1], "mem_bytes": 100, "sm_pct": 90})
        );
        // Not on a GPU.
        assert_eq!(st["processes"][1]["gpu"], Value::Null);
    }

    #[test]
    fn processes_filtered_searched_sorted_and_cut() {
        let mut s = machine();
        assert_eq!(listed(&mut s, json!({"gpu": 1})), [11, 12]);
        assert_eq!(listed(&mut s, json!({"gpu_only": true})), [11, 12, 10]);
        assert_eq!(listed(&mut s, json!({"search": "CMD12"})), [12]);
        assert_eq!(
            listed(&mut s, json!({"sort": "gpu_memory"})),
            [12, 10, 11, 13]
        );
        assert_eq!(
            listed(&mut s, json!({"sort": "pid", "descending": false})),
            [10, 11, 12, 13]
        );
        let r = call(&mut s, "processes", json!({"sort": "memory", "limit": 2}));
        assert_eq!(pids(ok(&r)), [12, 10]);
        assert_eq!(ok(&r)["total"], 4);
    }

    #[test]
    fn processes_are_cut_at_twenty_unless_asked() {
        let snap = Snap {
            procs: (1..=25).map(|p| proc_(p, p as f32, 0)).collect(),
            ..Default::default()
        };
        let mut s = server(vec![snap], true);
        let r = call(&mut s, "processes", json!({}));
        assert_eq!(ok(&r)["processes"].as_array().unwrap().len(), 20);
        assert_eq!(ok(&r)["total"], 25);
        assert_eq!(listed(&mut s, json!({"limit": 100})).len(), 25);
    }

    #[test]
    fn bad_arguments_are_tool_errors_the_agent_can_read() {
        let mut s = machine();
        let mut err = |tool: &str, args: Value| error_text(&call(&mut s, tool, args));
        let e = err("processes", json!({"sort": "colour"}));
        assert!(e.contains("gpu_memory"), "{e}");
        assert!(err("processes", json!({"gpu": "one"})).contains("gpu"));
        assert!(err("processes", json!({"gpu": 2})).contains("no GPU 2"));
        assert!(err("processes", json!({"search": 5})).contains("search"));
        assert!(err("processes", json!({"limit": 0})).contains("limit"));
        assert!(err("processes", json!({"gpu_only": "yes"})).contains("gpu_only"));
        assert!(err("watch", json!({"seconds": 0})).contains("seconds"));
        assert!(err("watch", json!({"seconds": 31})).contains("seconds"));
        assert!(err("watch", json!({"interval_ms": 100})).contains("interval_ms"));
        assert!(err("status", json!([1])).contains("object"));
    }

    #[test]
    fn asking_for_gpu_processes_where_gpus_do_not_say_is_an_error() {
        let mut s = silent_gpu();
        for args in [
            json!({"gpu_only": true}),
            json!({"gpu": 0}),
            json!({"sort": "gpu_memory"}),
            json!({"sort": "sm"}),
        ] {
            let e = error_text(&call(&mut s, "processes", args));
            assert!(e.contains("which processes"), "{e}");
        }
    }

    #[test]
    fn asking_for_gpu_processes_without_gpus_says_there_are_none() {
        let mut s = server(vec![Snap::default()], false);
        let e = error_text(&call(&mut s, "processes", json!({"gpu_only": true})));
        assert!(e.contains("no GPUs") && e.contains("driver 1"), "{e}");
    }

    #[test]
    fn watch_summarises_samples_over_the_window() {
        let snaps = [0, 10, 30, 50]
            .map(|u| Snap {
                cpu: u as f32,
                mem_used: 10 * u as u64,
                gpus: vec![gpu(0, u, 10 * u as u64)],
                ..Default::default()
            })
            .into();
        let mut s = server(snaps, true);
        s.source.busy = Duration::from_millis(20);
        let r = call(&mut s, "watch", json!({"seconds": 1, "interval_ms": 500}));
        let st = ok(&r);
        assert_eq!(st["samples"], 3);
        let spread = json!({"min": 10.0, "mean": 30.0, "max": 50.0});
        assert_eq!(st["cpu_pct"], spread);
        assert_eq!(st["mem_used_pct"], spread);
        let g = &st["gpus"][0];
        assert_eq!(g["index"], 0);
        assert_eq!(g["name"], "GPU-0");
        assert_eq!(g["util_pct"], spread);
        assert_eq!(g["mem_used_pct"], spread);
        // What each sample took is not slept again, so the window is kept.
        let slept = s.source.slept;
        assert!(slept <= Duration::from_millis(965), "{slept:?}");
        assert!(slept >= Duration::from_millis(900), "{slept:?}");
    }

    #[test]
    fn unknown_tools_and_methods_are_protocol_errors() {
        let mut s = machine();
        let r = request(&mut s, "tools/call", json!({"name": "nope"}));
        assert_eq!(r["error"]["code"], -32602);
        let r = request(&mut s, "resources/list", json!({}));
        assert_eq!(r["error"]["code"], -32601);
    }

    #[test]
    fn unreadable_lines_are_parse_errors() {
        let mut s = machine();
        let r = s.handle("{not json").expect("a reply");
        assert_eq!(r["error"]["code"], -32700);
        assert_eq!(r["id"], Value::Null);
        let r = s.handle("[1, 2]").expect("a reply");
        assert_eq!(r["error"]["code"], -32600);
    }

    #[test]
    fn requests_need_an_id_that_is_not_null() {
        let mut s = machine();
        let line = json!({"jsonrpc": "2.0", "id": null, "method": "tools/list"});
        let r = s.handle(&line.to_string()).expect("a reply");
        assert_eq!(r["error"]["code"], -32600);
    }

    #[test]
    fn bytes_that_are_not_text_get_a_parse_error_and_serving_goes_on() {
        let mut s = machine();
        let ping = json!({"jsonrpc": "2.0", "id": 2, "method": "ping"}).to_string();
        let mut input = b"\xff\xfe\n".to_vec();
        input.extend(ping.as_bytes());
        let lines = serve_lines(&mut s, &input);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["error"]["code"], -32700);
        assert_eq!(lines[1]["id"], 2);
    }

    #[test]
    fn serves_one_line_for_each_request() {
        let mut s = machine();
        let input = [
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
            json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json!({"jsonrpc": "2.0", "id": "b", "method": "tools/call",
                   "params": {"name": "status"}}),
        ]
        .map(|v| v.to_string())
        .join("\n");
        let lines = serve_lines(&mut s, format!("{input}\n\n").as_bytes());
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["id"], 1);
        assert_eq!(lines[1]["id"], "b");
        assert_eq!(
            lines[1]["result"]["structuredContent"]["gpus"][0]["util_pct"],
            40
        );
    }
}

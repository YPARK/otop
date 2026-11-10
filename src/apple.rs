//! Apple's GPUs, on macOS, read from the IORegistry.
//!
//! Metal reports no GPU use, but each GPU's IOAccelerator publishes
//! `PerformanceStatistics`, which `ioreg` prints without root: the GPU's
//! use in percent and the memory it has in use. Apple's GPUs share the
//! system's memory, so that is what their memory is out of.

#[cfg(target_os = "macos")]
pub use source::Source;

/// The IORegistry does not say which processes use the GPU.
#[cfg(target_os = "macos")]
pub const PER_PROCESS: bool = false;

#[cfg(target_os = "macos")]
mod source {
    use super::parse;
    use crate::gpu::{Gpu, GpuUse};
    use std::collections::HashMap;
    use std::process::Command;

    pub struct Source {
        mem_total: u64,
    }

    fn ioreg() -> Result<String, String> {
        let out = Command::new("ioreg")
            .args(["-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])
            .output()
            .map_err(|e| format!("cannot run ioreg: {e}"))?;
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    impl Source {
        pub fn open() -> Result<(Self, String, Vec<Gpu>), String> {
            let readings = parse(&ioreg()?);
            if readings.is_empty() {
                return Err("no GPU in the IORegistry".into());
            }
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            let source = Source {
                mem_total: sys.total_memory(),
            };
            let gpus = readings
                .into_iter()
                .enumerate()
                .map(|(i, r)| Gpu {
                    index: i as u32,
                    name: r.name,
                    cores: r.cores,
                    ..Default::default()
                })
                .collect();
            Ok((source, "Apple GPU".into(), gpus))
        }

        pub fn sample(&mut self, gpus: &mut [Gpu], _: &mut HashMap<u32, GpuUse>) {
            let Ok(out) = ioreg() else { return };
            for (g, r) in gpus.iter_mut().zip(parse(&out)) {
                g.util = r.util;
                g.mem_used = r.mem_used.unwrap_or(0);
                g.mem_total = self.mem_total;
            }
        }
    }
}

/// One GPU's entry in `ioreg`.
#[derive(Debug, PartialEq, Default)]
pub struct Reading {
    pub name: String,
    pub util: Option<u32>,
    pub mem_used: Option<u64>,
    pub cores: Option<u32>,
}

/// The GPUs in `ioreg -r -d 1 -w 0 -c IOAccelerator`'s output.
pub fn parse(out: &str) -> Vec<Reading> {
    out.split("+-o ")
        .skip(1)
        .map(|entry| {
            let class = entry.split_whitespace().next().unwrap_or("GPU");
            Reading {
                name: string_after(entry, "\"model\"").unwrap_or_else(|| class.to_string()),
                util: number_after(entry, "\"Device Utilization %\"").map(|n| n as u32),
                mem_used: number_after(entry, "\"In use system memory\""),
                cores: number_after(entry, "\"gpu-core-count\"").map(|n| n as u32),
            }
        })
        .collect()
}

/// What follows `key`, past `=` and any spaces.
fn value_after<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    let rest = &s[s.find(key)? + key.len()..];
    rest.trim_start().strip_prefix('=').map(str::trim_start)
}

fn number_after(s: &str, key: &str) -> Option<u64> {
    let v = value_after(s, key)?;
    let end = v.find(|c: char| !c.is_ascii_digit()).unwrap_or(v.len());
    v[..end].parse().ok()
}

fn string_after(s: &str, key: &str) -> Option<String> {
    let v = value_after(s, key)?.strip_prefix('"')?;
    Some(v[..v.find('"')?].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const M1: &str = r#"+-o AGXAcceleratorG13X  <class AGXAcceleratorG13X, id 0x1000002f1, registered, matched, active, busy 0 (0 ms), retain 79>
    {
      "IOClass" = "AGXAcceleratorG13X"
      "gpu-core-count" = 16
      "PerformanceStatistics" = {"In use system memory (driver)"=0,"Alloc system memory"=2136571904,"Tiler Utilization %"=7,"recoveryCount"=0,"Renderer Utilization %"=9,"Device Utilization %"=12,"SplitSceneCount"=0,"In use system memory"=611713024}
      "model" = "Apple M1 Pro"
    }
"#;

    #[test]
    fn reads_an_apple_gpu() {
        assert_eq!(
            parse(M1),
            vec![Reading {
                name: "Apple M1 Pro".into(),
                util: Some(12),
                mem_used: Some(611713024),
                cores: Some(16),
            }]
        );
    }

    #[test]
    fn names_a_gpu_by_its_class_without_a_model() {
        let r = parse(
            "+-o AMDRadeonX6000  <class AMDRadeonX6000>\n  {\n    \"PerformanceStatistics\" = {\"Device Utilization %\"=3}\n  }\n",
        );
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].name, "AMDRadeonX6000");
        assert_eq!(r[0].util, Some(3));
        assert_eq!(r[0].mem_used, None);
    }

    #[test]
    fn finds_no_gpus_in_nothing() {
        assert!(parse("").is_empty());
    }
}

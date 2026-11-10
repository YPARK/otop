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
    use super::{Reading, collided, parse};
    use crate::gpu::{Gpu, GpuUse};
    use std::collections::HashMap;
    use std::process::Command;
    use std::time::Duration;

    /// How long to wait before reading again a reading that collided.
    const AGAIN: Duration = Duration::from_millis(50);

    pub struct Source {
        last: Vec<Reading>,
    }

    fn ioreg() -> Vec<Reading> {
        Command::new("ioreg")
            .args(["-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])
            .output()
            .map(|o| parse(&String::from_utf8_lossy(&o.stdout)))
            .unwrap_or_default()
    }

    /// The GPUs' readings, read again a moment later if they look like
    /// they collided with another program's, given the last ones.
    fn read_settled(prev: &[Reading]) -> Vec<Reading> {
        let r = ioreg();
        if !collided(prev, &r) {
            return r;
        }
        std::thread::sleep(AGAIN);
        ioreg()
    }

    impl Source {
        pub fn open() -> Result<(Self, String, Vec<Gpu>), String> {
            let last = ioreg();
            if last.is_empty() {
                return Err("no GPU in the IORegistry".into());
            }
            // Apple's GPUs share the system's memory.
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            let gpus = last
                .iter()
                .enumerate()
                .map(|(i, r)| Gpu {
                    index: i as u32,
                    name: r.name.clone(),
                    cores: r.cores,
                    mem_total: sys.total_memory(),
                    ..Default::default()
                })
                .collect();
            Ok((Source { last }, "Apple GPU".into(), gpus))
        }

        pub fn sample(&mut self, gpus: &mut [Gpu], _: &mut HashMap<u32, GpuUse>) {
            self.last = read_settled(&self.last);
            for (g, r) in gpus.iter_mut().zip(&self.last) {
                g.util = r.util;
                g.mem_used = r.mem_used.unwrap_or(0);
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
    /// `TiledSceneBytes`: on Apple's GPUs never 0 but in a collided reading.
    pub scene: Option<u64>,
}

/// Whether a reading looks like one that collided with another program's:
/// when two read the IORegistry at once, macOS can give one of them 0% for
/// a busy GPU, `TiledSceneBytes` 0 where it is otherwise never 0, or no
/// statistics at all. Such a reading is taken again, a moment later, when
/// the other has finished. A GPU that has gone idle reads 0% again, and so
/// is shown as idle, only 50 ms late.
pub fn collided(prev: &[Reading], next: &[Reading]) -> bool {
    next.len() != prev.len()
        || next.iter().zip(prev).any(|(n, p)| {
            // Statistics left out, where they were there before: a GPU
            // whose driver never gives them is not read twice each time.
            (n.mem_used.is_none() && p.mem_used.is_some())
                || (n.util == Some(0) && p.util.is_some_and(|u| u > 0))
                // Even when the last reading was idle, so the GPU's change
                // from idle to busy is not lost: where `TiledSceneBytes`
                // has been above 0, a 0 there is a collision.
                || (n.scene == Some(0) && p.scene.is_some_and(|b| b > 0))
        })
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
                scene: number_after(entry, "\"TiledSceneBytes\""),
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
                scene: None,
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

    fn reading(util: u32) -> Reading {
        Reading {
            util: Some(util),
            mem_used: Some(1 << 20),
            scene: Some(294912),
            ..Default::default()
        }
    }

    #[test]
    fn a_collision_after_an_idle_reading_is_read_again() {
        let mut r = reading(0);
        r.scene = Some(0);
        assert!(collided(&[reading(0)], &[r]));
    }

    #[test]
    fn a_gpu_whose_scene_bytes_stay_0_is_not_read_again() {
        let mut r = reading(0);
        r.scene = Some(0);
        assert!(!collided(
            std::slice::from_ref(&r),
            std::slice::from_ref(&r)
        ));
    }

    #[test]
    fn a_gpu_that_never_gives_scene_bytes_is_not_read_again() {
        let mut r = reading(0);
        r.scene = None;
        assert!(!collided(
            std::slice::from_ref(&r),
            std::slice::from_ref(&r)
        ));
    }

    #[test]
    fn a_busy_gpu_dropping_to_zero_is_read_again() {
        assert!(collided(&[reading(99)], &[reading(0)]));
    }

    #[test]
    fn missing_statistics_are_read_again() {
        let mut r = reading(50);
        r.mem_used = None;
        assert!(collided(&[reading(50)], std::slice::from_ref(&r)));
        assert!(collided(&[reading(50)], &[]));
        // Never given, so not read again.
        assert!(!collided(
            std::slice::from_ref(&r),
            std::slice::from_ref(&r)
        ));
    }

    #[test]
    fn steady_and_idle_readings_are_kept() {
        assert!(!collided(&[reading(99)], &[reading(98)]));
        assert!(!collided(&[reading(0)], &[reading(0)]));
        assert!(!collided(&[reading(0)], &[reading(40)]));
    }

    #[test]
    fn finds_no_gpus_in_nothing() {
        assert!(parse("").is_empty());
    }
}

# otop

CPUs, memory, GPUs and processes in the terminal, like
[btop](https://github.com/aristocratos/btop), with the GPU detail of
[nvtop](https://github.com/Syllo/nvtop). On Linux and macOS.

- **CPU**: total use over time, and each CPU's use.
- **Memory**: RAM and swap, and RAM over time.
- **GPUs**: a panel for each, with utilisation and memory, and charts of
  both over time.
  - NVIDIA's, through NVML, loaded at run time from the driver
    (`libnvidia-ml.so`), so no CUDA toolkit is needed: temperature,
    P-state, power, fan, clocks, PCIe, encoder and decoder too.
  - Apple's, on macOS, from the IORegistry, without root.
  - Without either, otop runs without GPU panels.
- **Processes**: every process, with its user, CPU, memory, and for those on
  an NVIDIA GPU, which GPUs, GPU memory and SM use. Search them with `/`,
  and stop the one picked with `k`.

## Install

```sh
cargo install otop
```

## Use

```sh
otop             # sample every second
otop -d 500      # every 500 ms
otop --once      # print one sample as text, and exit
```

| Key                   | Does                                                  |
|-----------------------|-------------------------------------------------------|
| `q`                   | quit                                                  |
| `↑` `↓` `PgUp` `PgDn` | pick a process                                        |
| `/`                   | search processes by command, user or PID              |
| `Enter` / `Esc`       | keep the search / clear it                            |
| `k`, `F9`             | stop the process picked: `t` terminates, `9` kills    |
| `s`                   | sort by CPU, memory, GPU memory, SM, PID, user        |
| `r`                   | reverse the sort                                      |
| `g`                   | list every process, those on a GPU, or on one GPU     |
| `h`                   | hide or show the charts                               |
| `v`                   | GPU charts: split, one per GPU, or combined           |
| `+` `-`               | refresh faster or slower                              |

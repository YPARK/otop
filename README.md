# otop

Monitor NVIDIA GPUs, and the processes on them, in the terminal, like
[nvtop](https://github.com/Syllo/nvtop).

GPUs are read through NVML, loaded at run time from NVIDIA's driver
(`libnvidia-ml.so`), and drawn with [ratatui](https://ratatui.rs). Nothing
in C is built and no CUDA toolkit is needed.

Each GPU gets a panel with utilisation and memory gauges, temperature,
performance state, power, fan, clocks, PCIe throughput and encoder/decoder
use. Below that, each GPU's utilisation and memory over time are charted:
in two charts for each GPU, in one chart for each GPU, or in one chart of
every GPU's utilisation and one of their memory, switched with `v`. Then a table of the processes on them: PID, GPU, compute or graphics, user,
GPU memory, SM/encoder/decoder use, and command line.

## Install

```sh
cargo install --git https://github.com/YPARK/otop
# or, from a clone
cargo install --path .
```

## Use

```sh
otop             # sample every second
otop -d 500      # every 500 ms
otop --once      # print one sample as text, and exit
```

| Key        | Does                                          |
|------------|-----------------------------------------------|
| `q`, `Esc` | quit                                          |
| `↑` `↓` / `j` `k` | pick a process                         |
| `s`        | sort by memory, SM, PID, GPU, user            |
| `r`        | reverse the sort                              |
| `g`        | show one GPU's processes at a time, or all    |
| `h`        | hide or show the history charts               |
| `v`        | charts: split, one per GPU, or combined       |
| `+` `-`    | refresh faster or slower                      |

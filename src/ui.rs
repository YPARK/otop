//! Drawing: CPUs and memory, a panel per GPU, charts of their history,
//! and the processes, like btop.

use crate::app::{App, Charts, Mode, Only, SortBy};
use crate::gpu::{Gpu, GpuUse};
use crate::hist::{HISTORY, History};
use crate::sys::frac;
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Flex, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Bar, BarChart, BarGroup, Block, Borders, Cell, Chart, Clear, Dataset, GraphType,
    LegendPosition, LineGauge, Paragraph, Row, Table, TableState,
};
use std::io::{self, Write};
use std::time::Duration;

const COLORS: [Color; 8] = [
    Color::Cyan,
    Color::Magenta,
    Color::Yellow,
    Color::Green,
    Color::LightBlue,
    Color::LightRed,
    Color::LightGreen,
    Color::LightMagenta,
];

fn gpu_color(i: u32) -> Color {
    COLORS[i as usize % COLORS.len()]
}

/// Green when idle, yellow when busy, red when nearly full.
fn level_color(frac: f64) -> Color {
    if frac >= 0.9 {
        Color::Red
    } else if frac >= 0.6 {
        Color::Yellow
    } else {
        Color::Green
    }
}

pub fn opt(v: Option<u32>) -> String {
    v.map_or("-".into(), |v| v.to_string())
}

pub fn bytes(b: u64) -> String {
    const GIB: f64 = (1u64 << 30) as f64;
    const MIB: f64 = (1u64 << 20) as f64;
    let b = b as f64;
    if b >= GIB {
        format!("{:.1}G", b / GIB)
    } else {
        format!("{:.0}M", b / MIB)
    }
}

/// The GPUs a process is on, as `0,1`.
pub fn gpu_list(g: &GpuUse) -> String {
    let s: Vec<String> = g.gpus.iter().map(u32::to_string).collect();
    s.join(",")
}

/// Rows of the CPU and memory panels, inside their borders.
const TOP: u16 = 8;
/// Rows of each GPU panel.
const PANEL: u16 = 4;
/// Rows the process table keeps, at least.
const PROCS: u16 = 8;

pub fn draw(f: &mut Frame, app: &mut App) {
    let area = f.area();
    let n = app.gpus.gpus.len();
    // Two columns of GPU panels where there is room for them.
    let cols = if area.width >= 140 && n > 1 { 2 } else { 1 };
    let panels = n.div_ceil(cols) as u16 * PANEL;
    // Charts in the top panels where there is height for them as well as
    // for the processes.
    let tall = app.show_charts && area.height >= 30;
    let top = if tall { TOP + 2 } else { 4 };
    let (charts, layout) = chart_layout(app, area, top + panels);

    let [header, top_area, gpus, charts_area, procs, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(top),
        Constraint::Length(panels),
        Constraint::Length(charts),
        Constraint::Min(4),
        Constraint::Length(1),
    ])
    .areas(area);

    draw_header(f, app, header);
    let [cpu, mem] = Layout::horizontal([Constraint::Percentage(60), Constraint::Percentage(40)])
        .areas(top_area);
    draw_cpu(f, app, cpu, tall);
    draw_mem(f, app, mem, tall);
    draw_gpus(f, app, gpus, cols);
    draw_charts(f, app, charts_area, layout);
    draw_procs(f, app, procs);
    draw_footer(f, app, footer);
    if let Mode::Kill(pid, _) = app.mode {
        draw_kill(f, app, pid);
    }
}

fn draw_header(f: &mut Frame, app: &App, area: Rect) {
    let s = &app.sys;
    let n = app.gpus.gpus.len();
    let gpus = match n {
        0 => format!("{}  ", app.gpus.info),
        1 => format!("{}  1 GPU  ", app.gpus.info),
        _ => format!("{}  {n} GPUs  ", app.gpus.info),
    };
    let line = Line::from(vec![
        concat!(" ", env!("CARGO_PKG_NAME"), " ")
            .bold()
            .black()
            .on_cyan(),
        format!("  {}  ", s.host).bold(),
        format!("up {}  ", uptime(sysinfo::System::uptime())).into(),
        gpus.into(),
        format!("every {} ms", app.interval.as_millis()).dark_gray(),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

fn uptime(secs: u64) -> String {
    let (d, h, m) = (secs / 86400, secs / 3600 % 24, secs / 60 % 60);
    if d > 0 {
        format!("{d}d {h}h")
    } else {
        format!("{h}h {m}m")
    }
}

fn gauge<'a>(label: String, ratio: f64) -> LineGauge<'a> {
    LineGauge::default()
        .label(label)
        .ratio(ratio.clamp(0.0, 1.0))
        .filled_style(Style::new().fg(level_color(ratio)))
        .unfilled_style(Style::new().fg(Color::DarkGray))
}

fn draw_cpu(f: &mut Frame, app: &App, area: Rect, tall: bool) {
    let s = &app.sys;
    let block = Block::bordered()
        .border_style(Style::new().fg(Color::Cyan))
        .title(Line::from(vec![
            " CPU ".cyan().bold(),
            format!("{} ", s.cpu_brand).dark_gray(),
        ]))
        .title(Line::from(format!(" {:.0}% ", s.cpu)).right_aligned());
    let inner = block.inner(area);
    f.render_widget(block, area);
    let cores = if tall {
        let (chart, divider, cores) = cpu_layout(s.cores.len() as u16, inner);
        draw_history(f, chart, &s.hist_cpu, Color::Cyan);
        if let Some(d) = divider {
            let side = if d.width == 1 {
                Borders::LEFT
            } else {
                Borders::TOP
            };
            f.render_widget(Block::new().borders(side).dark_gray(), d);
        }
        cores
    } else {
        // No chart: the total as a gauge, over the CPUs.
        let [total, cores] =
            Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).areas(inner);
        f.render_widget(
            gauge(format!("CPU {:>3.0}%", s.cpu), s.cpu as f64 / 100.0),
            total,
        );
        cores
    };
    draw_cores(f, &s.cores, cores);
}

/// The CPU chart's area, a divider's, and the CPUs' own, inside the CPU
/// panel: side by side where the chart keeps 20 columns, or else the chart
/// over a few rows of a bar per CPU; the divider is a column or a row
/// between them, so the chart's line does not run into the bars. Without
/// CPUs, the chart alone.
fn cpu_layout(n: u16, inner: Rect) -> (Rect, Option<Rect>, Rect) {
    const CHART: u16 = 20;
    if n == 0 {
        return (inner, None, Rect::default());
    }
    let grid = n.div_ceil(inner.height.max(1)) * CELL;
    let spaced = n * 2 - 1;
    let bars = if spaced + CHART < inner.width {
        spaced
    } else {
        n
    };
    let side = if grid + 1 + CHART <= inner.width {
        grid
    } else {
        bars
    };
    let (dir, last) = if side + 1 + CHART <= inner.width {
        (Direction::Horizontal, Constraint::Length(side))
    } else {
        (Direction::Vertical, Constraint::Length(3))
    };
    let [chart, divider, cores] = Layout::default()
        .direction(dir)
        .constraints([Constraint::Min(0), Constraint::Length(1), last])
        .areas(inner);
    (chart, Some(divider), cores)
}

/// Columns a gauge per CPU takes, labelled.
const CELL: u16 = 16;

/// Each CPU's use: a small labelled gauge each where they fit, or else a
/// bar each, its height its use.
fn draw_cores(f: &mut Frame, cores: &[f32], area: Rect) {
    let n = cores.len() as u16;
    if area.height == 0 || n == 0 {
        return;
    }
    let cols = (area.width / CELL).max(1);
    let rows = n.div_ceil(cols);
    if rows <= area.height {
        for (i, &u) in cores.iter().enumerate() {
            let (r, c) = (i as u16 % rows, i as u16 / rows);
            let cell = Rect::new(area.x + c * CELL, area.y + r, CELL - 1, 1).intersection(area);
            f.render_widget(gauge(format!("{i:>3} {u:>3.0}%"), u as f64 / 100.0), cell);
        }
        return;
    }
    let bars: Vec<Bar> = cores
        .iter()
        .map(|&u| {
            Bar::default()
                .value(u.round() as u64)
                .text_value("")
                .style(Style::new().fg(level_color(u as f64 / 100.0)))
        })
        .collect();
    // A gap between bars, where there is room for one.
    let gap = u16::from(n * 2 <= area.width + 1);
    f.render_widget(
        BarChart::default()
            .data(BarGroup::default().bars(&bars))
            .bar_width(1)
            .bar_gap(gap)
            .max(100),
        area,
    );
}

fn draw_mem(f: &mut Frame, app: &App, area: Rect, tall: bool) {
    let s = &app.sys;
    let block = Block::bordered()
        .border_style(Style::new().fg(Color::Magenta))
        .title(" Memory ".magenta().bold());
    let inner = block.inner(area);
    f.render_widget(block, area);
    let [ram, swap, chart] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(0),
    ])
    .areas(inner);
    let mem = frac(s.mem_used, s.mem_total);
    f.render_widget(
        gauge(
            format!("RAM  {}/{}", bytes(s.mem_used), bytes(s.mem_total)),
            mem,
        ),
        ram,
    );
    f.render_widget(
        gauge(
            format!("SWAP {}/{}", bytes(s.swap_used), bytes(s.swap_total)),
            frac(s.swap_used, s.swap_total),
        ),
        swap,
    );
    if tall {
        draw_history(f, chart, &s.hist_mem, Color::Magenta);
    }
}

/// How many samples fit in `area`, less `margin` columns for the axis:
/// Braille gives two points a cell.
fn samples(area: Rect, margin: u16) -> usize {
    (area.width.saturating_sub(margin) as usize * 2).clamp(10, HISTORY)
}

/// A line chart from 0 to 100% of the last `width` samples.
fn line_chart<'a>(datasets: Vec<Dataset<'a>>, width: usize, y: &'a [&'a str]) -> Chart<'a> {
    Chart::new(datasets)
        .x_axis(
            Axis::default()
                .bounds([0.0, width as f64])
                .style(Style::new().dark_gray()),
        )
        .y_axis(
            Axis::default()
                .bounds([0.0, 100.0])
                .labels(y.iter().copied())
                .style(Style::new().dark_gray()),
        )
}

fn line<'a>(color: Color, data: &'a [(f64, f64)]) -> Dataset<'a> {
    Dataset::default()
        .marker(Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::new().fg(color))
        .data(data)
}

/// A chart of one history, without borders or legend, for the top panels.
fn draw_history(f: &mut Frame, area: Rect, h: &History, color: Color) {
    let width = samples(area, 4);
    let data = points(h, width);
    f.render_widget(
        line_chart(vec![line(color, &data)], width, &["0", "100"]),
        area,
    );
}

/// A history's last `width` samples as points, the newest at `width`.
fn points(h: &History, width: usize) -> Vec<(f64, f64)> {
    let start = width - h.len().min(width);
    h.last(width)
        .enumerate()
        .map(|(i, v)| ((start + i) as f64, v))
        .collect()
}

/// `n` cells, `cols` to a row, each row `row` high.
fn grid(area: Rect, n: usize, cols: usize, row: Constraint) -> Vec<Rect> {
    let rows = Layout::vertical(vec![row; n.div_ceil(cols)]).split(area);
    rows.iter()
        .flat_map(|r| {
            Layout::horizontal(vec![Constraint::Ratio(1, cols as u32); cols])
                .split(*r)
                .to_vec()
        })
        .take(n)
        .collect()
}

fn draw_gpus(f: &mut Frame, app: &App, area: Rect, cols: usize) {
    let gpus = &app.gpus.gpus;
    for (g, cell) in gpus
        .iter()
        .zip(grid(area, gpus.len(), cols, Constraint::Length(PANEL)))
    {
        draw_gpu(f, g, cell);
    }
}

fn draw_gpu(f: &mut Frame, g: &Gpu, area: Rect) {
    let color = gpu_color(g.index);
    let title = Line::from(vec![
        Span::styled(format!(" GPU{} ", g.index), Style::new().fg(color).bold()),
        Span::raw(format!("{} ", g.name)),
    ]);
    let mut right = Vec::new();
    if let Some(t) = g.temp {
        let c = if t >= 85 {
            Color::Red
        } else if t >= 70 {
            Color::Yellow
        } else {
            Color::Reset
        };
        right.push(Span::styled(format!(" {t}°C "), Style::new().fg(c)));
    }
    if let Some(p) = &g.pstate {
        right.push(Span::raw(format!("{p} ")).dark_gray());
    }
    let block = Block::bordered()
        .border_style(Style::new().fg(color))
        .title(title)
        .title(Line::from(right).right_aligned());
    let inner = block.inner(area);
    f.render_widget(block, area);

    let [gauges, stats] =
        Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(inner);
    let [gu, gm] =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(gauges);
    f.render_widget(
        gauge(
            format!("GPU {:>3}%", opt(g.util)),
            g.util.unwrap_or(0) as f64 / 100.0,
        ),
        gu,
    );
    f.render_widget(
        gauge(
            format!(" MEM {}/{}", bytes(g.mem_used), bytes(g.mem_total)),
            g.mem_frac(),
        ),
        gm,
    );

    let mut parts: Vec<String> = Vec::new();
    if let Some(c) = g.cores {
        parts.push(format!("{c} cores"));
    }
    if let Some(p) = g.power {
        match g.power_limit {
            Some(l) => parts.push(format!("POW {p}/{l}W")),
            None => parts.push(format!("POW {p}W")),
        }
    }
    if let Some(fan) = g.fan {
        parts.push(format!("FAN {fan}%"));
    }
    if let Some(c) = g.clock_gr {
        match g.clock_gr_max {
            Some(m) => parts.push(format!("GR {c}/{m}MHz")),
            None => parts.push(format!("GR {c}MHz")),
        }
    }
    if let Some(c) = g.clock_mem {
        parts.push(format!("MEM {c}MHz"));
    }
    if let (Some(tx), Some(rx)) = (g.pcie_tx, g.pcie_rx) {
        parts.push(format!("PCIe ↑{} ↓{}", rate(tx), rate(rx)));
    }
    if let (Some(e), Some(d)) = (g.enc, g.dec) {
        parts.push(format!("ENC {e}% DEC {d}%"));
    }
    f.render_widget(Paragraph::new(parts.join("  ")).dark_gray(), stats);
}

/// A throughput, given in bytes per second.
fn rate(b: u64) -> String {
    const K: f64 = 1024.0;
    let b = b as f64;
    if b >= K * K * K {
        format!("{:.1}GB/s", b / (K * K * K))
    } else if b >= K * K {
        format!("{:.0}MB/s", b / (K * K))
    } else {
        format!("{:.0}KB/s", b / K)
    }
}

/// How the GPUs' charts are laid out, as there is room.
#[derive(Clone, Copy)]
enum GpuCharts {
    None,
    /// For each GPU, `cols` GPUs to a row, as `Charts` says.
    PerGpu(usize),
    /// One chart of every GPU's utilisation, and one of their memory.
    Combined,
}

/// Rows of GPU chart, and how they are laid out, that fit under `above`
/// rows while leaving room for some processes.
fn chart_layout(app: &App, area: Rect, above: u16) -> (u16, GpuCharts) {
    let n = app.gpus.gpus.len();
    if !app.show_charts || n == 0 {
        return (0, GpuCharts::None);
    }
    // The header, the footer, and a table of a few processes.
    let room = area.height.saturating_sub(above + 2 + PROCS);
    if app.charts != Charts::Combined {
        // As many GPUs to a row as have 50 columns a chart, at most 4.
        let per_gpu = if app.charts == Charts::Split { 2 } else { 1 };
        let cols = (area.width as usize / (50 * per_gpu)).clamp(1, 4).min(n);
        let rows = n.div_ceil(cols) as u16;
        // Each chart 10 rows high, or less, down to 7, to fit them all.
        let per = (room / rows).min(10);
        if per >= 7 {
            return (per * rows, GpuCharts::PerGpu(cols));
        }
    }
    if room >= 8 {
        (room.min(12), GpuCharts::Combined)
    } else {
        (0, GpuCharts::None)
    }
}

/// A line on a chart: its name, its colour, and its samples.
type Series<'a> = (String, Color, &'a History);

fn draw_charts(f: &mut Frame, app: &App, area: Rect, layout: GpuCharts) {
    let gpus = &app.gpus.gpus;
    let cols = match layout {
        GpuCharts::None => return,
        GpuCharts::PerGpu(cols) => cols,
        GpuCharts::Combined => {
            let [left, right] =
                Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .areas(area);
            let lines = |pick: fn(&Gpu) -> &History| -> Vec<Series> {
                gpus.iter()
                    .map(|g| (format!("GPU{}", g.index), gpu_color(g.index), pick(g)))
                    .collect()
            };
            let all = Color::DarkGray;
            draw_chart(
                f,
                left,
                " GPU utilisation % ".into(),
                all,
                lines(|g| &g.hist_util),
            );
            draw_chart(
                f,
                right,
                " GPU memory used % ".into(),
                all,
                lines(|g| &g.hist_mem),
            );
            return;
        }
    };
    let rows = gpus.len().div_ceil(cols) as u32;
    for (g, cell) in gpus
        .iter()
        .zip(grid(area, gpus.len(), cols, Constraint::Ratio(1, rows)))
    {
        let color = gpu_color(g.index);
        let gpu = Span::styled(format!(" GPU{} ", g.index), Style::new().fg(color).bold());
        let util = (
            format!("util {:>3}%", opt(g.util)),
            Color::Green,
            &g.hist_util,
        );
        let mem = (
            format!("mem {:>3.0}%", 100.0 * g.mem_frac()),
            Color::Magenta,
            &g.hist_mem,
        );
        if app.charts == Charts::Split {
            let [left, right] =
                Layout::horizontal([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)]).areas(cell);
            let util_title = Line::from(vec![gpu.clone(), "utilisation ".into()]);
            let mem_title = Line::from(vec![
                gpu,
                "memory ".into(),
                format!("{}/{} ", bytes(g.mem_used), bytes(g.mem_total)).dark_gray(),
            ]);
            draw_chart(f, left, util_title, color, vec![util]);
            draw_chart(f, right, mem_title, color, vec![mem]);
        } else {
            let title = Line::from(vec![gpu, format!("{} ", g.name).dark_gray()]);
            draw_chart(f, cell, title, color, vec![util, mem]);
        }
    }
}

fn draw_chart(f: &mut Frame, area: Rect, title: Line, border: Color, series: Vec<Series>) {
    let width = samples(area, 8);
    let data: Vec<Vec<(f64, f64)>> = series.iter().map(|(_, _, h)| points(h, width)).collect();
    let datasets = series
        .iter()
        .zip(&data)
        .map(|((name, color, _), d)| line(*color, d).name(name.clone()))
        .collect();
    let span = series
        .first()
        .map_or(0, |(_, _, h)| h.span(width).as_secs());
    let mut chart = line_chart(datasets, width, &["0", "50", "100"])
        .block(
            Block::bordered()
                .title(title)
                .border_style(Style::new().fg(border)),
        )
        .legend_position(Some(LegendPosition::TopLeft))
        // Show the legend however small the chart is.
        .hidden_legend_constraints((Constraint::Min(0), Constraint::Min(0)));
    chart = chart.x_axis(
        Axis::default()
            .bounds([0.0, width as f64])
            .labels([format!("-{span}s"), "now".into()])
            .style(Style::new().dark_gray()),
    );
    f.render_widget(chart, area);
}

fn draw_procs(f: &mut Frame, app: &mut App, area: Rect) {
    // GPU columns where the GPUs' source says which process uses them.
    let has_gpus = app.gpus.per_process();
    let arrow = if app.descending { "▼" } else { "▲" };
    let head = |label: &str, by: Option<SortBy>| {
        if by == Some(app.sort) {
            Cell::from(format!("{label}{arrow}")).style(Style::new().fg(Color::Cyan))
        } else {
            Cell::from(label.to_string())
        }
    };
    let mut header = vec![
        head("PID", Some(SortBy::Pid)),
        head("USER", Some(SortBy::User)),
        head("CPU%", Some(SortBy::Cpu)),
        head("MEM", Some(SortBy::Mem)),
    ];
    let mut widths = vec![
        Constraint::Length(8),
        Constraint::Length(10),
        Constraint::Length(6),
        Constraint::Length(7),
    ];
    if has_gpus {
        header.extend([
            head("GPU", None),
            head("GPU MEM", Some(SortBy::GpuMem)),
            head("SM%", Some(SortBy::Sm)),
        ]);
        widths.extend([
            Constraint::Length(4),
            Constraint::Length(8),
            Constraint::Length(4),
        ]);
    }
    header.push(head("COMMAND", None));
    widths.push(Constraint::Min(10));
    let header =
        Row::new(header).style(Style::new().add_modifier(Modifier::BOLD | Modifier::REVERSED));

    // Rows for those on screen alone, scrolled to keep the one picked in
    // sight: inside the borders, under the header.
    let height = area.height.saturating_sub(3) as usize;
    let selected = app.selected();
    if let Some(i) = selected {
        if i < app.offset {
            app.offset = i;
        } else if i >= app.offset + height {
            app.offset = i + 1 - height;
        }
    }
    app.offset = app.offset.min(app.rows.len().saturating_sub(height));
    let shown = app.offset..(app.offset + height).min(app.rows.len());
    let rows = shown.clone().map(|i| {
        let (p, g) = app.row(i);
        let mut cells = vec![
            Cell::from(p.pid.to_string()),
            Cell::from(&*p.user),
            Cell::from(format!("{:.1}", p.cpu))
                .style(Style::new().fg(level_color(p.cpu as f64 / 100.0))),
            Cell::from(bytes(p.mem)),
        ];
        if has_gpus {
            let color = g
                .and_then(|g| g.gpus.first())
                .map_or(Color::Reset, |&i| gpu_color(i));
            cells.extend([
                Cell::from(g.map_or(String::new(), gpu_list)).style(Style::new().fg(color)),
                Cell::from(g.and_then(|g| g.mem).map_or(String::new(), bytes)),
                Cell::from(
                    g.and_then(|g| g.sm)
                        .map_or(String::new(), |s| s.to_string()),
                ),
            ]);
        }
        cells.push(Cell::from(p.command.as_str()));
        Row::new(cells)
    });

    let mut title = vec![" Processes".bold()];
    match app.only {
        Only::All => {}
        Only::AnyGpu => title.push(" on GPUs".into()),
        Only::Gpu(g) => title.push(format!(" on GPU{g}").into()),
    }
    if !app.query.is_empty() {
        title.push(format!(" matching \"{}\"", app.query).yellow());
    }
    title.push(" ".into());
    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::bordered()
                .title(Line::from(title))
                .title(Line::from(format!(" {} ", app.rows.len())).right_aligned())
                .border_style(Style::new().dark_gray()),
        )
        .row_highlight_style(Style::new().bg(Color::DarkGray))
        .highlight_symbol("▶");
    let mut state = TableState::default().with_selected(selected.map(|i| i - shown.start));
    f.render_stateful_widget(table, area, &mut state);
}

fn draw_footer(f: &mut Frame, app: &App, area: Rect) {
    if app.mode == Mode::Search {
        let line = Line::from(vec![
            " / ".black().on_yellow(),
            format!(" {}", app.query).into(),
            "█".slow_blink(),
            "   Enter keeps it, Esc clears it".dark_gray(),
        ]);
        f.render_widget(Paragraph::new(line), area);
        return;
    }
    if let Some((text, at)) = &app.message
        && at.elapsed() < Duration::from_secs(4)
    {
        f.render_widget(Paragraph::new(format!(" {text}")).yellow(), area);
        return;
    }
    // The keys, most wanted first, as many as fit.
    let gpu = app.gpus.per_process();
    let charts = !app.gpus.gpus.is_empty();
    let mut hints: Vec<(&str, String)> = vec![
        ("q", "quit".into()),
        ("/", "search".into()),
        ("k", "kill".into()),
        ("s", format!("sort: {}", app.sort.label())),
        ("h", "charts".into()),
    ];
    if gpu {
        hints.push(("g", "GPU filter".into()));
    }
    if charts {
        hints.push(("v", app.charts.label().into()));
    }
    hints.extend([("r", "reverse".into()), ("+/-", "refresh".into())]);
    let mut line = Line::default();
    for (k, what) in hints {
        let key = Span::styled(k, Style::new().black().on_cyan());
        let what = Span::raw(format!(" {what}  "));
        if line.width() + key.width() + what.width() > area.width as usize {
            break;
        }
        line.push_span(key);
        line.push_span(what);
    }
    f.render_widget(Paragraph::new(line), area);
}

/// Asks whether to stop the process picked, over the rest.
fn draw_kill(f: &mut Frame, app: &App, pid: u32) {
    let what = app
        .sys
        .procs
        .iter()
        .find(|p| p.pid == pid)
        .map_or(String::new(), |p| format!("{} {}", p.user, p.command));
    let [area] = Layout::horizontal([Constraint::Length(64)])
        .flex(Flex::Center)
        .areas(f.area());
    let [area] = Layout::vertical([Constraint::Length(6)])
        .flex(Flex::Center)
        .areas(area);
    let text = vec![
        Line::from(vec![format!("Stop process {pid}?").bold()]),
        Line::from(what).dark_gray(),
        Line::from(""),
        Line::from(vec![
            " t ".black().on_yellow(),
            " terminate   ".into(),
            " 9 ".black().on_red(),
            " kill   ".into(),
            "any other key cancels".dark_gray(),
        ]),
    ];
    f.render_widget(Clear, area);
    f.render_widget(
        Paragraph::new(text).block(
            Block::bordered()
                .title(" Kill ".red().bold())
                .border_style(Style::new().fg(Color::Red)),
        ),
        area,
    );
}

/// `--once`: the CPUs, memory, GPUs and busiest processes, as plain text,
/// with the same columns as the table.
pub fn print_once(app: &App, out: &mut impl Write) -> io::Result<()> {
    let (sys, gpus) = (&app.sys, &app.gpus);
    writeln!(
        out,
        "{}  {}  CPU {:.0}%  MEM {}/{}  SWAP {}/{}",
        sys.host,
        sys.cpu_brand,
        sys.cpu,
        bytes(sys.mem_used),
        bytes(sys.mem_total),
        bytes(sys.swap_used),
        bytes(sys.swap_total),
    )?;
    writeln!(out, "{}", gpus.info)?;
    for g in &gpus.gpus {
        writeln!(
            out,
            "GPU{} {:<28} util {:>3}%  mem {}/{}  {}  {}",
            g.index,
            g.name,
            opt(g.util),
            bytes(g.mem_used),
            bytes(g.mem_total),
            g.temp.map_or("-".into(), |t| format!("{t}°C")),
            g.power.map_or("-".into(), |p| format!("{p}W")),
        )?;
    }
    let has_gpus = gpus.per_process();
    writeln!(out)?;
    write!(
        out,
        "{:>8} {:<10} {:>6} {:>8}",
        "PID", "USER", "CPU%", "MEM"
    )?;
    if has_gpus {
        write!(out, " {:>5} {:>8} {:>4}", "GPU", "GPU MEM", "SM%")?;
    }
    writeln!(out, "  COMMAND")?;
    for i in 0..app.rows.len().min(20) {
        let (p, g) = app.row(i);
        write!(
            out,
            "{:>8} {:<10} {:>6.1} {:>8}",
            p.pid,
            p.user,
            p.cpu,
            bytes(p.mem)
        )?;
        if has_gpus {
            write!(
                out,
                " {:>5} {:>8} {:>4}",
                g.map_or("-".into(), gpu_list),
                g.and_then(|g| g.mem).map_or("-".into(), bytes),
                opt(g.and_then(|g| g.sm)),
            )?;
        }
        writeln!(out, "  {}", p.command)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpus_beside_the_chart_where_it_keeps_its_width() {
        // A 10-CPU Mac's panel at 80 columns.
        let (chart, divider, cores) = cpu_layout(10, Rect::new(0, 0, 46, 8));
        let d = divider.unwrap();
        assert_eq!((d.width, d.height), (1, 8));
        assert!(chart.width >= 20);
        assert_eq!(chart.right(), d.x);
        assert_eq!(d.right(), cores.x);
    }

    #[test]
    fn cpus_under_the_chart_where_they_do_not_fit_beside_it() {
        let (chart, divider, cores) = cpu_layout(64, Rect::new(0, 0, 56, 8));
        let d = divider.unwrap();
        assert_eq!((d.width, d.height), (56, 1));
        assert_eq!(chart.bottom(), d.y);
        assert_eq!(d.bottom(), cores.y);
        assert_eq!(cores.height, 3);
    }

    #[test]
    fn no_cpus_is_the_chart_alone() {
        let inner = Rect::new(0, 0, 46, 8);
        assert_eq!(cpu_layout(0, inner), (inner, None, Rect::default()));
    }
}

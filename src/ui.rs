//! Drawing: a panel per GPU, charts of their history, and their processes.

use crate::app::{App, Charts};
use crate::gpu::{Gpu, HISTORY};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, Cell, Chart, Dataset, GraphType, LineGauge, Paragraph, Row, Table,
};
use std::collections::VecDeque;

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

pub fn draw(f: &mut Frame, app: &mut App) {
    let area = f.area();
    let n = app.sampler.gpus.len().max(1);
    // Two columns of GPU panels where there is room for them.
    let cols = if area.width >= 140 && n > 1 { 2 } else { 1 };
    let rows = n.div_ceil(cols) as u16;
    let (charts, chart_cols) = chart_layout(app, area, rows * 4);

    let [header, gpus, charts_area, procs, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(rows * 4),
        Constraint::Length(charts),
        Constraint::Min(4),
        Constraint::Length(1),
    ])
    .areas(area);

    draw_header(f, app, header);
    draw_gpus(f, app, gpus, cols);
    if charts > 0 {
        draw_charts(f, app, charts_area, chart_cols);
    }
    draw_procs(f, app, procs);
    draw_footer(f, app, footer);
}

fn draw_header(f: &mut Frame, app: &App, area: Rect) {
    let s = &app.sampler;
    let line = Line::from(vec![
        concat!(" ", env!("CARGO_PKG_NAME"), " ")
            .bold()
            .black()
            .on_cyan(),
        format!("  Driver {}  CUDA {}  ", s.driver, s.cuda).into(),
        format!(
            "{} GPU{}  ",
            s.gpus.len(),
            if s.gpus.len() == 1 { "" } else { "s" }
        )
        .into(),
        format!("every {} ms", app.interval.as_millis()).dark_gray(),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

fn draw_gpus(f: &mut Frame, app: &App, area: Rect, cols: usize) {
    let gpus = &app.sampler.gpus;
    let rows = gpus.len().div_ceil(cols);
    let row_areas = Layout::vertical(vec![Constraint::Length(4); rows]).split(area);
    for (r, row) in row_areas.iter().enumerate() {
        let cells = Layout::horizontal(vec![Constraint::Ratio(1, cols as u32); cols]).split(*row);
        for c in 0..cols {
            if let Some(g) = gpus.get(r * cols + c) {
                draw_gpu(f, g, cells[c]);
            }
        }
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

    let util = g.util.unwrap_or(0) as f64 / 100.0;
    f.render_widget(
        LineGauge::default()
            .label(format!("GPU {:>3}%", opt(g.util)))
            .ratio(util.clamp(0.0, 1.0))
            .filled_style(Style::new().fg(level_color(util)))
            .unfilled_style(Style::new().fg(Color::DarkGray)),
        gu,
    );
    let mem = g.mem_frac();
    f.render_widget(
        LineGauge::default()
            .label(format!(" MEM {}/{}", bytes(g.mem_used), bytes(g.mem_total)))
            .ratio(mem.clamp(0.0, 1.0))
            .filled_style(Style::new().fg(level_color(mem)))
            .unfilled_style(Style::new().fg(Color::DarkGray)),
        gm,
    );

    let mut parts: Vec<String> = Vec::new();
    if let Some(p) = g.power {
        match g.power_limit {
            Some(l) => parts.push(format!("POW {}/{}W", p / 1000, l / 1000)),
            None => parts.push(format!("POW {}W", p / 1000)),
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

/// PCIe throughput, which NVML gives in KB/s.
fn rate(kb: u32) -> String {
    if kb >= 1024 * 1024 {
        format!("{:.1}GB/s", kb as f64 / (1024.0 * 1024.0))
    } else if kb >= 1024 {
        format!("{:.0}MB/s", kb as f64 / 1024.0)
    } else {
        format!("{kb}KB/s")
    }
}

/// Rows of chart, and charts to a row, that fit under `panels` rows of
/// GPU panels while leaving room for some processes; (0, 0) for none.
fn chart_layout(app: &App, area: Rect, panels: u16) -> (u16, usize) {
    if !app.show_charts {
        return (0, 0);
    }
    // The header, the footer, and a table of a few processes.
    let room = area.height.saturating_sub(panels + 2 + 8);
    let n = app.sampler.gpus.len();
    if app.charts != Charts::Combined && n > 0 {
        // As many GPUs to a row as have 50 columns a chart, at most 4.
        let per_gpu = if app.charts == Charts::Split { 2 } else { 1 };
        let cols = (area.width as usize / (50 * per_gpu)).clamp(1, 4).min(n);
        let rows = n.div_ceil(cols) as u16;
        // Each chart 10 rows high, or less, down to 7, to fit them all.
        let per = (room / rows).min(10);
        if per >= 7 {
            return (per * rows, cols);
        }
    }
    if room >= 8 { (room.min(12), 0) } else { (0, 0) }
}

/// A line on a chart: its name, its colour, and its samples.
type Series<'a> = (String, Color, &'a VecDeque<f64>);

fn draw_charts(f: &mut Frame, app: &App, area: Rect, cols: usize) {
    let gpus = &app.sampler.gpus;
    if cols == 0 {
        // One chart of utilisation, and one of memory, a line per GPU.
        let [left, right] =
            Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .areas(area);
        let util = gpus
            .iter()
            .map(|g| (format!("GPU{}", g.index), gpu_color(g.index), &g.hist_util))
            .collect();
        let mem = gpus
            .iter()
            .map(|g| (format!("GPU{}", g.index), gpu_color(g.index), &g.hist_mem))
            .collect();
        let title = |t: &str| Line::from(t.to_string());
        draw_chart(
            f,
            app,
            left,
            title(" GPU utilisation % "),
            Color::DarkGray,
            util,
        );
        draw_chart(
            f,
            app,
            right,
            title(" Memory used % "),
            Color::DarkGray,
            mem,
        );
        return;
    }
    // For each GPU, a chart of its utilisation and its memory, or one of
    // each.
    let rows = gpus.len().div_ceil(cols);
    let row_areas = Layout::vertical(vec![Constraint::Ratio(1, rows as u32); rows]).split(area);
    for (r, row) in row_areas.iter().enumerate() {
        let cells = Layout::horizontal(vec![Constraint::Ratio(1, cols as u32); cols]).split(*row);
        for (c, cell) in cells.iter().enumerate() {
            let Some(g) = gpus.get(r * cols + c) else {
                continue;
            };
            let color = gpu_color(g.index);
            let title = Line::from(vec![
                Span::styled(format!(" GPU{} ", g.index), Style::new().fg(color).bold()),
                Span::raw(format!("{} ", g.name)).dark_gray(),
            ]);
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
                    Layout::horizontal([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)])
                        .areas(*cell);
                let gpu = Span::styled(format!(" GPU{} ", g.index), Style::new().fg(color).bold());
                let util_title = Line::from(vec![gpu.clone(), "utilisation ".into()]);
                let mem_title = Line::from(vec![
                    gpu,
                    "memory ".into(),
                    Span::raw(format!("{}/{} ", bytes(g.mem_used), bytes(g.mem_total))).dark_gray(),
                ]);
                draw_chart(f, app, left, util_title, color, vec![util]);
                draw_chart(f, app, right, mem_title, color, vec![mem]);
            } else {
                draw_chart(f, app, *cell, title, color, vec![util, mem]);
            }
        }
    }
}

fn draw_chart(
    f: &mut Frame,
    app: &App,
    area: Rect,
    title: Line,
    border: Color,
    series: Vec<Series>,
) {
    // Braille gives two points a cell; show as many samples as fit.
    let width = (area.width.saturating_sub(8) as usize * 2).clamp(10, HISTORY);
    let data: Vec<Vec<(f64, f64)>> = series
        .iter()
        .map(|(_, _, h)| {
            let skip = h.len().saturating_sub(width);
            let start = h.len() - skip;
            h.iter()
                .skip(skip)
                .enumerate()
                .map(|(i, &v)| ((i + width - start) as f64, v))
                .collect()
        })
        .collect();
    let datasets: Vec<Dataset> = series
        .iter()
        .zip(&data)
        .map(|((name, color, _), d)| {
            Dataset::default()
                .name(name.clone())
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::new().fg(*color))
                .data(d)
        })
        .collect();
    let span = (width as f64 * app.interval.as_secs_f64()).round() as u64;
    let chart = Chart::new(datasets)
        .block(
            Block::bordered()
                .title(title)
                .border_style(Style::new().fg(border)),
        )
        .x_axis(
            Axis::default()
                .bounds([0.0, width as f64])
                .labels([format!("-{span}s"), "now".into()])
                .style(Style::new().dark_gray()),
        )
        .y_axis(
            Axis::default()
                .bounds([0.0, 100.0])
                .labels(["0", "50", "100"])
                .style(Style::new().dark_gray()),
        )
        .legend_position(Some(ratatui::widgets::LegendPosition::TopLeft))
        // Show the legend however small the chart is.
        .hidden_legend_constraints((Constraint::Min(0), Constraint::Min(0)));
    f.render_widget(chart, area);
}

fn draw_procs(f: &mut Frame, app: &mut App, area: Rect) {
    let procs = app.processes();
    let arrow = if app.descending { "▼" } else { "▲" };
    let head = |label: &str, by: crate::app::SortBy| {
        if app.sort == by {
            Cell::from(format!("{label}{arrow}")).style(Style::new().fg(Color::Cyan))
        } else {
            Cell::from(label.to_string())
        }
    };
    use crate::app::SortBy;
    let header = Row::new(vec![
        head("PID", SortBy::Pid),
        head("GPU", SortBy::Gpu),
        Cell::from("TYPE"),
        head("USER", SortBy::User),
        head("GPU MEM", SortBy::Mem),
        head("SM%", SortBy::Sm),
        Cell::from("ENC%"),
        Cell::from("DEC%"),
        Cell::from("COMMAND"),
    ])
    .style(Style::new().add_modifier(Modifier::BOLD | Modifier::REVERSED));

    let rows = procs.iter().map(|p| {
        Row::new(vec![
            Cell::from(p.pid.to_string()),
            Cell::from(Span::styled(
                p.gpu.to_string(),
                Style::new().fg(gpu_color(p.gpu)),
            )),
            Cell::from(p.kind.label()),
            Cell::from(p.user.clone()),
            Cell::from(p.mem.map_or("-".into(), bytes)),
            Cell::from(opt(p.sm)),
            Cell::from(opt(p.enc)),
            Cell::from(opt(p.dec)),
            Cell::from(p.command.clone()),
        ])
    });
    let which = match app.only_gpu {
        Some(g) => format!(" Processes on GPU{g} "),
        None => " Processes ".into(),
    };
    let table = Table::new(
        rows,
        [
            Constraint::Length(8),
            Constraint::Length(4),
            Constraint::Length(4),
            Constraint::Length(12),
            Constraint::Length(9),
            Constraint::Length(5),
            Constraint::Length(5),
            Constraint::Length(5),
            Constraint::Min(10),
        ],
    )
    .header(header)
    .block(
        Block::bordered()
            .title(which)
            .title(Line::from(format!(" {} ", procs.len())).right_aligned())
            .border_style(Style::new().dark_gray()),
    )
    .row_highlight_style(Style::new().bg(Color::DarkGray))
    .highlight_symbol("▶");

    if procs.is_empty() {
        app.table.select(None);
    } else if app.table.selected().is_none_or(|i| i >= procs.len()) {
        app.table.select(Some(
            procs
                .len()
                .saturating_sub(1)
                .min(app.table.selected().unwrap_or(0)),
        ));
    }
    f.render_stateful_widget(table, area, &mut app.table);
}

fn draw_footer(f: &mut Frame, app: &App, area: Rect) {
    let key = |k: &'static str| Span::styled(k, Style::new().black().on_cyan());
    let line = Line::from(vec![
        key("q"),
        " quit  ".into(),
        key("↑↓"),
        " select  ".into(),
        key("s"),
        format!(" sort: {}  ", app.sort.label()).into(),
        key("r"),
        " reverse  ".into(),
        key("g"),
        " GPU filter  ".into(),
        key("h"),
        " charts  ".into(),
        key("v"),
        format!(" charts: {}  ", app.charts.label()).into(),
        key("+/-"),
        " refresh".into(),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

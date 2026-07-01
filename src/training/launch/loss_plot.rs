use std::fs;
use std::path::{Path, PathBuf};

use plotters::prelude::*;

use super::output::RunOutput;
use crate::AppResult;

const SIZE: (u32, u32) = (900, 530);
const BG: RGBColor = RGBColor(12, 17, 23);
const GRID: RGBColor = RGBColor(31, 40, 52);
const AXIS: RGBColor = RGBColor(64, 74, 88);
const TEXT: RGBColor = RGBColor(199, 208, 219);
const MUTED: RGBColor = RGBColor(146, 156, 170);
const TRAIN_RAW: RGBColor = RGBColor(255, 145, 0);
const TRAIN_EMA: RGBColor = RGBColor(36, 202, 255);
const VALID: RGBColor = RGBColor(230, 250, 143);

pub(super) fn write_loss_plot(run_output: &RunOutput) -> AppResult<Option<PathBuf>> {
    let train = read_series(&run_output.path("metrics/train/epoch-1/Loss.log"))?;
    if train.is_empty() {
        return Ok(None);
    }

    let log_interval = read_log_interval(run_output)?.max(1);
    let train = train
        .into_iter()
        .enumerate()
        .map(|(index, loss)| ((index * log_interval) as f64, loss))
        .collect::<Vec<_>>();
    let train_ema = ema(&train);

    let valid_loss = read_series(&run_output.path("metrics/valid/epoch-1/Validation_loss.log"))?;
    let valid_steps = read_series(&run_output.path("metrics/valid/epoch-1/Completed_steps.log"))?;
    let valid = valid_steps.into_iter().zip(valid_loss).collect::<Vec<_>>();

    let path = run_output.path("loss.png");
    draw(&path, &train, &train_ema, &valid)?;
    Ok(Some(path))
}

fn draw(
    path: &Path,
    train: &[(f64, f64)],
    train_ema: &[(f64, f64)],
    valid: &[(f64, f64)],
) -> AppResult {
    let root = BitMapBackend::new(path, SIZE).into_drawing_area();
    root.fill(&BG).map_err(plot_err)?;

    let ((x_min, x_max), (y_min, y_max)) = bounds(train, valid);
    let mut chart = ChartBuilder::on(&root)
        .caption("Loss vs step", ("sans-serif", 18).into_font().color(&TEXT))
        .margin(14)
        .x_label_area_size(45)
        .y_label_area_size(55)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .map_err(plot_err)?;

    chart
        .configure_mesh()
        .axis_style(AXIS)
        .bold_line_style(GRID)
        .light_line_style(GRID)
        .label_style(("sans-serif", 12).into_font().color(&MUTED))
        .x_desc("step")
        .y_desc("loss")
        .draw()
        .map_err(plot_err)?;

    chart
        .draw_series(LineSeries::new(train.iter().copied(), TRAIN_RAW))
        .map_err(plot_err)?;
    chart
        .draw_series(LineSeries::new(train_ema.iter().copied(), TRAIN_EMA))
        .map_err(plot_err)?;
    if !valid.is_empty() {
        chart
            .draw_series(LineSeries::new(valid.iter().copied(), VALID))
            .map_err(plot_err)?;
        chart
            .draw_series(
                valid
                    .iter()
                    .map(|point| Circle::new(*point, 4, VALID.filled())),
            )
            .map_err(plot_err)?;
    }

    draw_legend(&root)?;
    draw_summary(&root, train, valid)?;
    root.present().map_err(plot_err)?;
    Ok(())
}

fn bounds(train: &[(f64, f64)], valid: &[(f64, f64)]) -> ((f64, f64), (f64, f64)) {
    let mut x_max = 1.0f64;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for &(x, y) in train.iter().chain(valid.iter()) {
        x_max = x_max.max(x);
        y_min = y_min.min(y);
        y_max = y_max.max(y);
    }
    let y_pad = ((y_max - y_min) * 0.08).max(0.25);
    ((0.0, x_max), ((y_min - y_pad).max(0.0), y_max + y_pad))
}

fn draw_legend(root: &DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>) -> AppResult {
    for (label, x, color) in [
        ("train raw", 690, TRAIN_RAW),
        ("train ema", 772, TRAIN_EMA),
        ("valid", 854, VALID),
    ] {
        root.draw(&Text::new(
            label,
            (x, 24),
            ("sans-serif", 12).into_font().color(&color),
        ))
        .map_err(plot_err)?;
    }
    Ok(())
}

fn draw_summary(
    root: &DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    train: &[(f64, f64)],
    valid: &[(f64, f64)],
) -> AppResult {
    let min_train = train
        .iter()
        .copied()
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap_or((0.0, 0.0));
    let final_train = train.last().copied().unwrap_or((0.0, 0.0));
    let summary = if let Some((step, loss)) = valid.last().copied() {
        format!(
            "min train {:.4} at step {:.0} | final train {:.4} | valid {:.4} @ step {:.0}",
            min_train.1, min_train.0, final_train.1, loss, step
        )
    } else {
        format!(
            "min train {:.4} at step {:.0} | final train {:.4}",
            min_train.1, min_train.0, final_train.1
        )
    };

    root.draw(&Text::new(
        summary,
        (70, SIZE.1 as i32 - 14),
        ("sans-serif", 11).into_font().color(&MUTED),
    ))
    .map_err(plot_err)?;
    Ok(())
}

fn ema(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut last = None;
    points
        .iter()
        .map(|&(step, loss)| {
            let value = last.map_or(loss, |prev| prev + 0.08 * (loss - prev));
            last = Some(value);
            (step, value)
        })
        .collect()
}

fn read_series(path: &Path) -> AppResult<Vec<f64>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    fs::read_to_string(path)?
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            line.trim().parse::<f64>().map_err(|err| {
                format!(
                    "failed to parse {} line {} as f64: {err}",
                    path.display(),
                    index + 1
                )
                .into()
            })
        })
        .collect()
}

fn read_log_interval(run_output: &RunOutput) -> AppResult<usize> {
    let path = run_output.path("run_info.txt");
    if !path.exists() {
        return Ok(1);
    }
    for line in fs::read_to_string(&path)?.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key == "log_interval" {
            return value.trim().parse::<usize>().map_err(|err| {
                format!("failed to parse log_interval in {}: {err}", path.display()).into()
            });
        }
    }
    Ok(1)
}

fn plot_err<E: std::fmt::Debug>(err: E) -> Box<dyn std::error::Error> {
    format!("{err:?}").into()
}

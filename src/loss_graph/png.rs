use std::fs;
use std::io;
use std::path::Path;

use plotters::prelude::*;

use super::LossPoint;
use crate::AppResult;

const WIDTH: u32 = 900;
const HEIGHT: u32 = 420;
const LEFT: i32 = 58;
const RIGHT: i32 = 18;
const TOP: i32 = 26;
const BOTTOM: i32 = 46;

const BACKGROUND: RGBColor = RGBColor(16, 17, 20);
const AXIS: RGBColor = RGBColor(113, 113, 122);
const TEXT: RGBColor = RGBColor(244, 244, 245);
const MUTED: RGBColor = RGBColor(161, 161, 170);
const RAW: RGBColor = RGBColor(249, 115, 22);
const EMA: RGBColor = RGBColor(56, 189, 248);

pub fn render(path: &Path, points: &[LossPoint]) -> AppResult {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let bounds = Bounds::from_points(points);
    let area = BitMapBackend::new(path, (WIDTH, HEIGHT)).into_drawing_area();
    area.fill(&BACKGROUND).map_err(plot_error)?;

    area.draw(&Text::new(
        format!("training loss {}", summary(points)),
        (LEFT, 18),
        ("monospace", 13).into_font().color(&TEXT),
    ))
    .map_err(plot_error)?;
    area.draw(&PathElement::new(
        vec![(LEFT, TOP), (LEFT, HEIGHT as i32 - BOTTOM)],
        AXIS.stroke_width(1),
    ))
    .map_err(plot_error)?;
    area.draw(&PathElement::new(
        vec![
            (LEFT, HEIGHT as i32 - BOTTOM),
            (WIDTH as i32 - RIGHT, HEIGHT as i32 - BOTTOM),
        ],
        AXIS.stroke_width(1),
    ))
    .map_err(plot_error)?;

    draw_label(&area, format!("{:.4}", bounds.max_loss), (8, TOP + 4))?;
    draw_label(
        &area,
        format!("{:.4}", bounds.min_loss),
        (8, HEIGHT as i32 - BOTTOM),
    )?;
    draw_label(&area, format!("step {}", bounds.min_step), (LEFT, 402))?;
    draw_label(
        &area,
        format!("step {}", bounds.max_step),
        (WIDTH as i32 - 112, 402),
    )?;

    area.draw(&PathElement::new(
        line(points, &bounds, |point| point.loss),
        RAW.stroke_width(2),
    ))
    .map_err(plot_error)?;
    area.draw(&PathElement::new(
        line(points, &bounds, |point| point.ema),
        EMA.stroke_width(2),
    ))
    .map_err(plot_error)?;

    area.draw(&Text::new(
        "raw",
        (680, 18),
        ("monospace", 12).into_font().color(&RAW),
    ))
    .map_err(plot_error)?;
    area.draw(&Text::new(
        "ema",
        (730, 18),
        ("monospace", 12).into_font().color(&EMA),
    ))
    .map_err(plot_error)?;
    area.present().map_err(plot_error)?;
    Ok(())
}

fn draw_label(
    area: &DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    label: String,
    pos: (i32, i32),
) -> AppResult {
    area.draw(&Text::new(
        label,
        pos,
        ("monospace", 11).into_font().color(&MUTED),
    ))
    .map_err(plot_error)?;
    Ok(())
}

struct Bounds {
    min_step: usize,
    max_step: usize,
    min_loss: f32,
    max_loss: f32,
}

impl Bounds {
    fn from_points(points: &[LossPoint]) -> Self {
        let min_step = points.first().map(|point| point.step).unwrap_or(0);
        let max_step = points
            .last()
            .map(|point| point.step)
            .unwrap_or(min_step + 1);
        let mut min_loss = f32::INFINITY;
        let mut max_loss = f32::NEG_INFINITY;
        for point in points {
            min_loss = min_loss.min(point.loss).min(point.ema);
            max_loss = max_loss.max(point.loss).max(point.ema);
        }
        if min_loss == max_loss {
            min_loss -= 1.0;
            max_loss += 1.0;
        }
        Self {
            min_step,
            max_step,
            min_loss,
            max_loss,
        }
    }
}

fn line(points: &[LossPoint], bounds: &Bounds, value: fn(&LossPoint) -> f32) -> Vec<(i32, i32)> {
    points
        .iter()
        .map(|point| {
            let x = scale(
                point.step as f32,
                bounds.min_step as f32,
                bounds.max_step as f32,
            );
            let y = 1.0 - scale(value(point), bounds.min_loss, bounds.max_loss);
            (
                LEFT + (x * plot_width()) as i32,
                TOP + (y * plot_height()) as i32,
            )
        })
        .collect()
}

fn scale(value: f32, min: f32, max: f32) -> f32 {
    if max > min {
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn summary(points: &[LossPoint]) -> String {
    if points.len() < 2 {
        return String::new();
    }
    let mut max_abs_delta = 0.0_f32;
    let mut sum_abs_delta = 0.0_f32;
    for pair in points.windows(2) {
        let delta = (pair[1].loss - pair[0].loss).abs();
        max_abs_delta = max_abs_delta.max(delta);
        sum_abs_delta += delta;
    }
    let mean_abs_delta = sum_abs_delta / (points.len() - 1) as f32;
    format!("mean_abs_delta={mean_abs_delta:.4} max_abs_delta={max_abs_delta:.4}")
}

fn plot_width() -> f32 {
    (WIDTH as i32 - LEFT - RIGHT) as f32
}

fn plot_height() -> f32 {
    (HEIGHT as i32 - TOP - BOTTOM) as f32
}

fn plot_error(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}

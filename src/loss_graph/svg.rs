use super::LossPoint;

const WIDTH: f32 = 900.0;
const HEIGHT: f32 = 420.0;
const LEFT: f32 = 58.0;
const RIGHT: f32 = 18.0;
const TOP: f32 = 26.0;
const BOTTOM: f32 = 46.0;
const PLOT_W: f32 = WIDTH - LEFT - RIGHT;
const PLOT_H: f32 = HEIGHT - TOP - BOTTOM;

pub fn render(points: &[LossPoint]) -> String {
    let bounds = Bounds::from_points(points);
    let raw = polyline(points, &bounds, |point| point.loss);
    let ema = polyline(points, &bounds, |point| point.ema);
    let summary = summary(points);
    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH:.0} {HEIGHT:.0}">
<rect width="100%" height="100%" fill="#101114"/>
<text x="{LEFT}" y="18" fill="#f4f4f5" font-family="monospace" font-size="13">training loss {summary}</text>
<line x1="{LEFT}" y1="{top}" x2="{LEFT}" y2="{bottom}" stroke="#71717a" stroke-width="1"/>
<line x1="{LEFT}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#71717a" stroke-width="1"/>
<text x="8" y="{top_label}" fill="#a1a1aa" font-family="monospace" font-size="11">{max_loss:.4}</text>
<text x="8" y="{bottom_label}" fill="#a1a1aa" font-family="monospace" font-size="11">{min_loss:.4}</text>
<text x="{LEFT}" y="402" fill="#a1a1aa" font-family="monospace" font-size="11">step {min_step}</text>
<text x="{right_label}" y="402" fill="#a1a1aa" font-family="monospace" font-size="11">step {max_step}</text>
<polyline fill="none" stroke="#f97316" stroke-width="2" points="{raw}"/>
<polyline fill="none" stroke="#38bdf8" stroke-width="2" points="{ema}"/>
<text x="680" y="18" fill="#f97316" font-family="monospace" font-size="12">raw</text>
<text x="730" y="18" fill="#38bdf8" font-family="monospace" font-size="12">ema</text>
</svg>
"##,
        top = TOP,
        bottom = HEIGHT - BOTTOM,
        right = WIDTH - RIGHT,
        top_label = TOP + 4.0,
        bottom_label = HEIGHT - BOTTOM,
        right_label = WIDTH - 112.0,
        min_loss = bounds.min_loss,
        max_loss = bounds.max_loss,
        min_step = bounds.min_step,
        max_step = bounds.max_step,
    )
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

fn polyline(points: &[LossPoint], bounds: &Bounds, value: fn(&LossPoint) -> f32) -> String {
    points
        .iter()
        .map(|point| {
            let x = scale(
                point.step as f32,
                bounds.min_step as f32,
                bounds.max_step as f32,
            );
            let y = 1.0 - scale(value(point), bounds.min_loss, bounds.max_loss);
            format!("{:.2},{:.2}", LEFT + x * PLOT_W, TOP + y * PLOT_H)
        })
        .collect::<Vec<_>>()
        .join(" ")
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

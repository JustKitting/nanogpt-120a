use gpt2_nvfp4::Gpt2Rng;

pub(super) fn sample_top_k(
    tokens: &[u32],
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    rng: &mut Gpt2Rng,
) -> u32 {
    let temperature = positive_finite_or(temperature, 1.0);
    let top_p = positive_finite_or(top_p, 1.0).clamp(0.0, 1.0);
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |max, value| max.max(value));
    let mut weights = Vec::with_capacity(logits.len());
    let mut total = 0.0_f64;

    for &logit in logits {
        let weight = ((logit - max_logit) / temperature).exp() as f64;
        let weight = if weight.is_finite() { weight } else { 0.0 };
        weights.push(weight);
        total += weight;
    }

    if total <= 0.0 || !total.is_finite() {
        return tokens[0];
    }

    let sample_total = nucleus_total(&weights, total, top_p);
    let mut draw = uniform01(rng) * sample_total;
    for (&token, weight) in tokens.iter().zip(weights) {
        if draw <= weight {
            return token;
        }
        draw -= weight;
    }
    tokens[0]
}

fn positive_finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

fn nucleus_total(weights: &[f64], total: f64, top_p: f32) -> f64 {
    if top_p >= 1.0 {
        return total;
    }

    let cutoff = total * top_p as f64;
    let mut selected = 0.0_f64;
    for &weight in weights {
        selected += weight;
        if selected >= cutoff {
            return selected.max(weight);
        }
    }
    selected.max(weights.first().copied().unwrap_or(total))
}

fn uniform01(rng: &mut Gpt2Rng) -> f64 {
    (rng.next_u32() as f64 + 0.5) / (u32::MAX as f64 + 1.0)
}

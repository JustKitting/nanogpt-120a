pub fn f64_6(value: f64) -> String {
    format!("{value:.6}")
}

pub fn optional_f64_6(value: Option<f64>) -> String {
    value.map(f64_6).unwrap_or_default()
}

pub fn optional_f64_6_or_nan(value: Option<f64>) -> String {
    value.map(f64_6).unwrap_or_else(|| "NaN".to_string())
}

pub fn optional_usize(value: Option<usize>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

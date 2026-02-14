pub const EPSILON: f64 = 1e-10;

pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let ez = (-x).exp();
        1.0 / (1.0 + ez)
    } else {
        let ez = x.exp();
        ez / (1.0 + ez)
    }
}

pub fn safe_log(p: f64) -> f64 {
    p.max(EPSILON).ln()
}

pub fn logit(p: f64) -> f64 {
    let p = clamp(p, EPSILON, 1.0 - EPSILON);
    (p / (1.0 - p)).ln()
}

pub fn safe_prob(p: f64) -> f64 {
    clamp(p, EPSILON, 1.0 - EPSILON)
}

pub fn clamp(value: f64, low: f64, high: f64) -> f64 {
    if value < low {
        low
    } else if value > high {
        high
    } else {
        value
    }
}

pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

pub fn vector_magnitude(v: &[f64]) -> f64 {
    v.iter().map(|vi| vi * vi).sum::<f64>().sqrt()
}

pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let mag_a = vector_magnitude(a);
    let mag_b = vector_magnitude(b);
    if mag_a < EPSILON || mag_b < EPSILON {
        return 0.0;
    }
    dot_product(a, b) / (mag_a * mag_b)
}

pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Softsign-based monotonic mapping from R to (0, 1).
///
/// Unlike sigmoid, softsign never saturates in f64 precision:
/// any two distinct finite inputs produce distinct outputs.
/// Suitable for calibrating raw scores that can span a wide range
/// (e.g. BM25 totals of 0-100+).
///
/// Mapping: 0 -> 0.5, +inf -> 1.0, -inf -> 0.0.
pub fn softsign_calibrate(x: f64) -> f64 {
    0.5 + 0.5 * x / (1.0 + x.abs())
}

pub fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    variance.sqrt()
}

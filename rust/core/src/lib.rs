//! Rolling z-score anomaly flags for power plant telemetry.

pub fn rolling_zscore_flags(series: &[f64], window: usize, threshold: f64) -> Vec<f64> {
    let n = series.len();
    let w = window.max(2);
    let mut out = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(w - 1);
        let slice = &series[start..=i];
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / slice.len() as f64;
        let std = var.sqrt().max(1e-12);
        let z = (series[i] - mean) / std;
        if z.abs() > threshold {
            out[i] = 1.0;
        }
    }
    out
}

use finding_the_needles_anomaly_detection_in_power_plant_operations_core::rolling_zscore_flags;

fn main() {
    let s: Vec<f64> = (0..5000).map(|i| (i as f64 * 0.01).sin() + 100.0).collect();
    for _ in 0..2000 {
        let _ = rolling_zscore_flags(&s, 24, 3.0);
    }
}

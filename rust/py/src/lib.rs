use finding_the_needles_anomaly_detection_in_power_plant_operations_core::rolling_zscore_flags;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
fn rolling_zscore_flags_py<'py>(py: Python<'py>, series: PyReadonlyArray1<f64>, window: usize, threshold: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(rolling_zscore_flags(series.as_slice()?, window, threshold).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (series, window, threshold, iterations=500))]
fn bench_kernel_py(series: PyReadonlyArray1<f64>, window: usize, threshold: f64, iterations: usize) -> PyResult<f64> {
    let series_buf = series.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = rolling_zscore_flags(&series_buf, window, threshold);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn finding_the_needles_anomaly_detection_in_power_plant_operations_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rolling_zscore_flags_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}

import numpy as np

from power_plant_anomaly.data.loaders import synthetic_production_series


def test_synthetic_series_length():
    ts = synthetic_production_series(n_years=20)
    assert len(ts) == 20
    assert np.isfinite(ts.values).all()

import pytest
import numpy as np
from src.data.preprocessing import preprocess_pipeline

def test_preprocess_pipeline():
    # Mock messy raw data
    wl_raw = np.linspace(0.5, 5.0, 1500)
    flux_raw = np.ones(1500) + np.random.normal(0, 0.05, 1500)
    
    # Introduce anomalies
    flux_raw[10] = 50.0  # Cosmic ray spike
    
    target_grid = np.linspace(0.5, 5.0, 1000)
    wl_clean, flux_clean = preprocess_pipeline(wl_raw, flux_raw, target_wavelength_grid=target_grid)
    
    # Preprocessor natively forces extraction into 1000 dimensional representation
    assert len(wl_clean) == 1000, "Clean wavelength array must be standardized to length 1000"
    assert len(flux_clean) == 1000, "Clean flux array must be standardized to length 1000"
    
    # Assert outlier capping (basic check)
    assert np.max(flux_clean) <= 1.0, "Flux normalization failed to bound outliers completely within [0,1]"

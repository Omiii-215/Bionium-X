import pytest
import numpy as np
from bioniumx.molecules import BIOSIGNATURE_MOLECULES, get_template, get_templates_parallel, clear_template_cache, get_cache_size, compute_disequilibrium

def test_catalog_contains_expected():
    assert "H2O" in BIOSIGNATURE_MOLECULES
    assert "CH4" in BIOSIGNATURE_MOLECULES

def test_get_template():
    wl, depth = get_template("H2O")
    assert len(wl) == len(depth)
    assert wl.min() >= BIOSIGNATURE_MOLECULES["H2O"]["wavelength_range"][0]

def test_get_template_with_custom_conditions():
    """Template generation with custom T and P."""
    # Ensure we start from a clean cache to make the test deterministic
    clear_template_cache()

    wl1, depth1 = get_template("H2O", resolving_power=100, T_gas=1000.0, pressure=0.1)
    cache_size_after_first = get_cache_size()

    # Force recompute for the second call and verify the cache is not modified.
    wl2, depth2 = get_template("H2O", resolving_power=100, T_gas=800.0, pressure=0.1, use_cache=False)

    assert np.array_equal(wl1, wl2)
    assert len(depth2) == len(wl2)
    assert get_cache_size() == cache_size_after_first

def test_template_caching():
    """Caching: second call with same params is faster."""
    import time
    start = time.time()
    wl1, depth1 = get_template("CO2", resolving_power=100, use_cache=True)
    time1 = time.time() - start
    
    start = time.time()
    wl2, depth2 = get_template("CO2", resolving_power=100, use_cache=True)
    time2 = time.time() - start
    
    assert (wl1 == wl2).all()
    assert (depth1 == depth2).all()

def test_bypass_cache():
    """use_cache=False forces recomputation."""
    wl, depth = get_template("H2O", resolving_power=100, use_cache=False)
    assert len(wl) == len(depth)

def test_cache_size_and_clear():
    """Test cache size and clearing."""
    size_before = get_cache_size()
    get_template("NH3", resolving_power=100)
    size_after = get_cache_size()
    assert size_after >= size_before
    clear_template_cache()
    assert get_cache_size() == 0

def test_compute_disequilibrium():
    # Provide highly significant Earth-like signatures
    sigs = {"O2": 5.0, "CH4": 4.0, "H2O": 3.0}
    result = compute_disequilibrium(sigs)
    assert ("O2", "CH4") in result.detected_pairs
    assert result.is_significant()

def test_compute_disequilibrium_insignificant():
    sigs = {"O2": 1.0, "CH4": 1.0}
    result = compute_disequilibrium(sigs)
    assert not result.is_significant()

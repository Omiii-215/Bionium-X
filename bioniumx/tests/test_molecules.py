import pytest
from bioniumx.molecules import BIOSIGNATURE_MOLECULES, get_template, get_templates_parallel, compute_disequilibrium

def test_catalog_contains_expected():
    assert "H2O" in BIOSIGNATURE_MOLECULES
    assert "CH4" in BIOSIGNATURE_MOLECULES

def test_get_template():
    wl, depth = get_template("H2O")
    assert len(wl) == len(depth)
    assert wl.min() >= BIOSIGNATURE_MOLECULES["H2O"]["wavelength_range"][0]

def test_get_templates_parallel():
    """Test parallel template fetching for multiple molecules."""
    molecules = ["H2O", "CO2"]
    templates = get_templates_parallel(molecules, max_workers=2)
    
    # Verify all molecules were fetched
    assert len(templates) == len(molecules)
    assert all(mol in templates for mol in molecules)
    
    # Verify each template has correct structure
    for mol in molecules:
        wl, depth = templates[mol]
        assert len(wl) == len(depth)
        assert wl.min() >= BIOSIGNATURE_MOLECULES[mol]["wavelength_range"][0]
        assert wl.max() <= BIOSIGNATURE_MOLECULES[mol]["wavelength_range"][1]

def test_get_templates_parallel_single_molecule():
    """Test parallel fetch with a single molecule."""
    templates = get_templates_parallel(["H2O"], max_workers=1)
    assert len(templates) == 1
    assert "H2O" in templates

def test_get_templates_parallel_empty():
    """Test parallel fetch with empty molecule list."""
    templates = get_templates_parallel([])
    assert len(templates) == 0

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

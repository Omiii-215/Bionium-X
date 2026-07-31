import numpy as np
from bioniumx.physics import equilibrium_temperature, habitable_zone_bounds, habitability_score, mean_molecular_weight, scale_height

def test_equilibrium_temperature():
    # Earth analog: T_star=5778, R_star=1.0, a=1.0
    T_eq = equilibrium_temperature(5778, 1.0, 1.0)
    assert 250 < T_eq < 260  # ~255K

def test_habitable_zone_bounds():
    inner, outer = habitable_zone_bounds(5778, 1.0)
    assert inner < 1.0
    assert outer > 1.0

def test_habitability_score():
    score = habitability_score(255, 1.0)
    assert score > 0.9  # Earth is highly habitable

def test_habitability_score_rejects_non_physical_catalog_values():
    assert habitability_score(255, 0.0) == 0.0
    assert habitability_score(255, -1.0) == 0.0
    assert habitability_score(255, np.nan) == 0.0
    assert habitability_score(0.0, 1.0) == 0.0

def test_mean_molecular_weight():
    mu = mean_molecular_weight({"H2": 0.85, "He": 0.15})
    assert 2.0 < mu < 3.0


def test_mean_molecular_weight_normalizes_fractions():
    mu_unscaled = mean_molecular_weight({"H2": 85, "He": 15})
    mu_scaled = mean_molecular_weight({"H2": 0.85, "He": 0.15})
    assert mu_unscaled == mu_scaled


def test_mean_molecular_weight_rejects_unknown_species():
    import pytest

    with pytest.raises(ValueError, match="Unknown molecule weight"):
        mean_molecular_weight({"H2": 1.0, "Xe": 0.0})


def test_mean_molecular_weight_rejects_empty_abundances():
    import pytest

    with pytest.raises(ValueError, match="at least one positive mole fraction"):
        mean_molecular_weight({})


def test_atmosphere_public_api_docstrings():
    import inspect
    from bioniumx.physics import atmosphere

    for name in ("mean_molecular_weight", "scale_height"):
        obj = getattr(atmosphere, name)
        doc = inspect.getdoc(obj)
        assert doc is not None
        assert "Parameters" in doc
        assert "Returns" in doc


def test_scale_height():
    H = scale_height(255, 28.97, 9.81)
    assert 7000 < H < 9000  # ~8km for Earth

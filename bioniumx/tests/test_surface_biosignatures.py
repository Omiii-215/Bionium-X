import numpy as np

from bioniumx.surface_biosignatures import (
    vre_reflectance,
    rust_slope_continuum,
    microbial_biopigment_template,
    classify_bio_vs_rust,
    anoxygenic_photosynthesis_template,
)


def test_vre_reflectance_edge_increase():
    wl = np.linspace(0.55, 0.85, 301)
    r = vre_reflectance(wl, base_albedo=0.2, amplitude=0.25, edge_center_um=0.7, edge_width_um=0.02)

    blue = r[(wl >= 0.64) & (wl <= 0.68)].mean()
    red = r[(wl >= 0.72) & (wl <= 0.76)].mean()
    assert red > blue


def test_rust_slope_is_classified_as_not_bio():
    wl = np.linspace(0.55, 0.85, 301)
    r = rust_slope_continuum(wl, albedo0=0.1, slope=0.35, curvature=0.0)
    res = classify_bio_vs_rust(wl, r)
    assert res["is_bio"] is False


def test_bio_vre_is_classified_as_bio():
    wl = np.linspace(0.55, 0.85, 301)
    r = vre_reflectance(wl, base_albedo=0.15, amplitude=0.25, edge_center_um=0.7, edge_width_um=0.02, redward_slope=0.02)
    # Add a modest anoxygenic component to mimic combined pigments
    r2 = anoxygenic_photosynthesis_template(wl, base_albedo=0.1, amplitude=0.08)
    r_mix = 0.6 * r + 0.4 * r2
    res = classify_bio_vs_rust(wl, r_mix)
    assert res["is_bio"] is True


def test_microbial_pigment_template_produces_reasonable_range():
    wl = np.linspace(0.4, 1.0, 301)
    r = microbial_biopigment_template(wl, pigment="carotenoids")
    assert np.all(np.isfinite(r))
    assert np.min(r) >= 0.0
    assert np.max(r) <= 1.0


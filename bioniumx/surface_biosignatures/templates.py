"""bioniumx.surface_biosignatures.templates

Spectral template functions for surface/reflectance biosignatures.

These models are *parametric approximations* intended for pipeline testing and
library integration. They provide controlled, physically-motivated shapes around
key features (VRE near ~0.7 μm) and pigment absorption bands.

All functions accept wavelengths in microns and return a dimensionless
reflectance/albedo spectrum.
"""

from __future__ import annotations

import numpy as np


def _as_microns(wavelength):
    wl = np.asarray(wavelength, dtype=float)
    return wl


def vre_reflectance(
    wavelength,
    base_albedo: float = 0.2,
    amplitude: float = 0.2,
    edge_center_um: float = 0.70,
    edge_width_um: float = 0.03,
    saturation: float = 1.0,
    redward_slope: float = 0.0,
):
    """Vegetation Red Edge (VRE) reflectance model.

    Uses a smooth logistic-like rise around ~0.70 μm.

    Parameters
    ----------
    wavelength : array-like
        Wavelength in microns.
    base_albedo : float
        Baseline reflectance level.
    amplitude : float
        Peak increase across the edge.
    edge_center_um : float
        Edge center wavelength in microns (default ~0.70 μm).
    edge_width_um : float
        Controls steepness of edge.
    saturation : float
        Upper cap on reflectance increase.
    redward_slope : float
        Optional linear slope for λ > edge_center.

    Returns
    -------
    reflectance : np.ndarray
        Model reflectance.
    """
    wl = _as_microns(wavelength)

    # Logistic rise: 0 (blueward) -> 1 (redward)
    x = (wl - edge_center_um) / max(edge_width_um, 1e-6)
    rise = 1.0 / (1.0 + np.exp(-x))

    # Allow gentle saturation and optional redward slope
    inc = amplitude * (rise ** 1.0)
    inc = np.minimum(inc, saturation * amplitude)

    r = base_albedo + inc

    if redward_slope != 0.0:
        mask = wl > edge_center_um
        # Linear term anchored at edge_center_um
        r[mask] = r[mask] + redward_slope * (wl[mask] - edge_center_um)

    return np.clip(r, 0.0, 1.0)


def anoxygenic_photosynthesis_template(
    wavelength,
    base_albedo: float = 0.15,
    amplitude: float = 0.15,
    # absorption/reflectance features are rough parametric bands
    feature1_center_um: float = 0.62,  # near-IR/visible chlorophyll-like
    feature1_width_um: float = 0.06,
    feature2_center_um: float = 0.88,  # bacteriochlorophyll-like bump/absorption
    feature2_width_um: float = 0.08,
    reflectance_bump: float = 0.12,
):
    """Template-like anoxygenic photosynthesis reflectance.

    Includes broad absorption trough(s) in the visible/NIR and a modest
    reflectance bump into the NIR. This is designed to reduce false positives
    when combined with VRE-like edge detection.

    Returns a dimensionless reflectance spectrum.
    """
    wl = _as_microns(wavelength)

    # Two broad absorption-like components (reduce reflectance)
    absorb1 = np.exp(-0.5 * ((wl - feature1_center_um) / max(feature1_width_um, 1e-6)) ** 2)
    absorb2 = np.exp(-0.5 * ((wl - feature2_center_um) / max(feature2_width_um, 1e-6)) ** 2)

    # Convert to reflectance change: absorption reduces by amplitude fraction
    r = base_albedo + reflectance_bump * (wl > feature2_center_um) * (wl - feature2_center_um) / (wl.max() - feature2_center_um + 1e-9)

    r -= amplitude * (0.6 * absorb1 + 0.4 * absorb2)

    return np.clip(r, 0.0, 1.0)


def microbial_biopigment_template(
    wavelength,
    pigment: str = "carotenoids",
    base_albedo: float = 0.12,
    scale: float = 0.18,
):
    """Microbial pigment spectral template.

    Parameters
    ----------
    wavelength : array-like
        Wavelength in microns.
    pigment : str
        One of: 'carotenoids', 'bacteriochlorophyll_a', 'bacteriochlorophyll_b'.
    base_albedo : float
        Baseline reflectance.
    scale : float
        Global amplitude scale.

    Notes
    -----
    These templates use simplified Gaussian absorption/bump components.
    """
    wl = _as_microns(wavelength)

    p = pigment.lower()

    if p in {"carotenoids", "carotenoid"}:
        # Carotenoids: strong absorption in blue/green (~0.45-0.55 μm)
        # with redward reflectance increase.
        a_blue = np.exp(-0.5 * ((wl - 0.50) / 0.05) ** 2)
        a_green = np.exp(-0.5 * ((wl - 0.58) / 0.06) ** 2)
        bump_red = np.exp(-0.5 * ((wl - 0.75) / 0.10) ** 2)
        r = base_albedo + scale * (0.55 * bump_red - 0.9 * a_blue - 0.4 * a_green)

    elif p in {"bacteriochlorophyll_a", "bchl_a", "bacteriochlorophyll a"}:
        # BChl-a: absorption in visible with strong NIR features (~0.78-0.86 μm)
        a1 = np.exp(-0.5 * ((wl - 0.66) / 0.06) ** 2)
        a2 = np.exp(-0.5 * ((wl - 0.81) / 0.07) ** 2)
        r = base_albedo + 0.6 * scale * (1.0 - 0.8 * a1 - 0.6 * a2)

    elif p in {"bacteriochlorophyll_b", "bchl_b", "bacteriochlorophyll b"}:
        # BChl-b: similar but shifted features.
        a1 = np.exp(-0.5 * ((wl - 0.64) / 0.07) ** 2)
        a2 = np.exp(-0.5 * ((wl - 0.83) / 0.08) ** 2)
        r = base_albedo + 0.6 * scale * (1.0 - 0.7 * a1 - 0.7 * a2)

    else:
        raise ValueError(
            "Unknown pigment. Use 'carotenoids', 'bacteriochlorophyll_a', or 'bacteriochlorophyll_b'."
        )

    return np.clip(r, 0.0, 1.0)


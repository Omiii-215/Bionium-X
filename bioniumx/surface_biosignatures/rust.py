"""bioniumx.surface_biosignatures.rust

Abiotic iron-oxide (“rust”) continuum model.

The main failure mode for VRE/biosignature detection in directly imaged spectra is
that abiotic spectra can mimic a smooth redward slope/curvature.

Here we provide a simple continuum/slope model with optional curvature.
"""

from __future__ import annotations

import numpy as np


def rust_slope_continuum(
    wavelength,
    albedo0: float = 0.15,
    slope: float = 0.25,
    curvature: float = 0.0,
    pivot_um: float = 0.55,
):
    """Iron-oxide rust continuum as a smooth slope/curvature.

    Parameters
    ----------
    wavelength : array-like
        Wavelength in microns.
    albedo0 : float
        Continuum level at pivot_um.
    slope : float
        Linear slope coefficient.
    curvature : float
        Quadratic curvature coefficient.
    pivot_um : float
        Pivot wavelength in microns.

    Returns
    -------
    reflectance : np.ndarray
    """
    wl = np.asarray(wavelength, dtype=float)
    dx = wl - pivot_um
    r = albedo0 + slope * dx + curvature * dx * dx
    return np.clip(r, 0.0, 1.0)


"""bioniumx.surface_biosignatures.discrimination

Logic to distinguish biological pigments from abiotic iron-oxide rust slopes.

This module provides an interpretable, lightweight decision mechanism based on
spectral *feature shapes* rather than absolute flux.

Current strategy
----------------
- Fit/remove a smooth baseline (rust-like continuum) using a robust low-order
  polynomial.
- Measure VRE-like edge statistics: (redward mean - blueward mean) and the
  local gradient around ~0.70 μm.
- Compare these to what would be expected from a pure rust slope.

The API returns both a boolean classification and diagnostic metrics.
"""

from __future__ import annotations

import numpy as np


def _robust_polyfit(x, y, deg: int):
    # Lightweight robust fit: iterative sigma clipping on residuals
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < deg + 1:
        raise ValueError("Not enough finite points to fit continuum.")

    w = np.ones_like(y)
    for _ in range(3):
        coeffs = np.polyfit(x, y, deg, w=w)
        model = np.polyval(coeffs, x)
        resid = y - model
        sigma = np.std(resid)
        if sigma <= 0:
            break
        keep = np.abs(resid) <= 2.5 * sigma
        w = keep.astype(float)
    return coeffs


def classify_bio_vs_rust(
    wavelength,
    reflectance,
    vre_edge_center_um: float = 0.70,
    vre_blue_window_um: tuple[float, float] = (0.64, 0.68),
    vre_red_window_um: tuple[float, float] = (0.72, 0.76),
    continuum_poly_deg: int = 2,
    threshold_edge_contrast: float = 0.04,
    threshold_local_gradient: float = 0.15,
):
    """Classify whether a spectrum is more consistent with biology than rust.

    Returns
    -------
    result : dict
        Keys:
        - 'is_bio' (bool)
        - 'edge_contrast' (float) : mean(red) - mean(blue) after continuum removal
        - 'local_gradient' (float) : dR/dλ around the edge center
        - 'continuum_coeffs' (np.ndarray)
    """
    wl = np.asarray(wavelength, dtype=float)
    r = np.asarray(reflectance, dtype=float)
    if wl.shape != r.shape:
        raise ValueError("wavelength and reflectance must have the same shape")

    # Continuum removal
    coeffs = _robust_polyfit(wl, r, deg=continuum_poly_deg)
    cont = np.polyval(coeffs, wl)

    # Avoid division by ~0; use relative residuals around 1
    r_norm = r - cont

    blue_mask = (wl >= vre_blue_window_um[0]) & (wl <= vre_blue_window_um[1])
    red_mask = (wl >= vre_red_window_um[0]) & (wl <= vre_red_window_um[1])

    if blue_mask.sum() < 3 or red_mask.sum() < 3:
        raise ValueError("Not enough wavelength coverage for VRE windows")

    edge_contrast = float(np.mean(r_norm[red_mask]) - np.mean(r_norm[blue_mask]))

    # Local gradient estimate around edge center
    eps = 1e-9
    mask_local = (wl >= vre_edge_center_um - 0.03) & (wl <= vre_edge_center_um + 0.03)
    if mask_local.sum() < 3:
        local_gradient = 0.0
    else:
        xloc = wl[mask_local]
        yloc = r_norm[mask_local]
        # Simple slope from linear fit
        g, _ = np.polyfit(xloc, yloc, 1)
        local_gradient = float(g)

    is_bio = (edge_contrast >= threshold_edge_contrast) and (local_gradient >= threshold_local_gradient)

    return {
        "is_bio": bool(is_bio),
        "edge_contrast": edge_contrast,
        "local_gradient": local_gradient,
        "continuum_coeffs": coeffs,
    }


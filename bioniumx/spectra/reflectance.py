"""bioniumx.spectra.reflectance

Lightweight data structures for directly imaged planet *surface/reflectance*
spectra.

The core library currently supports transmission and emission spectra.
This module introduces a minimal ReflectanceSpectrum container so that we can
attach surface/biosignature spectral models to directly-imaged data.
"""

from __future__ import annotations

import numpy as np

from bioniumx.core import BioniumXObject


class ReflectanceSpectrum(BioniumXObject):
    """Directly imaged planet surface reflectance (albedo) spectrum.

    Parameters
    ----------
    wavelength : array-like
        Wavelength in microns.
    reflectance : array-like
        Surface reflectance / albedo (dimensionless).
    err : array-like, optional
        1-sigma uncertainty on reflectance. If None, set to zeros.
    target_name : str, optional
        Planet identifier.
    instrument : str, optional
        Instrument name.
    resolution : float, optional
        Resolving power R = λ/Δλ.
    meta : dict, optional
        Additional metadata.
    """

    _required_attrs = ["wavelength", "reflectance"]
    _metadata_attrs = ["target_name", "instrument", "resolution"]

    def __init__(
        self,
        wavelength,
        reflectance,
        err=None,
        target_name: str = "Unknown",
        instrument: str = "Unknown",
        resolution: float | None = None,
        meta: dict | None = None,
        gti: list | None = None,
    ):
        self.wavelength = np.asarray(wavelength, dtype=float)
        self.reflectance = np.asarray(reflectance, dtype=float)

        if err is not None:
            self.err = np.asarray(err, dtype=float)
        else:
            self.err = np.zeros_like(self.reflectance)

        self.meta = {
            "target_name": target_name,
            "instrument": instrument,
            "resolution": resolution,
        }
        if meta:
            self.meta.update(meta)

        self.gti = gti
        self._validate()

    def apply_wavelength_mask(self, wl_min: float, wl_max: float):
        mask = (self.wavelength >= wl_min) & (self.wavelength <= wl_max)
        return ReflectanceSpectrum(
            wavelength=self.wavelength[mask],
            reflectance=self.reflectance[mask],
            err=self.err[mask],
            target_name=self.meta.get("target_name", "Unknown"),
            instrument=self.meta.get("instrument", "Unknown"),
            resolution=self.meta.get("resolution", None),
            meta={k: v for k, v in self.meta.items() if k not in {"target_name", "instrument", "resolution"}},
        )

    def rebin(self, factor: int):
        n = len(self.wavelength) // factor * factor
        wl = self.wavelength[:n].reshape(-1, factor).mean(axis=1)
        r = self.reflectance[:n].reshape(-1, factor).mean(axis=1)
        err = np.sqrt((self.err[:n].reshape(-1, factor) ** 2).sum(axis=1)) / factor
        return ReflectanceSpectrum(
            wl,
            r,
            err=err,
            target_name=self.meta.get("target_name", "Unknown"),
            instrument=self.meta.get("instrument", "Unknown"),
            resolution=self.meta.get("resolution", None),
            meta={k: v for k, v in self.meta.items() if k not in {"target_name", "instrument", "resolution"}},
        )

    def snr(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.err > 0, self.reflectance / self.err, np.inf)

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        ax.errorbar(self.wavelength, self.reflectance, yerr=self.err, fmt="o", ms=3, **kwargs)
        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel("Reflectance / Albedo")
        ax.set_title(self.meta.get("target_name", ""))
        return ax


"""
Bionium-X: Biosignature Detection in Exoplanet Atmospheres
==========================================================

A research-grade, pure-Python scientific library for detecting
biosignatures in exoplanet transmission and emission spectra.

Designed after Stingray (stingray.science) for spectral-timing astronomy:
clean API, no UI dependencies in core, fully pip-installable.
"""

__version__ = "0.2.0"
__author__ = "Bionium-X Research Collaborative"

# Data classes
from bioniumx.spectra.transmission import TransmissionSpectrum
from bioniumx.spectra.emission import EmissionSpectrum

# Preprocessing
from bioniumx.preprocessing.filters import savitzky_golay, gaussian_smooth
from bioniumx.preprocessing.normalizer import continuum_normalize
from bioniumx.preprocessing.masking import mask_wavelength_regions

# Detection
from bioniumx.detection.cross_correlation import cross_correlate_template
from bioniumx.detection.equivalent_width import equivalent_width
from bioniumx.detection.bayesian import bayes_factor

# Molecules
from bioniumx.molecules.catalog import BIOSIGNATURE_MOLECULES, get_template
from bioniumx.molecules.disequilibrium import compute_disequilibrium

# Assessment
from bioniumx.assessment import assess_false_positive 

# Physics
from bioniumx.physics.habitability import (
    equilibrium_temperature,
    habitable_zone_bounds,
    habitability_score, 
) 

# Surface biosignatures (directly imaged planets)
from bioniumx.surface_biosignatures import (
    vre_reflectance,
    anoxygenic_photosynthesis_template,
    microbial_biopigment_template,
    rust_slope_continuum,
    classify_bio_vs_rust,
)

__all__ = [
    "TransmissionSpectrum",
    "EmissionSpectrum",
    "savitzky_golay",
    "gaussian_smooth",
    "continuum_normalize",
    "mask_wavelength_regions",
    "assess_false_positive",  
    "cross_correlate_template",
    "equivalent_width",
    "bayes_factor",
    "BIOSIGNATURE_MOLECULES",
    "get_template",
    "compute_disequilibrium",
    "equilibrium_temperature",
    "habitable_zone_bounds",
    "habitability_score",
    "vre_reflectance",
    "anoxygenic_photosynthesis_template",
    "microbial_biopigment_template",
    "rust_slope_continuum",
    "classify_bio_vs_rust",
]


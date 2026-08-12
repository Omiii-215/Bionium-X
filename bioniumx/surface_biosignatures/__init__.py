"""Surface biosignatures for directly imaged planets.

This subpackage adds spectral models and discrimination logic for:
- Vegetation Red Edge (VRE)
- Anoxygenic photosynthesis biosignatures (template-like)
- Microbial biopigment reflectance templates (carotenoids, bacteriochlorophylls)
- Distinguishing biological pigments from abiotic iron-oxide (“rust”) slopes

The implementations are deliberately lightweight and designed to integrate with
Bionium-X's existing spectral container patterns.
"""

from .templates import (
    vre_reflectance,
    anoxygenic_photosynthesis_template,
    microbial_biopigment_template,
)
from .rust import rust_slope_continuum
from .discrimination import classify_bio_vs_rust

__all__ = [
    "vre_reflectance",
    "anoxygenic_photosynthesis_template",
    "microbial_biopigment_template",
    "rust_slope_continuum",
    "classify_bio_vs_rust",
]



import pytest
from src.data.generator import SpectrumGenerator
import numpy as np

def test_spectrum_generator_shapes():
    gen = SpectrumGenerator()
    present_molecules = {'O2': 0.2, 'CH4': 0.15}
    wl, flux, _ = gen.generate_spectrum(present_molecules, noise_level=0.01)
    
    assert len(wl) == 1000, "Wavelength array should be exactly 1000 bins"
    assert len(flux) == 1000, "Flux array should be exactly 1000 bins"
    assert not np.isnan(wl).any(), "Wavelength array contains NaNs"
    assert not np.isnan(flux).any(), "Flux array contains NaNs"

def test_spectrum_generator_noise():
    gen = SpectrumGenerator()
    wl, base_flux, _ = gen.generate_spectrum({}, noise_level=0.0)
    wl, noisy_flux, _ = gen.generate_spectrum({}, noise_level=0.1)
    
    # The noisy flux should have greater variance than the clean baseline
    assert np.var(noisy_flux) > np.var(base_flux), "Noise injection failed to increase variance"

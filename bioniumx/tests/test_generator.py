import numpy as np

from bioniumx.simulator.generator import SpectrumGenerator
from bioniumx.spectra import TransmissionSpectrum


class TestGenerateSpectrumUncertainty:
    """Regression tests for generate_spectrum uncertainty (issue #55)."""

    def test_uncertainty_is_non_negative(self):
        # The third return value is a 1-sigma uncertainty array and must be
        # non-negative — it previously returned the signed noise realization,
        # so about half the values were negative.
        gen = SpectrumGenerator()
        _, _, err = gen.generate_spectrum(
            {"O2": 0.1, "CH4": 0.15}, noise_level=0.02
        )
        assert np.all(err >= 0), "uncertainty array must be non-negative"

    def test_uncertainty_equals_noise_level(self):
        gen = SpectrumGenerator()
        noise_level = 0.02
        _, _, err = gen.generate_spectrum(
            {"O2": 0.1}, noise_level=noise_level
        )
        assert np.allclose(err, noise_level)

    def test_snr_is_finite_with_generated_err(self):
        # Downstream symptom of the old bug: negative err made snr() return
        # inf for about half the channels. A proper uncertainty array keeps
        # snr() finite everywhere.
        gen = SpectrumGenerator()
        wl, flux, err = gen.generate_spectrum({"O2": 0.1, "CH4": 0.15})
        spec = TransmissionSpectrum(wl, flux, err=err)
        snr = spec.snr()
        assert np.all(np.isfinite(snr))

    def test_noise_still_added_to_flux(self):
        # The fix must not stop the noise realization from being added to
        # flux; only the returned third value changes.
        gen = SpectrumGenerator()
        flux_a = gen.generate_spectrum({"O2": 0.1}, noise_level=0.0)[1]
        flux_b = gen.generate_spectrum({"O2": 0.1}, noise_level=0.05)[1]
        assert not np.allclose(flux_a, flux_b)

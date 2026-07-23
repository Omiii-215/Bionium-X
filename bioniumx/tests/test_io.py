import re

import numpy as np
import pytest

from bioniumx.core import BioniumXObject
from bioniumx.io import SUPPORTED_FORMATS, read_object, write_object
from bioniumx.spectra import TransmissionSpectrum


@pytest.fixture
def spectrum():
    wl = np.linspace(0.6, 5.3, 32)
    depth = 0.014 + 1e-4 * np.random.randn(32)
    err = 1e-5 * np.ones(32)
    return TransmissionSpectrum(wl, depth, err=err, target_name="Test b")


class TestSupportedFormats:
    """The advertised format contract must match the implementation."""

    def test_hdf5_is_supported(self):
        assert "hdf5" in SUPPORTED_FORMATS

    def test_hdf5_roundtrip(self, spectrum, tmp_path):
        path = str(tmp_path / "spec.h5")
        spectrum.write(path, fmt="hdf5")
        loaded = TransmissionSpectrum.read(path, fmt="hdf5")
        np.testing.assert_allclose(loaded.wavelength, spectrum.wavelength)
        np.testing.assert_allclose(loaded.transit_depth, spectrum.transit_depth)

    @pytest.mark.parametrize("fmt", ["fits", "ascii"])
    def test_write_unsupported_format_error_lists_supported(self, spectrum, tmp_path, fmt):
        path = str(tmp_path / f"spec.{fmt}")
        with pytest.raises(NotImplementedError) as excinfo:
            spectrum.write(path, fmt=fmt)
        msg = str(excinfo.value)
        # The message must name the rejected format and point at what IS supported.
        assert fmt in msg
        assert "hdf5" in msg

    @pytest.mark.parametrize("fmt", ["fits", "ascii"])
    def test_read_unsupported_format_error_lists_supported(self, tmp_path, fmt):
        path = str(tmp_path / f"spec.{fmt}")
        with pytest.raises(NotImplementedError) as excinfo:
            TransmissionSpectrum.read(path, fmt=fmt)
        msg = str(excinfo.value)
        assert fmt in msg
        assert "hdf5" in msg

    def test_lowlevel_write_object_rejects_unsupported(self, spectrum, tmp_path):
        with pytest.raises(NotImplementedError, match="hdf5"):
            write_object(spectrum, str(tmp_path / "spec.fits"), fmt="fits")

    def test_lowlevel_read_object_rejects_unsupported(self, tmp_path):
        with pytest.raises(NotImplementedError, match="hdf5"):
            read_object(TransmissionSpectrum, str(tmp_path / "spec.fits"), fmt="fits")


class TestDocstringHonesty:
    """Documented, copy-pasteable examples must not call an unsupported format."""

    @pytest.mark.parametrize("doc", [BioniumXObject.write.__doc__, BioniumXObject.read.__doc__])
    def test_no_runnable_unsupported_format_example(self, doc):
        # e.g. `spec.write("K2-18b.fits", fmt="fits")` used to be documented but raises.
        assert not re.search(r"""fmt=["'](fits|ascii)["']""", doc)

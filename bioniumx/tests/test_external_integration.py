import pytest
import numpy as np
import pandas as pd
import h5py
from bioniumx.datasets import (
    parse_vpl_spectrum,
    parse_multirex_spectrum,
    parse_garlic_spectrum,
    load_forward_model_spectrum,
    parse_poseidon_retrieval,
    parse_aurora_retrieval,
    parse_chimera_retrieval,
    parse_petitradtrans_retrieval,
    parse_atmos_profile,
    GCMWorkflows,
)
from bioniumx.spectra import TransmissionSpectrum, EmissionSpectrum


# ==============================================================================
# Fixtures for Forward Models
# ==============================================================================


@pytest.fixture
def vpl_file_2col(tmp_path):
    filepath = tmp_path / "vpl_2col.txt"
    with open(filepath, "w") as f:
        f.write("# VPL Forward Model Output\n")
        f.write("# Wavelength (um)  Depth\n")
        f.write("0.8   0.012\n")
        f.write("1.2   0.015\n")
        f.write("1.6   0.014\n")
    return str(filepath)


@pytest.fixture
def vpl_file_3col(tmp_path):
    filepath = tmp_path / "vpl_3col.txt"
    with open(filepath, "w") as f:
        f.write("% VPL Model with errors\n")
        f.write("0.8   0.012   0.001\n")
        f.write("1.2   0.015   0.002\n")
        f.write("1.6   0.014   0.0015\n")
    return str(filepath)


@pytest.fixture
def multirex_file(tmp_path):
    filepath = tmp_path / "multirex.csv"
    df = pd.DataFrame(
        {
            "wavelength": [1.0, 2.0, 3.0],
            "transit_depth": [0.01, 0.02, 0.015],
            "noise": [0.001, 0.002, 0.001],
        }
    )
    df.to_csv(filepath, index=False)
    return str(filepath)


@pytest.fixture
def multirex_dppm_file(tmp_path):
    filepath = tmp_path / "multirex_dppm.csv"
    df = pd.DataFrame(
        {
            "wl": [1.0, 2.0, 3.0],
            "dppm": [10000.0, 20000.0, 15000.0],
            "dppm_err": [100.0, 200.0, 150.0],
        }
    )
    df.to_csv(filepath, index=False)
    return str(filepath)


@pytest.fixture
def garlic_wavenumber_file(tmp_path):
    filepath = tmp_path / "garlic.txt"
    # Wavenumber is descending: 12500 cm-1 (0.8 um), 8333 cm-1 (1.2 um), 6250 cm-1 (1.6 um)
    with open(filepath, "w") as f:
        f.write("12500.0   0.012\n")
        f.write("8333.33   0.015\n")
        f.write("6250.0    0.014\n")
    return str(filepath)


# ==============================================================================
# Fixtures for Retrieval Codes
# ==============================================================================


@pytest.fixture
def poseidon_hdf5_file(tmp_path):
    filepath = tmp_path / "poseidon_output.h5"
    with h5py.File(filepath, "w") as f:
        f.create_dataset("wl", data=np.array([1.0, 2.0, 3.0]))
        f.create_dataset("spectrum_median", data=np.array([0.01, 0.02, 0.015]))
        f.create_dataset("spectrum_1sigma", data=np.array([0.001, 0.002, 0.0015]))
    return str(filepath)


@pytest.fixture
def aurora_hdf5_file(tmp_path):
    filepath = tmp_path / "aurora_output.h5"
    with h5py.File(filepath, "w") as f:
        f.create_dataset("wavelength", data=np.array([1.0, 2.0, 3.0]))
        f.create_dataset("median", data=np.array([0.01, 0.02, 0.015]))
        f.create_dataset("error", data=np.array([0.001, 0.002, 0.0015]))
    return str(filepath)


@pytest.fixture
def chimera_file(tmp_path):
    filepath = tmp_path / "chimera.txt"
    with open(filepath, "w") as f:
        f.write("1.0   0.010   0.001\n")
        f.write("2.0   0.020   0.002\n")
        f.write("3.0   0.015   0.001\n")
    return str(filepath)


@pytest.fixture
def petitradtrans_hdf5_file(tmp_path):
    filepath = tmp_path / "petitradtrans.h5"
    with h5py.File(filepath, "w") as f:
        f.create_dataset("wavelength", data=np.array([1.0, 2.0, 3.0]))
        f.create_dataset("transit_depth", data=np.array([0.01, 0.02, 0.015]))
        f.create_dataset("err", data=np.array([0.001, 0.002, 0.0015]))
    return str(filepath)


# ==============================================================================
# Fixtures for GCMs & Climate
# ==============================================================================


@pytest.fixture
def atmos_file(tmp_path):
    filepath = tmp_path / "atmos.out"
    with open(filepath, "w") as f:
        f.write("  ALT       PRESS       TEMP        H2O         CO2\n")
        f.write("  0.0       1.0         300.0       1.0e-4      3.5e-4\n")
        f.write("  10.0      0.5         260.0       5.0e-5      3.5e-4\n")
        f.write("  20.0      0.1         220.0       1.0e-5      3.5e-4\n")
    return str(filepath)


@pytest.fixture
def gcm_hdf5_file(tmp_path):
    filepath = tmp_path / "gcm_output.nc"
    with h5py.File(filepath, "w") as f:
        # 3 lats: -45, 0, 45
        f.create_dataset("lat", data=np.array([-45.0, 0.0, 45.0]))
        # 4 lons: 0, 90, 180, 270
        f.create_dataset("lon", data=np.array([0.0, 90.0, 180.0, 270.0]))
        # 3 pressure levels: 1.0, 0.1, 0.01
        f.create_dataset("lev", data=np.array([1.0, 0.1, 0.01]))

        # Temp shape (3, 3, 4) -> (lev, lat, lon)
        temp_data = np.zeros((3, 3, 4))
        temp_data[0, :, :] = 300.0
        temp_data[1, :, :] = 250.0
        temp_data[2, :, :] = 200.0
        f.create_dataset("T", data=temp_data)

        # H2O shape (3, 3, 4)
        h2o_data = np.ones((3, 3, 4)) * 1e-4
        f.create_dataset("H2O", data=h2o_data)

        # CO2 shape (3, 3, 4)
        co2_data = np.ones((3, 3, 4)) * 3e-4
        f.create_dataset("CO2", data=co2_data)

    return str(filepath)


# ==============================================================================
# Tests for Forward Model Ingestion
# ==============================================================================


def test_parse_vpl_2col(vpl_file_2col):
    spec = parse_vpl_spectrum(
        vpl_file_2col, spectrum_type="transmission", target_name="VPL Planet"
    )
    assert isinstance(spec, TransmissionSpectrum)
    np.testing.assert_allclose(spec.wavelength, [0.8, 1.2, 1.6])
    np.testing.assert_allclose(spec.transit_depth, [0.012, 0.015, 0.014])
    np.testing.assert_allclose(spec.err, [0.0, 0.0, 0.0])
    assert spec.meta["target_name"] == "VPL Planet"


def test_parse_vpl_3col(vpl_file_3col):
    spec = parse_vpl_spectrum(
        vpl_file_3col, spectrum_type="emission", target_name="VPL Planet"
    )
    assert isinstance(spec, EmissionSpectrum)
    np.testing.assert_allclose(spec.wavelength, [0.8, 1.2, 1.6])
    np.testing.assert_allclose(spec.flux, [0.012, 0.015, 0.014])
    np.testing.assert_allclose(spec.err, [0.001, 0.002, 0.0015])


def test_parse_multirex(multirex_file):
    spec = parse_multirex_spectrum(multirex_file, spectrum_type="transmission")
    assert isinstance(spec, TransmissionSpectrum)
    np.testing.assert_allclose(spec.wavelength, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(spec.transit_depth, [0.01, 0.02, 0.015])
    np.testing.assert_allclose(spec.err, [0.001, 0.002, 0.001])


def test_parse_multirex_dppm(multirex_dppm_file):
    spec = parse_multirex_spectrum(multirex_dppm_file, spectrum_type="transmission")
    np.testing.assert_allclose(spec.wavelength, [1.0, 2.0, 3.0])
    # dppm values should be divided by 1,000,000
    np.testing.assert_allclose(spec.transit_depth, [0.01, 0.02, 0.015])
    np.testing.assert_allclose(spec.err, [0.0001, 0.0002, 0.00015])


def test_parse_garlic_wavenumber(garlic_wavenumber_file):
    spec = parse_garlic_spectrum(garlic_wavenumber_file, spectrum_type="transmission")
    # 12500 cm-1 -> 0.8 um, 8333.33 -> 1.2 um, 6250 -> 1.6 um
    # Should be sorted in ascending wavelength order
    np.testing.assert_allclose(spec.wavelength, [0.8, 1.2, 1.6], rtol=0.01)
    np.testing.assert_allclose(spec.transit_depth, [0.012, 0.015, 0.014], rtol=0.01)


def test_load_forward_model_router(
    vpl_file_2col, multirex_file, garlic_wavenumber_file
):
    spec_v = load_forward_model_spectrum(vpl_file_2col, "vpl")
    spec_m = load_forward_model_spectrum(multirex_file, "multirex")
    spec_g = load_forward_model_spectrum(garlic_wavenumber_file, "garlic")

    assert len(spec_v.wavelength) == 3
    assert len(spec_m.wavelength) == 3
    assert len(spec_g.wavelength) == 3

    with pytest.raises(ValueError, match="Unsupported forward model"):
        load_forward_model_spectrum(vpl_file_2col, "nonexistent")


# ==============================================================================
# Tests for Retrieval Ingestion
# ==============================================================================


def test_parse_poseidon(poseidon_hdf5_file):
    spec = parse_poseidon_retrieval(poseidon_hdf5_file, spectrum_type="transmission")
    assert isinstance(spec, TransmissionSpectrum)
    np.testing.assert_allclose(spec.wavelength, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(spec.transit_depth, [0.01, 0.02, 0.015])
    np.testing.assert_allclose(spec.err, [0.001, 0.002, 0.0015])


def test_parse_aurora(aurora_hdf5_file):
    spec = parse_aurora_retrieval(aurora_hdf5_file, spectrum_type="transmission")
    np.testing.assert_allclose(spec.wavelength, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(spec.transit_depth, [0.01, 0.02, 0.015])


def test_parse_chimera(chimera_file):
    spec = parse_chimera_retrieval(chimera_file, spectrum_type="transmission")
    np.testing.assert_allclose(spec.wavelength, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(spec.transit_depth, [0.01, 0.02, 0.015])


def test_parse_petitradtrans(petitradtrans_hdf5_file):
    spec = parse_petitradtrans_retrieval(
        petitradtrans_hdf5_file, spectrum_type="transmission"
    )
    np.testing.assert_allclose(spec.wavelength, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(spec.transit_depth, [0.01, 0.02, 0.015])


# ==============================================================================
# Tests for GCM & Climate Ingestion
# ==============================================================================


def test_parse_atmos(atmos_file):
    res = parse_atmos_profile(atmos_file)
    np.testing.assert_allclose(res["altitude"], [0.0, 10.0, 20.0])
    np.testing.assert_allclose(res["pressure"], [1.0, 0.5, 0.1])
    np.testing.assert_allclose(res["temperature"], [300.0, 260.0, 220.0])
    np.testing.assert_allclose(res["abundances"]["H2O"], [1.0e-4, 5.0e-5, 1.0e-5])
    np.testing.assert_allclose(res["abundances"]["CO2"], [3.5e-4, 3.5e-4, 3.5e-4])


def test_gcm_workflows(gcm_hdf5_file):
    gcm = GCMWorkflows(gcm_hdf5_file)

    # Coordinates check
    np.testing.assert_allclose(gcm.lat, [-45.0, 0.0, 45.0])
    np.testing.assert_allclose(gcm.lon, [0.0, 90.0, 180.0, 270.0])
    np.testing.assert_allclose(gcm.pressure, [1.0, 0.1, 0.01])

    # 1. Global average
    glob = gcm.global_average()
    np.testing.assert_allclose(glob["pressure"], [1.0, 0.1, 0.01])
    np.testing.assert_allclose(glob["temperature"], [300.0, 250.0, 200.0])
    np.testing.assert_allclose(glob["abundances"]["H2O"], [1e-4, 1e-4, 1e-4])

    # 2. Limb average
    limb = gcm.limb_average(limb_lons=[90.0, 270.0])
    np.testing.assert_allclose(limb["temperature"], [300.0, 250.0, 200.0])
    np.testing.assert_allclose(limb["abundances"]["CO2"], [3e-4, 3e-4, 3e-4])

    # 3. Spatial average
    spatial = gcm.spatial_average(lat_bounds=(-10.0, 10.0), lon_bounds=(80.0, 100.0))
    np.testing.assert_allclose(spatial["temperature"], [300.0, 250.0, 200.0])

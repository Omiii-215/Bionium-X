"""
External integration data interfaces, parsers, and GCM workflows.
"""

import numpy as np
import pandas as pd
import h5py
from bioniumx.spectra import TransmissionSpectrum, EmissionSpectrum

# ==============================================================================
# 1. SPECTRAL FORWARD MODELS (VPL, MULTIREX, GARLIC)
# ==============================================================================


def parse_vpl_spectrum(filepath, spectrum_type="transmission", **metadata):
    """
    Parse Virtual Planetary Laboratory (VPL) ASCII/txt spectrum output.

    Expected format: Space/tab-separated columns, optionally starting with '#' or '%' comments.
    Columns: wavelength (microns) and transit depth (or flux), and optional error.
    """
    # Read lines and filter out comments starting with '#' or '%'
    clean_lines = []
    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith(("#", "%")):
                clean_lines.append(stripped)

    # Parse whitespace-separated numbers from clean lines
    data = []
    for line in clean_lines:
        row = [float(x) for x in line.split()]
        data.append(row)
    data = np.array(data)

    if data.ndim == 1:
        raise ValueError("VPL spectrum file must contain at least 2 columns.")

    wavelength = data[:, 0]
    flux_or_depth = data[:, 1]
    err = data[:, 2] if data.shape[1] > 2 else None

    if spectrum_type == "transmission":
        return TransmissionSpectrum(wavelength, flux_or_depth, err=err, **metadata)
    else:
        return EmissionSpectrum(wavelength, flux_or_depth, err=err, **metadata)


def parse_multirex_spectrum(filepath, spectrum_type="transmission", **metadata):
    """
    Parse MULTIREX CSV or ASCII spectrum output.

    Expected columns: 'wavelength' (microns), 'transit_depth' (or 'flux'), and optional 'error'/'noise'.
    """
    df = pd.read_csv(filepath, comment="#")

    # Identify columns
    wl_col = next(
        (c for c in df.columns if c.lower() in ["wavelength", "wave", "wl"]),
        df.columns[0],
    )

    if spectrum_type == "transmission":
        val_col = next(
            (c for c in df.columns if c.lower() in ["transit_depth", "depth", "dppm"]),
            df.columns[1],
        )
    else:
        val_col = next(
            (c for c in df.columns if c.lower() in ["flux", "emission", "intensity"]),
            df.columns[1],
        )

    err_col = next(
        (c for c in df.columns if c.lower() in ["error", "noise", "err", "dppm_err"]),
        None,
    )

    wavelength = df[wl_col].to_numpy()
    flux_or_depth = df[val_col].to_numpy()
    err = df[err_col].to_numpy() if err_col else None

    # Handle parts per million (dppm) scaling if needed
    if "dppm" in val_col.lower() and flux_or_depth.max() > 1.0:
        flux_or_depth = flux_or_depth / 1_000_000.0
        if err is not None:
            err = err / 1_000_000.0

    if spectrum_type == "transmission":
        return TransmissionSpectrum(wavelength, flux_or_depth, err=err, **metadata)
    else:
        return EmissionSpectrum(wavelength, flux_or_depth, err=err, **metadata)


def parse_garlic_spectrum(filepath, spectrum_type="transmission", **metadata):
    """
    Parse GARLIC output spectrum. Supports converting wavenumbers (cm-1) to wavelengths (microns).
    """
    df = pd.read_csv(filepath, comment="#", sep=r"\s+", header=None)
    data = df.to_numpy()

    if data.ndim == 1:
        raise ValueError("GARLIC spectrum file must contain at least 2 columns.")

    col0 = data[:, 0]
    flux_or_depth = data[:, 1]
    err = data[:, 2] if data.shape[1] > 2 else None

    # Check if first column is wavenumber (cm-1) or wavelength (microns)
    # Wavelength is typically in range 0.1 to 30. Wavenumbers are usually > 100.
    if col0.max() > 100:
        # Convert wavenumber to wavelength: microns = 10000 / wavenumber
        wavelength = 10000.0 / col0

        # Sort by wavelength in ascending order (wavenumber is descending)
        idx = np.argsort(wavelength)
        wavelength = wavelength[idx]
        flux_or_depth = flux_or_depth[idx]
        if err is not None:
            err = err[idx]
    else:
        wavelength = col0

    if spectrum_type == "transmission":
        return TransmissionSpectrum(wavelength, flux_or_depth, err=err, **metadata)
    else:
        return EmissionSpectrum(wavelength, flux_or_depth, err=err, **metadata)


def load_forward_model_spectrum(
    filepath, model_name, spectrum_type="transmission", **metadata
):
    """
    Unified loader router for forward model outputs.
    """
    name = model_name.lower().strip()
    if name == "vpl":
        return parse_vpl_spectrum(filepath, spectrum_type, **metadata)
    elif name == "multirex":
        return parse_multirex_spectrum(filepath, spectrum_type, **metadata)
    elif name == "garlic":
        return parse_garlic_spectrum(filepath, spectrum_type, **metadata)
    else:
        raise ValueError(
            f"Unsupported forward model: {model_name}. Supported: 'vpl', 'multirex', 'garlic'"
        )


# ==============================================================================
# 2. BAYESIAN RETRIEVAL CODES (POSEIDON, AURORA, CHIMERA, petitRADTRANS)
# ==============================================================================


def parse_poseidon_retrieval(filepath, spectrum_type="transmission", **metadata):
    """
    Parse POSEIDON HDF5 retrieval output file.

    Extracts wavelength, median spectrum, and 1-sigma uncertainty.
    """
    with h5py.File(filepath, "r") as f:
        # Look for wavelength dataset
        wl_key = next(
            (k for k in f if k.lower() in ["wl", "wavelength", "wl_grid"]), None
        )
        if not wl_key:
            raise KeyError("Could not find wavelength dataset in POSEIDON file.")
        wavelength = f[wl_key][:]

        # Look for spectrum dataset (median)
        spec_keys = [
            "spectrum_median",
            "transit_depth_median",
            "depth_median",
            "transit_depth",
            "flux_median",
            "flux",
            "spectrum",
        ]
        spec_key = next((k for k in f if k.lower() in spec_keys), None)
        if not spec_key:
            raise KeyError(
                "Could not find spectrum/transit depth dataset in POSEIDON file."
            )
        flux_or_depth = f[spec_key][:]

        # Look for uncertainty (1-sigma confidence bounds or standard error)
        err = None
        err_keys = [
            "spectrum_1sigma",
            "depth_1sigma",
            "error",
            "noise",
            "err",
            "spectrum_err",
        ]
        err_key = next((k for k in f if k.lower() in err_keys), None)

        if err_key:
            err = f[err_key][:]
        else:
            # Check for lower/upper bounds
            up_key = next(
                (
                    k
                    for k in f
                    if k.lower() in ["spectrum_up", "spectrum_1sigma_upper", "depth_up"]
                ),
                None,
            )
            down_key = next(
                (
                    k
                    for k in f
                    if k.lower()
                    in ["spectrum_down", "spectrum_1sigma_lower", "depth_down"]
                ),
                None,
            )
            if up_key and down_key:
                err = (f[up_key][:] - f[down_key][:]) / 2.0

    if spectrum_type == "transmission":
        return TransmissionSpectrum(wavelength, flux_or_depth, err=err, **metadata)
    else:
        return EmissionSpectrum(wavelength, flux_or_depth, err=err, **metadata)


def parse_aurora_retrieval(filepath, spectrum_type="transmission", **metadata):
    """
    Parse AURORA retrieval output (supports HDF5 or CSV/ASCII representation).
    """
    if filepath.endswith(".h5") or filepath.endswith(".hdf5"):
        with h5py.File(filepath, "r") as f:
            keys = list(f.keys())
            wl_key = next((k for k in f if k.lower() in ["wl", "wavelength"]), keys[0])
            val_key = next(
                (
                    k
                    for k in f
                    if k.lower() in ["spectrum", "median", "flux", "transit_depth"]
                ),
                keys[1],
            )
            err_key = next(
                (k for k in f if k.lower() in ["error", "sigma", "std", "err"]), None
            )

            wavelength = f[wl_key][:]
            flux_or_depth = f[val_key][:]
            err = f[err_key][:] if err_key else None
    else:
        # Assume CSV/ASCII format
        df = pd.read_csv(filepath, comment="#")
        wavelength = df.iloc[:, 0].to_numpy()
        flux_or_depth = df.iloc[:, 1].to_numpy()
        err = df.iloc[:, 2].to_numpy() if df.shape[1] > 2 else None

    if spectrum_type == "transmission":
        return TransmissionSpectrum(wavelength, flux_or_depth, err=err, **metadata)
    else:
        return EmissionSpectrum(wavelength, flux_or_depth, err=err, **metadata)


def parse_chimera_retrieval(filepath, spectrum_type="transmission", **metadata):
    """
    Parse CHIMERA output file (typically ASCII format with columns).
    """
    df = pd.read_csv(filepath, comment="#", sep=r"\s+", header=None)
    data = df.to_numpy()

    wavelength = data[:, 0]
    flux_or_depth = data[:, 1]
    err = data[:, 2] if data.shape[1] > 2 else None

    if spectrum_type == "transmission":
        return TransmissionSpectrum(wavelength, flux_or_depth, err=err, **metadata)
    else:
        return EmissionSpectrum(wavelength, flux_or_depth, err=err, **metadata)


def parse_petitradtrans_retrieval(filepath, spectrum_type="transmission", **metadata):
    """
    Parse petitRADTRANS generated or saved output file.
    """
    if filepath.endswith(".h5") or filepath.endswith(".hdf5"):
        with h5py.File(filepath, "r") as f:
            wavelength = f["wavelength"][:]
            flux_or_depth = (
                f["transit_depth"][:] if "transit_depth" in f else f["flux"][:]
            )
            err = f["err"][:] if "err" in f else None
    else:
        df = pd.read_csv(filepath, comment="#")
        wavelength = df.iloc[:, 0].to_numpy()
        flux_or_depth = df.iloc[:, 1].to_numpy()
        err = df.iloc[:, 2].to_numpy() if df.shape[1] > 2 else None

    if spectrum_type == "transmission":
        return TransmissionSpectrum(wavelength, flux_or_depth, err=err, **metadata)
    else:
        return EmissionSpectrum(wavelength, flux_or_depth, err=err, **metadata)


# ==============================================================================
# 3. 3D GCMS & PHOTOCHEMICAL MODELS (ATMOS, ROCKE-3D, EXOCAM)
# ==============================================================================


def parse_atmos_profile(filepath):
    """
    Parse Atmos 1D photochemical and climate vertical atmospheric profile.

    Expected format: ASCII table with columns for Alt, Pressure, Temp, and Species.
    """
    # Find start of data if there are header lines
    skiprows = 0
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if any(term in line.upper() for term in ["ALT", "PRESS", "TEMP", "DEPTH"]):
                skiprows = i
                break

    df = pd.read_csv(filepath, sep=r"\s+", skiprows=skiprows)

    # Identify columns
    alt_col = next(
        (c for c in df.columns if c.upper() in ["ALT", "Z", "ALTITUDE"]), None
    )
    press_col = next(
        (c for c in df.columns if c.upper() in ["PRESS", "P", "PRESSURE"]), None
    )
    temp_col = next(
        (c for c in df.columns if c.upper() in ["TEMP", "T", "TEMPERATURE"]), None
    )

    if press_col is None or temp_col is None:
        raise ValueError("Atmos profile must contain pressure and temperature columns.")

    altitude = df[alt_col].to_numpy() if alt_col else None
    pressure = df[press_col].to_numpy()
    temperature = df[temp_col].to_numpy()

    # Extract remaining columns as chemical species mixing ratios
    exclude_cols = {alt_col, press_col, temp_col}
    abundances = {}
    for col in df.columns:
        if col not in exclude_cols:
            abundances[col] = df[col].to_numpy()

    return {
        "altitude": altitude,
        "pressure": pressure,
        "temperature": temperature,
        "abundances": abundances,
    }


class GCMWorkflows:
    """
    Workflow manager for ingesting and processing 3D GCM datasets (ROCKE-3D, ExoCAM, etc.).
    Reads NetCDF4 format files utilizing h5py to avoid heavy dependencies.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self._load_dataset()

    def _load_dataset(self):
        with h5py.File(self.filepath, "r") as f:
            # Locate coordinate names
            lat_key = next((k for k in f if k.lower() in ["lat", "latitude"]), None)
            lon_key = next((k for k in f if k.lower() in ["lon", "longitude"]), None)
            lev_key = next(
                (
                    k
                    for k in f
                    if k.lower() in ["lev", "plev", "level", "pressure", "p"]
                ),
                None,
            )

            if not all([lat_key, lon_key, lev_key]):
                raise KeyError(
                    f"Missing coordinate grids in GCM file. Found: lat={lat_key}, lon={lon_key}, level={lev_key}"
                )

            self.lat = f[lat_key][:]
            self.lon = f[lon_key][:]
            self.pressure = f[lev_key][:]

            # Locate temperature variable
            temp_key = next(
                (k for k in f if k.lower() in ["temp", "temperature", "t"]), None
            )
            if not temp_key:
                raise KeyError("Temperature dataset 'T'/'temp' not found in GCM file.")

            t_data = f[temp_key][:]
            # Ensure shape is (lev, lat, lon). If there is a leading time axis, take index 0.
            if t_data.ndim == 4:
                t_data = t_data[0]
            self.temperature = t_data

            # Extract chemistry variables (common atmospheric gases)
            self.abundances = {}
            for key in f.keys():
                if key.upper() in [
                    "H2O",
                    "CH4",
                    "CO2",
                    "CO",
                    "O2",
                    "O3",
                    "NH3",
                    "PH3",
                    "N2O",
                    "N2",
                    "HE",
                    "H2",
                    "AR",
                ]:
                    chem_data = f[key][:]
                    if chem_data.ndim == 4:
                        chem_data = chem_data[0]
                    self.abundances[key.upper()] = chem_data

    def global_average(self):
        """
        Compute the globally averaged 1D vertical profile.
        """
        # Average over latitude (axis 1) and longitude (axis 2)
        temp_1d = np.mean(self.temperature, axis=(1, 2))

        abundances_1d = {}
        for gas, data in self.abundances.items():
            abundances_1d[gas] = np.mean(data, axis=(1, 2))

        return {
            "pressure": self.pressure,
            "temperature": temp_1d,
            "abundances": abundances_1d,
        }

    def limb_average(self, limb_lons=None):
        """
        Compute the vertical profile averaged along the limb/terminator.
        By default, averages at longitudes around 90 and 270 degrees.
        """
        if limb_lons is None:
            limb_lons = [90.0, 270.0]

        # Find longitude indices closest to the limb/terminator
        indices = []
        for l_lon in limb_lons:
            # Map negative longitudes to 0-360 range if necessary
            mapped_lon = self.lon % 360.0
            idx = np.argmin(np.abs(mapped_lon - (l_lon % 360.0)))
            indices.append(idx)

        # Average over all latitudes (axis 1) and selected longitudes (axis 2)
        temp_slice = self.temperature[:, :, indices]
        temp_1d = np.mean(temp_slice, axis=(1, 2))

        abundances_1d = {}
        for gas, data in self.abundances.items():
            gas_slice = data[:, :, indices]
            abundances_1d[gas] = np.mean(gas_slice, axis=(1, 2))

        return {
            "pressure": self.pressure,
            "temperature": temp_1d,
            "abundances": abundances_1d,
        }

    def spatial_average(self, lat_bounds, lon_bounds):
        """
        Compute vertical profiles averaged over a specific spatial window.
        """
        lat_min, lat_max = lat_bounds
        lon_min, lon_max = lon_bounds

        # Lat/Lon masks
        lat_mask = (self.lat >= lat_min) & (self.lat <= lat_max)
        # Handle lon wrapping if needed
        lon_mask = (self.lon >= lon_min) & (self.lon <= lon_max)

        if not np.any(lat_mask) or not np.any(lon_mask):
            raise ValueError(
                "No grid points found within the specified lat/lon bounds."
            )

        # Extract slices and average
        temp_slice = self.temperature[:, lat_mask, :][:, :, lon_mask]
        temp_1d = np.mean(temp_slice, axis=(1, 2))

        abundances_1d = {}
        for gas, data in self.abundances.items():
            gas_slice = data[:, lat_mask, :][:, :, lon_mask]
            abundances_1d[gas] = np.mean(gas_slice, axis=(1, 2))

        return {
            "pressure": self.pressure,
            "temperature": temp_1d,
            "abundances": abundances_1d,
        }

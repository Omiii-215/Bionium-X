import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


def filter_noise(flux, window_length=15, polyorder=3):
    """
    Apply Savitzky-Golay filter to smooth the spectrum.
    """
    # ensure window_length is odd and less than len(flux)
    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(flux):
        window_length = len(flux) - 1 if (len(flux) -
                                          1) % 2 != 0 else len(flux) - 2

    smoothed_flux = savgol_filter(
        flux, window_length=max(
            3, window_length), polyorder=min(
            polyorder, window_length - 1))
    return smoothed_flux


def normalize_spectrum(flux, method='minmax'):
    """
    Normalize the spectrum flux.
    methods: 'minmax' or 'zscore'
    """
    if method == 'minmax':
        min_val = np.min(flux)
        max_val = np.max(flux)
        if max_val - min_val == 0:
            return flux
        return (flux - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = np.mean(flux)
        std_val = np.std(flux)
        if std_val == 0:
            return flux
        return (flux - mean_val) / std_val
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'.")


def interpolate_spectrum(old_wavelength, old_flux, new_wavelength):
    """
    Interpolate spectrum to a standard new wavelength grid.
    Useful for standardizing inputs to ML models.
    """
    f = interp1d(
        old_wavelength,
        old_flux,
        kind='linear',
        bounds_error=False,
        fill_value="extrapolate")
    new_flux = f(new_wavelength)
    return new_flux


def preprocess_pipeline(wavelength, flux, target_wavelength_grid=None):
    """
    End-to-end preprocessing pipeline for a single spectrum.
    """
    # 1. Filter noise
    smoothed_flux = filter_noise(flux)

    # 2. Interpolate if target grid provided
    if target_wavelength_grid is not None:
        intp_flux = interpolate_spectrum(
            wavelength, smoothed_flux, target_wavelength_grid)
        wl = target_wavelength_grid
    else:
        intp_flux = smoothed_flux
        wl = wavelength

    # 3. Normalize
    norm_flux = normalize_spectrum(intp_flux, method='minmax')

    return wl, norm_flux

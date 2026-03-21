from typing import Any
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths


def extract_features(wavelength, flux):
    """
    Extract physics-based features from the spectrum.

    Returns a dictionary of features:
    - num_absorption_peaks: int
    - max_absorption_depth: float
    - continuum_slope: float
    - top_peaks_wavelength: list of float
    """
    features: dict[str, Any] = {}

    # In transmission spectra, molecules show as absorption features (dips).
    # To use find_peaks effectively, we invert the spectrum to find the dips
    # as peaks.
    inverted_flux = -flux

    # Find peaks (valleys in original)
    # Using relative prominence to avoid finding too many noise peaks
    prominence = np.std(flux) * 0.5
    peaks, properties = find_peaks(
        inverted_flux, prominence=max(
            1e-5, prominence))

    features['num_absorption_peaks'] = len(peaks)

    if len(peaks) > 0:
        # Depth is the prominence in the inverted spectrum
        depths = properties['prominences']
        features['max_absorption_depth'] = float(np.max(depths))

        # Wwidths
        widths, width_heights, left_ips, right_ips = peak_widths(
            inverted_flux, peaks, rel_height=0.5)
        features['mean_line_width'] = float(np.mean(widths))

        # Sort peaks by depth (prominence) descending
        sorted_indices = np.argsort(depths)[::-1]
        top_indices = peaks[sorted_indices[:5]]  # pyre-ignore
        features['top_peaks_wavelengths'] = [
            float(w) for w in wavelength[top_indices]]
        # Pad context if fewer than 5
        while len(features['top_peaks_wavelengths']) < 5:
            features['top_peaks_wavelengths'].append(0.0)
    else:
        features['max_absorption_depth'] = 0.0
        features['mean_line_width'] = 0.0
        features['top_peaks_wavelengths'] = [0.0] * 5

    # Continuum slope (linear fit)
    slope, intercept = np.polyfit(wavelength, flux, 1)
    features['continuum_slope'] = float(slope)

    return features


def tabularize_features(wavelength, flux_array):
    """
    Extract features for a dataset of spectra and return tabular features.
    flux_array: (N_samples, N_wavelengths)
    """
    all_features = []
    for flux in flux_array:
        feat = extract_features(wavelength, flux)
        # Flatten for tabular format
        flat_feat = {
            'num_absorption_peaks': feat['num_absorption_peaks'],
            'max_absorption_depth': feat['max_absorption_depth'],
            'mean_line_width': feat['mean_line_width'],
            'continuum_slope': feat['continuum_slope'],
        }
        for i, w in enumerate(feat['top_peaks_wavelengths']):
            flat_feat[f'peak_{i + 1}_wl'] = w
        all_features.append(flat_feat)

    return pd.DataFrame(all_features)

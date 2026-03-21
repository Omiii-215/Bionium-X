import numpy as np
import pandas as pd


class SpectrumGenerator:
    def __init__(self, wl_min=0.5, wl_max=10.0, num_points=1000):
        self.wl_min = wl_min
        self.wl_max = wl_max
        self.num_points = num_points
        self.wavelengths = np.linspace(wl_min, wl_max, num_points)

        # Define absorption features: (wavelength, width)
        self.features = {
            'O2': (0.76, 0.05),
            'CH4': (1.65, 0.1),
            'H2O': (1.40, 0.15),
            'O3': (9.60, 0.3),
            'CO2': (4.30, 0.2)
        }

    def _gaussian_dip(self, wl, center, width, depth):
        return depth * np.exp(-((wl - center) ** 2) / (2 * width ** 2))

    def generate_spectrum(
            self,
            present_molecules,
            noise_level=0.02,
            base_flux=1.0):
        """
        Generate a synthetic transmission spectrum.
        present_molecules: dict of molecule name to depth, e.g.,
        {'O2': 0.1, 'CH4': 0.15}
        """
        # Start with base continuum
        flux = np.full(self.num_points, base_flux)

        # Add a slight slope to the continuum
        continuum_slope = np.linspace(0, 0.1, self.num_points)
        flux -= continuum_slope

        # Inject absorption features
        for mol, depth in present_molecules.items():
            if mol in self.features and depth > 0:
                center, width = self.features[mol]
                dip = self._gaussian_dip(
                    self.wavelengths, center, width, depth)
                flux -= dip

        # Add noise
        noise = np.random.normal(0, noise_level, self.num_points)
        flux += noise

        # Ensure flux doesn't drop below 0 unrealistically
        flux = np.clip(flux, 0, None)

        return self.wavelengths, flux, noise

    def generate_dataset(self, n_samples=1000):
        """
        Generate a dataset of spectra with random molecule presence.
        Returns tabular data where rows are samples, plus target labels.
        """
        data = []
        labels = []

        molecules = list(self.features.keys())

        for _ in range(n_samples):
            # Randomly decide which molecules are present (probability 0.5)
            # and assign them random depths
            present = {}
            label = {}
            for mol in molecules:
                is_present = np.random.rand() > 0.5
                if is_present:
                    present[mol] = np.random.uniform(0.05, 0.25)
                    label[mol] = 1
                else:
                    label[mol] = 0

            wl, flux, _ = self.generate_spectrum(present)
            data.append(flux)
            labels.append(label)

        df_flux = pd.DataFrame(
            data, columns=[
                f"wl_{
                    w:.3f}" for w in self.wavelengths])
        df_labels = pd.DataFrame(labels)

        return self.wavelengths, df_flux, df_labels


if __name__ == '__main__':
    gen = SpectrumGenerator()
    wl, df_flux, df_labels = gen.generate_dataset(n_samples=10)
    print("Generated 10 samples for testing.")
    print("Flux shape:", df_flux.shape)
    print("Labels shape:", df_labels.shape)

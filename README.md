<p align="center">
  <img src="docs/_static/bioniumx_logo.jpg" alt="Bionium-X Logo" width="300">
</p>

# BioniumX

<table>
<tr>
  <th>Usage</th>
  <th>Release</th>
  <th>Development</th>
  <th>Community</th>
</tr>
<tr>
  <td>
    <img src="https://img.shields.io/badge/python->=3.9-blue.svg" alt="Python Version"/>
    <br>
    <img src="https://img.shields.io/badge/docs-latest-brightgreen.svg" alt="Documentation Status"/>
    <br>
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"/>
  </td>
  <td>
    <img src="https://img.shields.io/badge/release-v1.0.0-blue.svg" alt="Release"/>
  </td>
  <td>
    <img src="https://img.shields.io/badge/repo%20status-Active-brightgreen.svg" alt="Repo Status"/>
  </td>
  <td>
  </td>
</tr>
</table>

## Exoplanetary Biosignature Detection Made Easy

Bionium-X is a rigorous, pure-Python scientific library for high-performance exoplanetary biosignature detection and atmospheric modeling. It merges existing efforts for transmission and emission spectra analysis in Python, and is structured with the best guidelines for modern open-source programming.

It provides:

* a library of preprocessing methods, including Savitzky-Golay filtering, 1D Gaussian smoothing, and polynomial continuum normalization;
* a set of scripts to natively fetch and load real JWST and Hubble spectroscopic data from public archives;
* template cross-correlation with native connection to the Harvard HITRAN API via `radis` to confidently detect specific gases;
* astrobiological physics calculators to compute Earth Similarity Indices (ESI), habitability scores, and chemical disequilibrium;
* experimental machine learning interoperability, including 1D CNNs, Random Forests, and Spectral Transformers for automated feature extraction.

There are a number of official software packages for exoplanet transit fitting and atmospheric retrieval. However, an equivalent widely-used package does not exist for automated, end-to-end biosignature detection: to date, that has generally been done with custom scripts. Bionium-X aims not only to fill that gap, but also to provide implementations of the most advanced spectral analysis techniques available in the literature. The ultimate goal of this project is to provide the community with a package that eases the learning curve for advanced biosignature detection with a correct statistical framework.

More details of current and planned capabilities are available in the [Bionium-X documentation](https://omiii-215.github.io/BioniumX/).

## Installation and Testing

Bionium-X can be installed directly from the source repository itself. Our documentation provides comprehensive installation instructions.

```bash
# Clone the repository
git clone https://github.com/omiii-215/BioniumX.git
cd BioniumX

# Initialize virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

After installation, it's a good idea to run the test suite. We use `pytest` for testing.

## Documentation

Bionium-X's documentation can be found locally after building:
```bash
cd docs
make clean && make html
# Open docs/_build/html/index.html in your browser
```

## Getting In Touch, and Getting Involved

We welcome contributions and feedback, and we need your help! The best way to get in touch is via the issues page. We're especially interested in hearing from you:

* If something breaks;
* If you spot missing functionality, find the API unintuitive, or have suggestions for future development;
* If you have your own code implementing any of the methods provided by Bionium-X and it produces different answers.

Even better: if you have code you'd be willing to contribute, please send a pull request or open an issue.

## Related Packages

* [radis](https://radis.readthedocs.io/) provides the high-resolution theoretical absorption templates and cross-sections used by Bionium-X.
* [pooch](https://www.fatiando.org/pooch/latest/) is used to seamlessly download and cache real observational datasets.

## Citing Bionium-X

If you find this package useful in your research, please provide appropriate acknowledgement and citation. Our documentation gives further guidance, including links to appropriate papers and convenient BibTeX entries.

## Copyright & Licensing

All content © 2026 The Authors. The code is distributed under the MIT license; see [LICENSE](LICENSE) for details.

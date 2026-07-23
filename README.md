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
git clone https://github.com/YourOrg/Bionium-X.git
cd Bionium-X

# Initialize virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

After installation, it's a good idea to run the test suite. We use `pytest` for testing.

## Quick Start

This section demonstrates a typical Bionium-X workflow, from loading spectral data to calculating planetary habitability metrics.

### Step 1: Import Required Modules

```python
from bioniumx.preprocessing import savgol_filter
from bioniumx.metrics import earth_similarity_index
```

Bionium-X provides modular components for spectral preprocessing, atmospheric analysis, biosignature detection, and habitability assessment.

---

### Step 2: Prepare Spectral Data

Observational data collected from telescopes often contains noise and instrumental artifacts. Before analysis, the spectrum should be cleaned and normalized.

```python
raw_spectrum = [
    0.12, 0.15, 0.18, 0.17,
    0.20, 0.23, 0.25, 0.24
]
```

---

### Step 3: Apply Spectral Preprocessing

Use the Savitzky-Golay filter to smooth the spectrum while preserving important spectral features.

```python
processed_spectrum = savgol_filter(raw_spectrum)

print(processed_spectrum)
```

Example Output:

```text
[0.13, 0.15, 0.17, 0.18, 0.21, 0.23, 0.24, 0.24]
```

This preprocessing step helps improve the reliability of downstream biosignature detection algorithms.

---

### Step 4: Compute Planetary Habitability Metrics

Bionium-X includes tools for evaluating exoplanet habitability.

```python
esi = earth_similarity_index(
    radius=1.02,
    density=1.01,
    escape_velocity=1.00,
    surface_temperature=288
)

print(f"Earth Similarity Index: {esi:.3f}")
```

Example Output:

```text
Earth Similarity Index: 0.985
```

The Earth Similarity Index (ESI) measures how closely an exoplanet resembles Earth based on several physical characteristics.

---

### Step 5: Integrate into Research Pipelines

The processed spectral data and computed metrics can be combined with:

- Atmospheric retrieval workflows
- Molecular detection pipelines
- Template cross-correlation analysis
- Machine learning classification models
- Biosignature confidence scoring systems

This modular design allows researchers to integrate Bionium-X into existing exoplanet analysis workflows with minimal effort.

---

### Complete Example

```python
from bioniumx.preprocessing import savgol_filter
from bioniumx.metrics import earth_similarity_index

# Example observational spectrum
raw_spectrum = [
    0.12, 0.15, 0.18, 0.17,
    0.20, 0.23, 0.25, 0.24
]

# Preprocess spectrum
processed_spectrum = savgol_filter(raw_spectrum)

# Calculate habitability metric
esi = earth_similarity_index(
    radius=1.02,
    density=1.01,
    escape_velocity=1.00,
    surface_temperature=288
)

print("Processed Spectrum:", processed_spectrum)
print("Earth Similarity Index:", esi)
```

---


## Documentation

Bionium-X's documentation can be found locally after building:
```bash
cd docs
make clean && make html
# Open docs/_build/html/index.html in your browser
```

## Project Structure

Bionium-X follows a modular architecture designed to separate scientific computation, data processing, machine learning workflows, testing, and documentation. This organization improves maintainability, scalability, and contributor onboarding.

```text
BioniumX/
│
├── bioniumx/                 # Core library source code
│   ├── preprocessing/        # Spectral cleaning and signal processing
│   ├── detection/            # Biosignature detection algorithms
│   ├── habitability/         # Habitability and ESI calculations
│   ├── ml/                   # Machine learning models and utilities
│   ├── datasets/             # Dataset loading and management
│   └── utils/                # Shared helper functions
│
├── docs/                     # Documentation source files
│   ├── source/
│   └── _static/
│
├── tests/                    # Unit and integration test suite
│
├── examples/                 # Example scripts demonstrating workflows
│
├── notebooks/                # Interactive Jupyter notebooks
│
├── .github/                  # GitHub workflows and templates
│   ├── workflows/
│   └── ISSUE_TEMPLATE/
│
├── README.md                 # Main project documentation
├── CONTRIBUTING.md           # Contribution guidelines
├── LICENSE                   # Project license
├── requirements.txt          # Python dependencies
└── pyproject.toml            # Build and package configuration
```

### Core Package (`bioniumx/`)

The `bioniumx` package contains the primary scientific and computational functionality of the project.

| Module | Purpose |
|----------|----------|
| `preprocessing` | Signal enhancement, filtering, smoothing, and normalization |
| `detection` | Biosignature identification and cross-correlation algorithms |
| `habitability` | Earth Similarity Index (ESI) and planetary habitability metrics |
| `ml` | Machine learning models for automated spectral analysis |
| `datasets` | Data acquisition and archive integration utilities |
| `utils` | Shared helper functions and reusable utilities |

---

### Documentation (`docs/`)

The documentation directory contains all resources used to build the project's official documentation.

Key responsibilities include:

- User guides
- API references
- Tutorials
- Installation instructions
- Developer documentation

Contributors updating functionality should also update relevant documentation.

---

### Testing (`tests/`)

The testing suite ensures reliability and scientific reproducibility across the project.

Tests may include:

- Unit tests
- Integration tests
- Regression tests
- Data validation tests

Before submitting a Pull Request, contributors should execute:

```bash
pytest
```

---

### Examples (`examples/`)

Contains practical examples demonstrating how to:

- Load observational datasets
- Preprocess spectra
- Detect atmospheric molecules
- Calculate habitability metrics
- Apply machine learning workflows

These examples serve as learning resources for new users.

---

### Notebooks (`notebooks/`)

Interactive notebooks provide:

- Step-by-step tutorials
- Research demonstrations
- Exploratory data analysis workflows
- Educational examples for students and researchers

---

### GitHub Configuration (`.github/`)

Stores repository automation and community management resources.

Examples include:

- Continuous Integration workflows
- Pull Request templates
- Issue templates
- Contribution automation

---

### Configuration Files

| File | Purpose |
|--------|----------|
| `README.md` | Project overview and onboarding guide |
| `CONTRIBUTING.md` | Contributor workflow and guidelines |
| `LICENSE` | Legal licensing information |
| `requirements.txt` | Dependency management |
| `pyproject.toml` | Package build configuration |

---

### Architectural Philosophy

Bionium-X is designed around the following principles:

- **Modularity** – Independent scientific components can evolve separately.
- **Reproducibility** – Scientific results should be verifiable and testable.
- **Extensibility** – New detection algorithms and models can be added easily.
- **Accessibility** – Lowering the barrier for students and first-time contributors.
- **Research-Oriented Development** – Aligning software design with modern exoplanetary science workflows.

## Architecture Overview

Bionium-X is designed using a modular scientific-computing architecture that separates data ingestion, preprocessing, analytical workflows, and machine learning components into independent but interoperable layers.

This design ensures scalability, maintainability, and ease of integration with future biosignature detection methodologies.

### High-Level Workflow

```text
Astronomical Data Sources
            │
            ▼
   Data Acquisition Layer
            │
            ▼
  Spectral Preprocessing Layer
            │
            ▼
   Feature Extraction Layer
            │
            ▼
 Biosignature Detection Layer
            │
            ▼
 Atmospheric & Habitability Analysis
            │
            ▼
 Machine Learning & Advanced Analytics
            │
            ▼
      Scientific Results
```

---

### 1. Data Acquisition Layer

The Data Acquisition Layer serves as the entry point for observational and simulated datasets.

Its responsibilities include:

- Fetching publicly available spectroscopic observations
- Loading JWST and Hubble datasets
- Managing data formats and metadata
- Ensuring compatibility between different astronomical archives

This layer minimizes the effort required to obtain and prepare real-world observational data for downstream analysis.

---

### 2. Spectral Preprocessing Layer

Raw astronomical spectra often contain instrumental noise, baseline drift, and observational artifacts.

The preprocessing layer provides robust signal-cleaning techniques including:

- Savitzky-Golay Filtering
- Gaussian Smoothing
- Continuum Normalization
- Noise Suppression
- Baseline Correction

The objective of this stage is to transform raw observations into scientifically reliable spectra suitable for biosignature analysis.

---

### 3. Feature Extraction Layer

Once spectra have been cleaned, relevant physical and chemical features must be extracted.

This layer is responsible for:

- Spectral peak identification
- Wavelength feature selection
- Absorption line characterization
- Statistical feature generation
- Candidate molecular signature extraction

The extracted features form the foundation for subsequent atmospheric and biosignature investigations.

---

### 4. Biosignature Detection Layer

This is the core scientific component of Bionium-X.

The detection engine focuses on identifying potential biosignature gases and molecular patterns through:

- Template Cross-Correlation
- Spectral Matching Algorithms
- Molecular Signature Detection
- HITRAN-Based Reference Comparisons
- Statistical Detection Confidence Estimation

The goal is to determine whether observed spectral features indicate the possible presence of biologically relevant compounds.

---

### 5. Atmospheric Characterization Layer

Beyond detecting individual molecules, understanding the overall atmospheric context is essential.

This layer enables:

- Atmospheric Composition Analysis
- Chemical Disequilibrium Assessment
- Planetary Environment Characterization
- Habitability Evaluation
- Earth Similarity Index (ESI) Computation

These metrics provide a broader scientific interpretation of the planetary environment.

---

### 6. Machine Learning and Advanced Analytics Layer

To support modern data-driven research, Bionium-X includes experimental machine learning interoperability.

Supported and planned capabilities include:

- Random Forest Models
- 1D Convolutional Neural Networks (CNNs)
- Spectral Transformers
- Automated Feature Extraction
- Classification and Ranking Pipelines

Machine learning workflows are designed to complement, rather than replace, traditional physics-based analysis techniques.

---

### Design Principles

Bionium-X follows several core architectural principles:

#### Modularity

Each subsystem can be developed, tested, and extended independently.

#### Scientific Reproducibility

Workflows are designed to produce consistent and verifiable results.

#### Extensibility

New detection algorithms, atmospheric models, and machine learning approaches can be integrated with minimal changes to existing code.

#### Performance

Computational workflows are optimized for large spectroscopic datasets while maintaining scientific accuracy.

#### Research Accessibility

The architecture is designed to be approachable for students and researchers while remaining powerful enough for advanced scientific investigations.

---

### Future Architectural Enhancements

Planned architectural improvements include:

- Distributed processing support
- Advanced atmospheric retrieval pipelines
- Real-time spectral analysis workflows
- Expanded molecular template databases
- Enhanced machine learning integration
- Interactive visualization and interpretation modules

These enhancements aim to establish Bionium-X as a comprehensive platform for next-generation exoplanet biosignature research.

## Troubleshooting

This section covers common installation, configuration, documentation, and testing issues that contributors and users may encounter while working with Bionium-X.

---

### Python Version Compatibility Issues

Bionium-X requires **Python 3.9 or higher**.

Verify your Python version:

```bash
python --version
```

or

```bash
python3 --version
```

If your version is below 3.9, please upgrade Python before proceeding.

---

### Virtual Environment Not Activating

Ensure that you have created the virtual environment successfully:

```bash
python -m venv venv
```

Activate it using:

#### Linux / macOS

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

If activation fails, verify that the virtual environment was created correctly and that execution permissions are enabled on your system.

---

### Dependency Installation Failures

Before installing project dependencies, upgrade `pip`, `setuptools`, and `wheel`:

```bash
pip install --upgrade pip setuptools wheel
```

Then reinstall dependencies:

```bash
pip install -r requirements.txt
```

If specific packages fail to install:

```bash
pip install <package-name>
```

Review the error logs carefully for missing system libraries or unsupported Python versions.

---

### ModuleNotFoundError

Example:

```text
ModuleNotFoundError: No module named 'bioniumx'
```

Possible solutions:

1. Ensure the virtual environment is activated.
2. Verify that dependencies are installed.
3. Install the package in editable mode:

```bash
pip install -e .
```

4. Restart your IDE or terminal session.

---

### Documentation Build Errors

If documentation generation fails:

```bash
cd docs
make clean
make html
```

Common causes include:

- Missing Sphinx dependencies
- Incorrect package imports
- Invalid Markdown or reStructuredText formatting
- Broken internal documentation references

Install documentation requirements if necessary:

```bash
pip install sphinx sphinx-rtd-theme
```

---

### Test Suite Failures

Run the test suite:

```bash
pytest bioniumx/tests/
```

If tests fail:

- Ensure all dependencies are installed.
- Confirm that you are using a supported Python version.
- Pull the latest changes from the main branch.
- Recreate the virtual environment if dependency conflicts exist.

For detailed output:

```bash
pytest -v
```

---

### Pre-Commit Hook Errors

Bionium-X uses automated code-quality checks through pre-commit hooks.

Install hooks:

```bash
pre-commit install
```

Run checks manually:

```bash
pre-commit run --all-files
```

If formatting issues are reported, apply the suggested fixes and commit again.

---

### Git Merge Conflicts

If your branch falls behind the main branch:

```bash
git checkout main
git pull origin main
git checkout your-branch
git merge main
```

Resolve conflicts manually, test your changes, and recommit.

---

### Continuous Integration (CI) Failures

A pull request may fail automated checks if:

- Tests are failing
- Documentation contains errors
- Formatting standards are not met
- Linting rules are violated

Before submitting a PR:

```bash
pytest
pre-commit run --all-files
```

Ensure all checks pass locally whenever possible.

---

### Unable to Build Documentation After Pulling Latest Changes

Clear previous build artifacts:

```bash
cd docs
make clean
```

Then rebuild:

```bash
make html
```

This resolves many issues caused by outdated cached files.

---

### Getting Additional Help

If the issue persists:

1. Search existing GitHub Issues.
2. Review the CONTRIBUTING.md guide.
3. Open a new issue with:
   - Operating System
   - Python Version
   - Error Logs
   - Steps to Reproduce

Providing detailed information helps maintainers diagnose and resolve problems more efficiently.

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

## Frequently Asked Questions (FAQ)

### What is Bionium-X?

Bionium-X is an open-source Python library focused on exoplanet biosignature detection, atmospheric characterization, and habitability analysis. It provides tools for processing astronomical spectra, detecting molecular signatures, evaluating planetary habitability, and exploring machine learning approaches for automated spectral analysis.

---

### Why was Bionium-X created?

While several excellent tools exist for exoplanet transit fitting and atmospheric retrieval, there is currently no widely adopted framework dedicated to end-to-end biosignature detection workflows. Bionium-X aims to bridge this gap by bringing together preprocessing, spectral analysis, atmospheric modeling, and machine learning capabilities within a single ecosystem.

---

### Who is Bionium-X designed for?

Bionium-X is intended for:

- Astronomy and Astrophysics Researchers
- Exoplanet Scientists
- Data Scientists working with astronomical datasets
- Undergraduate and Graduate Students
- Open-Source Contributors interested in scientific computing
- Educators and learners exploring astrobiology and planetary science

---

### What types of data can Bionium-X work with?

The library is designed to support spectroscopic observations commonly used in exoplanet research, including data obtained from:

- James Webb Space Telescope (JWST)
- Hubble Space Telescope (HST)
- Public astronomical archives
- User-generated spectral datasets

Support for additional observational sources will continue to expand as the project evolves.

---

### What biosignatures can be analyzed using Bionium-X?

Bionium-X aims to facilitate the detection and analysis of atmospheric molecules that may indicate biological or geological activity. Depending on available datasets and implemented models, users can investigate compounds such as:

- Water Vapor (H₂O)
- Methane (CH₄)
- Carbon Dioxide (CO₂)
- Oxygen (O₂)
- Ozone (O₃)

The list of supported molecules and detection methods is expected to grow with future releases.

---

### Does Bionium-X support machine learning workflows?

Yes. Bionium-X includes experimental machine learning interoperability and is being developed to support:

- Random Forest Models
- One-Dimensional Convolutional Neural Networks (1D CNNs)
- Spectral Transformer Architectures
- Automated Feature Extraction Pipelines

These capabilities are intended to assist researchers in identifying patterns within complex spectral datasets.

---

### Is Bionium-X suitable for beginners?

Absolutely. While some scientific background may be helpful, the project welcomes learners and first-time contributors. Documentation, examples, and community contributions are continuously being improved to make the learning curve more approachable.

Recommended starting areas include:

- Documentation Improvements
- Example Notebooks
- Unit Testing
- Bug Fixes
- Beginner-Friendly Issues

---

### How can I contribute to Bionium-X?

Contributions of all sizes are welcome. You can contribute by:

- Reporting bugs
- Improving documentation
- Writing tests
- Adding examples and tutorials
- Implementing new features
- Optimizing existing code

Please review the `CONTRIBUTING.md` guide before submitting a pull request.

---

### How do I report a bug or request a feature?

If you encounter a problem or have an idea for improvement:

1. Search existing GitHub issues to avoid duplicates.
2. Open a new issue with a clear title and description.
3. Include reproduction steps, screenshots, logs, or relevant context when applicable.

Well-documented issues help maintainers and contributors respond more effectively.

---

### How can I stay updated with project development?

You can stay informed by:

- Watching the GitHub repository
- Following release announcements
- Monitoring project discussions and issues
- Reviewing the project roadmap

Community participation and feedback play an important role in shaping future development.

---

### Is Bionium-X open source?

Yes. Bionium-X is released under the MIT License, allowing users to study, modify, distribute, and build upon the project while complying with the license terms.

---

### Where can I find the documentation?

Comprehensive documentation, installation instructions, API references, and usage guides are available through the project's documentation portal.

If you discover missing information or documentation gaps, feel free to open an issue or submit a pull request to help improve the project.

## Copyright & Licensing

All content © 2026 The Authors. The code is distributed under the MIT license; see [LICENSE](LICENSE) for details.

User Interface Overview
=======================

Once launched, Bionium-X initializes an interactive web-based UX.

Sidebar: Data Source & Control Panel
------------------------------------
The left-aligned sidebar governs global state configurations:

- **Data Source Engine**: Toggle between Known Exoplanet catalogs, Synthetic model generators, and Manual user uploads.
- **Spectral Band Highlights**: Enable/disable precision colored overlays bounding O₂, CH₄, O₃, H₂O, and CO₂ lines natively on visualization plots.

The Main Dashboard
------------------
The **Dashboard** tab is the analytical core of Bionium-X:

- **Transmission Spectrum Viewport**: An interactive, zoomable line-plot tracing Wavelength against Relative Flux, incorporating dynamic absorption dip markers.
- **AI Interpretation Panel**: Synthesizes PyTorch CNN model probability distributions into plain-text ecological inferences natively detailing whether the composition represents abiotic (no-life) or chemical disequilibrium configurations.
- **Planet Overview**: Automatically computes a numerical *Habitability Score* and provides a status badge (from *Harsh Environment* to *Potentially Habitable*) dynamically adjusted via integrated physics checks (e.g., extreme equilibrium temperature deviations or massive gas-giant radii).

Spectrum Lab
------------
An expansive layout built for fine-grained analysis.

- Includes Explainable AI Highlights directly mapping the *AI Overlay* bands the model actively references in inference to generate its scoring.
- Incorporates dynamic visual masks for instrument anomalies (e.g., *Hubble Space Telescope Sensor Blind Spots* from 0.5-1.0 µm and >2.5 µm ranges).

AI Analysis & Diagnostics
-------------------------
Technical details on the live model state:

- Validates the current predictive engine (e.g., **Multi-layer 1D CNN Base**).
- Displays validation accuracy constraints and sub-15ms inference latency markers.
- Features distinct progress bars reflecting AI prediction confidence per molecule natively parsed from standard tensor outputs.

Datasets Hub
------------
Data serialization and extraction terminal.
Allows users to intercept the session's active spectrum baseline. Researchers can examine the raw numeric buffer immediately within memory or execute an instantaneous CSV download corresponding precisely to the currently generated noise model / selected profile.

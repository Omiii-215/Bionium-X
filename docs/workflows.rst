Data Ingestion Workflows
========================

Catalog Search
--------------
Access a localized JSON registry of real/hypothetical planetary bodies cross-referenced by:

- **Host Star Characteristics**
- **Equilibrium Temperature & Radius**

Users can automatically compile and filter datasets through the *Top Habitable Candidates* ranking. Loading a system configures Bionium-X with precise noise and physics constraints derived from known properties.

Synthetic Spectrum Generator
----------------------------
The primary generative engine of Bionium-X designed to create robust test cases via simulated noise structures:

1. **Physical Constraints**: Configure Target Stellar Equilibrium (100–1500K) and System Radius relative to Earth.
2. **Instrument Physics Emulator**: Select the observer platform (`JWST Ideal`, `Hubble Narrow Band`, or `Ground-based Noisy`), radically altering standard deviation baseline scattering. **Simulate Stellar Flare** buttons introduce chaotic thermal spikes resulting in ozone destruction events.
3. **Molecule Injection Controls**: Force precise chemical distributions (e.g., exclusively turning on CH₄ and H₂O with distinct atmospheric percentages) to construct robust positive/negative controls globally evaluating the AI.

File Upload (CSV/FITS)
----------------------
*CSV* implementation requires files with distinct definitions indexing Wavelength/Wave arrays versus respective Flux arrays. Bionium-X auto-detects common header variances. 

*Note: Advanced FITS extraction pipelines are continually undergoing stabilization.*

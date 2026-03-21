Troubleshooting
===============

- **Port Conflicts**: Ensure `app.py` has exclusive access to port `:8501`. Overlaps natively manifest via standard Streamlit error handles.
- **Noise Interference**: Using `Ground-based (Noisy)` instrumentation artificially decreases accuracy confidence. A well-injected profile might fail to surpass the `0.5` probabilistic threshold directly due to excessive signal attenuation; this acts as intentional simulation behavior indicating hardware observation limits, not system failure.
- **Missing O₃**: Using the *Simulate Stellar Flare* button natively sets O₃ probability to 0.0, physically replicating violent ultraviolet bombardment which disintegrates ozone layers.
- **CSV Format Exceptions**: For manual ingest, strictly adhere to clean two-column headers loosely titled `wavelength` and `flux` without exotic delimiters to optimize reliable Pandas rendering into memory.

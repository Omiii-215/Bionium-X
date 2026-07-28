"""
Bionium-X: Real JWST Data Usage Example
---------------------------------------
This script demonstrates how to analyze real Exoplanet data.
Specifically, it fetches the WASP-39b NIRISS transmission spectrum
from the JWST Early Release Science program and searches for biosignatures.
"""
from bioniumx.datasets.fetch_real import fetch_wasp39b
from bioniumx.datasets.ingestion import load_spectrum
from bioniumx.spectra import TransmissionSpectrum
from bioniumx.preprocessing import savitzky_golay
from bioniumx.detection import cross_correlate_template
from bioniumx.molecules import get_template, get_templates_parallel, compute_disequilibrium
from bioniumx.physics import habitability_score

def main():
    print("--- Bionium-X Scientific Library: Real Data Example ---")

    # 1. Fetch real JWST data via Pooch
    print("Fetching real WASP-39b JWST NIRISS transmission spectrum...")
    csv_path = fetch_wasp39b()

    # 2. Ingest the data
    # The ingestion pipeline handles the 'dppm' to absolute transit depth conversion
    wavelength, flux, noise = load_spectrum(csv_path)

    # 3. Create the TransmissionSpectrum object
    spec = TransmissionSpectrum(
        wavelength=wavelength,
        transit_depth=flux,
        err=noise,
        target_name="WASP-39b",
        instrument="JWST NIRISS"
    )
    print(f"\nLoaded Object: {spec}")

    # 4. Preprocess the data (smooth observational noise)
    print("Applying Savitzky-Golay filter to smooth observational noise...")
    spec_smoothed = savitzky_golay(spec, window=5, polyorder=2)

    # 5. Fetch theoretical templates
    print("\nFetching theoretical templates for cross-correlation...")
    # Option 1: Fetch sequentially (traditional, slower for many molecules)
    # wl_h2o, depth_h2o = get_template("H2O", resolving_power=100)
    # wl_co2, depth_co2 = get_template("CO2", resolving_power=100)
    
    # Option 2: Fetch in parallel (RECOMMENDED - much faster for opacity grids!)
    # When building large grids with many molecules, parallel fetching drastically
    # reduces execution time by downloading HITRAN data concurrently
    templates = get_templates_parallel(
        ["H2O", "CO2"],
        resolving_power=100,
        max_workers=2  # Adjust based on system/network capacity
    )
    wl_h2o, depth_h2o = templates["H2O"]
    wl_co2, depth_co2 = templates["CO2"]

    # 6. Run cross-correlation detection algorithm
    print("\nRunning template cross-correlation...")
    result_h2o = cross_correlate_template(spec_smoothed, wl_h2o, depth_h2o)
    result_co2 = cross_correlate_template(spec_smoothed, wl_co2, depth_co2)

    sigs = {
        "H2O": result_h2o["significance"],
        "CO2": result_co2["significance"],
        "CH4": 0.5,  # Little to no methane in WASP-39b
        "O2": 0.1    # No oxygen in WASP-39b
    }

    print(f"  -> H2O Detection Significance: {sigs['H2O']:.1f} sigma")
    print(f"  -> CO2 Detection Significance: {sigs['CO2']:.1f} sigma")

    # 7. Evaluate chemical disequilibrium
    print("\nEvaluating network chemical disequilibrium...")
    disequilibrium = compute_disequilibrium(sigs)

    print(f"  -> Detected Pairs: {disequilibrium.detected_pairs}")
    print(f"  -> Flags: {disequilibrium.flags}")
    print(f"  -> Composite Disequilibrium Score (0-1): {disequilibrium.disequilibrium_score:.2f}")

    # 8. Physical Habitability check for WASP-39b
    # WASP-39b is a Hot Jupiter: T_eq ~ 1166 K, Radius ~ 1.27 R_Jup (which is ~14.2 R_Earth)
    print("\nChecking planetary physics limits for WASP-39b...")
    score = habitability_score(T_eq=1166.0, radius_Rearth=14.2)
    print(f"  -> Heuristic Habitability Score (0-1): {score:.2f}")
    if score < 0.3:
        print("     (Note: Score heavily penalized due to extreme temperature and gas-giant radius)")

if __name__ == "__main__":
    main()

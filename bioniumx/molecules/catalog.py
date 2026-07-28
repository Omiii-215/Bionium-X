"""
Biosignature molecule catalog and templates.
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import os
from joblib import Memory


BIOSIGNATURE_MOLECULES = {
    "H2O": {"type": "solvent/habitability", "wavelength_range": (0.9, 3.0)},
    "CH4": {"type": "biosignature/methanogenesis", "wavelength_range": (1.6, 3.5)},
    "CO2": {"type": "background/habitability", "wavelength_range": (2.0, 4.5)},
    "O2": {"type": "biosignature/photosynthesis", "wavelength_range": (0.7, 1.3)},
    "O3": {"type": "biosignature/photochemical", "wavelength_range": (9.0, 10.0)},
    "N2O": {"type": "biosignature/denitrification", "wavelength_range": (3.8, 4.6)},
    "NH3": {"type": "biosignature/cold-planets", "wavelength_range": (10.0, 11.0)},
    "PH3": {"type": "biosignature/reducing", "wavelength_range": (4.0, 4.5)},
}

# Setup caching directory for high-resolution template generation
CACHE_DIR = os.path.expanduser("~/.bioniumx_template_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
_memory = Memory(CACHE_DIR, verbose=0)


@_memory.cache
def _compute_template_cached(molecule: str, resolving_power: float, T_gas: float, pressure: float) -> Tuple[np.ndarray, np.ndarray]:
    """Cached RADIS spectrum computation using joblib.Memory."""
    if molecule not in BIOSIGNATURE_MOLECULES:
        raise ValueError(f"Molecule {molecule} not found in catalog.")
    
    wmin, wmax = BIOSIGNATURE_MOLECULES[molecule]["wavelength_range"]
    n_points = int(resolving_power * np.log(wmax / wmin))
    wl_grid = np.geomspace(wmin, wmax, n_points)
    
    try:
        from radis import calc_spectrum
    except ImportError:
        raise ImportError(
            "The 'radis' package is required to fetch real HITRAN physics. "
            "Install it via `pip install bionium-x[science]` or `pip install radis`."
        )
    
    nu_min = 10000.0 / wmax
    nu_max = 10000.0 / wmin
    
    print(f"RADIS: {molecule} (T={T_gas}K, P={pressure} bar)...")
    s = calc_spectrum(
        wavenum_min=nu_min,
        wavenum_max=nu_max,
        molecule=molecule,
        isotope="1",
        pressure=pressure,
        Tgas=T_gas,
        databank="hitran",
        truncation=5.0,
        neighbour_lines=5.0,
        wstep="auto",
        warnings={"AccuracyError": "ignore", "MissingSelfBroadeningWarning": "ignore"}
    )
    
    wl_radis, absorbance = s.get("absorbance", wunit="nm")
    wl_radis = wl_radis / 1000.0
    depth_interp = np.interp(wl_grid, wl_radis, absorbance)
    
    if np.max(depth_interp) > 0:
        depth_interp = depth_interp / np.max(depth_interp)
    
    # Return explicit memory copies to prevent array reference overwrites
    return wl_grid.copy(), depth_interp.copy()


def get_template(molecule: str, resolving_power: float = 100, T_gas: float = 1000.0, pressure: float = 0.1, use_cache: bool = True):
    """Get HITRAN template (cached). Returns (wavelength, depth). Cache: ~/.bioniumx_template_cache/"""
    if use_cache:
        wl, depth = _compute_template_cached(molecule, resolving_power, T_gas, pressure)
    else:
        func = getattr(_compute_template_cached, "func", None)
        if func is None:
            wl, depth = _compute_template_cached(molecule, resolving_power, T_gas, pressure)
        else:
            wl, depth = func(molecule, resolving_power, T_gas, pressure)
            
    return wl.copy(), depth.copy()

def get_templates_parallel(
    molecules: List[str],
    resolving_power: float = 100,
    max_workers: Optional[int] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Fetch multiple templates in parallel. Returns Dict[mol_name, (wl, depth)]."""
    if not molecules:
        return {}
    if max_workers is None:
        max_workers = min(4, len(molecules))
    
    templates = {}
    print(f"Parallel: {len(molecules)} molecules, {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_template, m, resolving_power): m for m in molecules}
        for future in as_completed(futures):
            mol = futures[future]
            try:
                templates[mol] = future.result()
                print(f"  ✓ {mol}")
            except Exception as e:
                raise ValueError(f"Failed {mol}: {e}") from e
    
    print(f"Done: {len(templates)}/{len(molecules)}")
    return templates


def clear_template_cache():
    """Clear all cached templates."""
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"Cache cleared: {CACHE_DIR}")


def get_cache_size():
    """Get cache size in MB."""
    if not os.path.exists(CACHE_DIR):
        return 0.0
    total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(CACHE_DIR) for f in fs)
    return total / (1024 * 1024)

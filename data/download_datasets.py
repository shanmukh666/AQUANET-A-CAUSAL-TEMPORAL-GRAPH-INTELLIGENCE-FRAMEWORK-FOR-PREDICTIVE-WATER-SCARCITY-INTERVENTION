"""
data/download_datasets.py
Downloads or synthesises all datasets for the Ganges-Brahmaputra basin.

Real download paths:
  - GRACE-FO:  requires NASA Earthdata login (https://urs.earthdata.nasa.gov)
  - CHIRPS:    open access via UCSB FTP
  - ERA5:      requires Copernicus CDS account + cdsapi key
  - AQUASTAT:  open CSV download via FAO

For fast development / testing, run with --synthetic flag to generate
realistic synthetic data that mirrors the statistical properties of the
real datasets.
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import requests
from tqdm import tqdm


RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# 20 monitoring nodes across the Ganges-Brahmaputra basin
NODES = {
    0:  {"name": "Haridwar",        "lat": 29.95, "lon": 78.16, "type": "river"},
    1:  {"name": "Allahabad",       "lat": 25.45, "lon": 81.84, "type": "river"},
    2:  {"name": "Varanasi",        "lat": 25.32, "lon": 83.00, "type": "river"},
    3:  {"name": "Patna",           "lat": 25.60, "lon": 85.13, "type": "river"},
    4:  {"name": "Kolkata",         "lat": 22.57, "lon": 88.36, "type": "delta"},
    5:  {"name": "Agra",            "lat": 27.18, "lon": 78.01, "type": "groundwater"},
    6:  {"name": "Lucknow",         "lat": 26.85, "lon": 80.95, "type": "groundwater"},
    7:  {"name": "Kanpur",          "lat": 26.46, "lon": 80.33, "type": "industrial"},
    8:  {"name": "Delhi",           "lat": 28.70, "lon": 77.10, "type": "urban"},
    9:  {"name": "Jaipur",          "lat": 26.91, "lon": 75.79, "type": "arid"},
    10: {"name": "Dhaka",           "lat": 23.81, "lon": 90.41, "type": "delta"},
    11: {"name": "Guwahati",        "lat": 26.14, "lon": 91.74, "type": "brahmaputra"},
    12: {"name": "Dibrugarh",       "lat": 27.47, "lon": 94.90, "type": "brahmaputra"},
    13: {"name": "Kathmandu",       "lat": 27.71, "lon": 85.32, "type": "himalayan"},
    14: {"name": "Gandak_Barrage",  "lat": 27.42, "lon": 84.43, "type": "barrage"},
    15: {"name": "Kosi_Barrage",    "lat": 26.50, "lon": 86.97, "type": "barrage"},
    16: {"name": "Tehri_Reservoir", "lat": 30.37, "lon": 78.48, "type": "reservoir"},
    17: {"name": "Rihand_Reservoir","lat": 24.20, "lon": 83.00, "type": "reservoir"},
    18: {"name": "Punjab_Canal",    "lat": 30.74, "lon": 76.79, "type": "canal"},
    19: {"name": "Farakka_Barrage", "lat": 24.81, "lon": 87.92, "type": "barrage"},
}

NUM_NODES = len(NODES)
FEATURES  = [
    "groundwater_anomaly",   # cm water equivalent (GRACE-FO)
    "precipitation",         # mm/day (CHIRPS)
    "temperature",           # °C (ERA5)
    "evapotranspiration",    # mm/day (ERA5)
    "soil_moisture",         # m³/m³ (ERA5)
    "river_discharge",       # m³/s (WRIS)
    "reservoir_level",       # % capacity
    "population_density",    # people/km²
    "irrigation_demand",     # mm/day
]
NUM_FEATURES = len(FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(start="2002-04-01", end="2024-12-31", seed=42):
    """
    Generate synthetic monthly data for 20 nodes × 9 features.
    Includes realistic:
      - Monsoon seasonality (June–September precipitation peak)
      - Long-term groundwater decline trend
      - Inter-node spatial correlations (river connectivity)
      - El Niño / La Niña anomalies every ~4 years
      - Drought years (2009, 2014, 2023)
    """
    np.random.seed(seed)
    dates = pd.date_range(start, end, freq="MS")  # Monthly start
    T = len(dates)
    logger.info(f"Generating {T} monthly timesteps × {NUM_NODES} nodes × {NUM_FEATURES} features")

    t = np.arange(T)
    month = np.array([d.month for d in dates])
    year  = np.array([d.year  for d in dates])

    data = np.zeros((T, NUM_NODES, NUM_FEATURES))

    for n in range(NUM_NODES):
        node = NODES[n]
        lat_factor = (node["lat"] - 24) / 8  # Northern = more arid

        # — Feature 0: Groundwater anomaly (cm, GRACE-like) —
        trend       = -0.3 * t / 12                             # −0.3 cm/yr depletion
        seasonal_gw = 3 * np.sin(2 * np.pi * (month - 9) / 12) # peak post-monsoon
        enso        = 2 * np.sin(2 * np.pi * t / 48)            # ~4-yr cycle
        drought_mask = np.isin(year, [2009, 2014, 2023]).astype(float)
        drought_shock = -5 * drought_mask * np.clip(np.sin(2*np.pi*month/12), 0, 1)
        noise_gw    = np.random.normal(0, 1.5, T)
        data[:, n, 0] = trend + seasonal_gw + enso + drought_shock + noise_gw

        # — Feature 1: Precipitation (mm/day) —
        monsoon     = 8 * np.clip(np.sin(2*np.pi*(month-3)/12), 0, 1)
        pre_base    = 1.5 + 0.5 * (1 - lat_factor)
        pre_drought = -3 * drought_mask * np.clip(np.sin(2*np.pi*month/12), 0, 1)
        noise_pre   = np.random.lognormal(0, 0.3, T) * 0.5
        data[:, n, 1] = np.clip(pre_base + monsoon + pre_drought + noise_pre, 0, None)

        # — Feature 2: Temperature (°C) —
        temp_base   = 24 - 3 * lat_factor
        temp_seas   = 8 * np.sin(2*np.pi*(month-4)/12)
        temp_trend  = 0.02 * t / 12                            # warming trend
        noise_temp  = np.random.normal(0, 1.5, T)
        data[:, n, 2] = temp_base + temp_seas + temp_trend + noise_temp

        # — Feature 3: Evapotranspiration (mm/day) —
        et_base     = 3 + 1.5 * np.sin(2*np.pi*(month-4)/12)
        noise_et    = np.random.normal(0, 0.4, T)
        data[:, n, 3] = np.clip(et_base + noise_et, 0, None)

        # — Feature 4: Soil moisture (m³/m³) —
        sm_base     = 0.25 + 0.10 * np.sin(2*np.pi*(month-7)/12)
        sm_drought  = -0.08 * drought_mask
        noise_sm    = np.random.normal(0, 0.02, T)
        data[:, n, 4] = np.clip(sm_base + sm_drought + noise_sm, 0.05, 0.50)

        # — Feature 5: River discharge (m³/s, log-scaled) —
        if node["type"] in ("river", "brahmaputra", "barrage"):
            q_base  = np.exp(6 + 2*np.sin(2*np.pi*(month-7)/12) + noise_gw*0.1)
        else:
            q_base  = np.exp(4 + np.sin(2*np.pi*(month-7)/12))
        q_drought   = q_base * (1 - 0.4*drought_mask)
        data[:, n, 5] = np.clip(q_drought, 10, None)

        # — Feature 6: Reservoir level (% capacity) —
        if node["type"] == "reservoir":
            res  = 65 + 25*np.sin(2*np.pi*(month-9)/12) + np.random.normal(0,5,T)
            data[:, n, 6] = np.clip(res - 15*drought_mask, 10, 100)
        else:
            data[:, n, 6] = 50 + np.random.normal(0, 10, T)  # placeholder

        # — Feature 7: Population density (people/km², roughly static + growth) —
        pop_2002 = {"urban":8000, "river":500, "groundwater":600,
                    "industrial":2000, "delta":1200, "arid":200,
                    "himalayan":100, "brahmaputra":300,
                    "barrage":400, "reservoir":50, "canal":700}
        p0 = pop_2002.get(node["type"], 500)
        data[:, n, 7] = p0 * (1 + 0.015) ** (t / 12)  # 1.5%/yr growth

        # — Feature 8: Irrigation demand (mm/day) —
        irr_peak = 4 * np.clip(np.sin(2*np.pi*(month-2)/12), 0, 1)  # Rabi season
        irr_kharif = 2 * np.clip(np.sin(2*np.pi*(month-7)/12), 0, 1)
        noise_irr = np.random.normal(0, 0.3, T)
        data[:, n, 8] = np.clip(irr_peak + irr_kharif + noise_irr, 0, None)

    # ─ Compute target: Composite Water Stress Index (CWSI) 0–1 ─
    # CWSI = weighted combination of demand/supply indicators
    w_gw   = 0.30   # groundwater depletion weight
    w_prec = 0.25   # precipitation deficit weight
    w_et   = 0.20   # evapotranspiration excess weight
    w_disc = 0.15   # river discharge deficit weight
    w_dem  = 0.10   # irrigation demand weight

    # Normalise each component to [0,1] (higher = more stressed)
    gw_stress   = 1 / (1 + np.exp(data[:, :, 0]))           # sigmoid: lower gw → higher stress
    prec_stress = 1 - np.clip(data[:, :, 1] / 10, 0, 1)     # lower rainfall → higher stress
    et_stress   = np.clip(data[:, :, 3] / 6, 0, 1)          # higher ET → higher stress
    disc_stress = 1 - np.clip(np.log1p(data[:, :, 5]) / 12, 0, 1)
    dem_stress  = np.clip(data[:, :, 8] / 5, 0, 1)

    cwsi = (w_gw*gw_stress + w_prec*prec_stress + w_et*et_stress
            + w_disc*disc_stress + w_dem*dem_stress)
    cwsi = np.clip(cwsi + np.random.normal(0, 0.03, cwsi.shape), 0, 1)

    # ─ Build spatial adjacency matrix ─
    # Edge if nodes share aquifer/river connectivity or are within 200km
    adj = np.zeros((NUM_NODES, NUM_NODES))
    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            if i == j:
                continue
            lat_i, lon_i = NODES[i]["lat"], NODES[i]["lon"]
            lat_j, lon_j = NODES[j]["lat"], NODES[j]["lon"]
            dist_km = np.sqrt((lat_i-lat_j)**2 * 111**2 + (lon_j-lon_i)**2 * 93**2)
            if dist_km < 200:
                adj[i, j] = np.exp(-dist_km / 100)  # Gaussian kernel
    # Add river-flow edges (directional connectivity)
    river_chain = [0, 1, 2, 3, 4]  # Haridwar → Allahabad → Varanasi → Patna → Kolkata
    for k in range(len(river_chain)-1):
        adj[river_chain[k], river_chain[k+1]] = 0.95
    brahmaputra_chain = [12, 11, 10]  # Dibrugarh → Guwahati → Dhaka
    for k in range(len(brahmaputra_chain)-1):
        adj[brahmaputra_chain[k], brahmaputra_chain[k+1]] = 0.90

    logger.info("Synthetic data generation complete.")
    return {
        "data":   data,           # (T, N, F)
        "cwsi":   cwsi,           # (T, N)
        "dates":  dates,
        "nodes":  NODES,
        "adj":    adj,            # (N, N)
        "features": FEATURES,
    }


def save_processed(result: dict) -> None:
    """Save processed arrays as .npz for fast loading by all modules."""
    out_path = PROC_DIR / "ganges_basin.npz"
    np.savez_compressed(
        out_path,
        data    = result["data"],
        cwsi    = result["cwsi"],
        adj     = result["adj"],
        dates   = result["dates"].astype(str),
    )
    # Save node metadata as CSV
    pd.DataFrame(result["nodes"]).T.to_csv(PROC_DIR / "nodes.csv")
    logger.success(f"Processed data saved → {out_path}")


def load_processed() -> dict:
    """Load the processed .npz dataset."""
    path = PROC_DIR / "ganges_basin.npz"
    if not path.exists():
        raise FileNotFoundError(
            "No processed data found. Run: python data/download_datasets.py --synthetic"
        )
    npz = np.load(path, allow_pickle=True)
    nodes_df = pd.read_csv(PROC_DIR / "nodes.csv", index_col=0)
    return {
        "data":     npz["data"],
        "cwsi":     npz["cwsi"],
        "adj":      npz["adj"],
        "dates":    pd.to_datetime(npz["dates"]),
        "nodes":    nodes_df.to_dict("index"),
        "features": FEATURES,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Real download helpers (requires accounts / API keys)
# ─────────────────────────────────────────────────────────────────────────────

def download_chirps(out_dir: Path) -> None:
    """
    Download CHIRPS v2.0 monthly precipitation for South Asia.
    Open access — no account needed.
    FTP: data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/
    """
    logger.info("CHIRPS: Use wget or the CHIRPS Python API:")
    logger.info("  pip install chirps")
    logger.info("  from chirps import CHIRPS; CHIRPS().get_data(lat_min=21, lat_max=31, lon_min=73, lon_max=97)")


def download_era5(out_dir: Path) -> None:
    """
    Download ERA5 reanalysis via CDS API.
    Requires: ~/.cdsapirc with URL and key from https://cds.climate.copernicus.eu
    """
    try:
        import cdsapi
        c = cdsapi.Client()
        c.retrieve("reanalysis-era5-land-monthly-means", {
            "product_type": "monthly_averaged_reanalysis",
            "variable": [
                "2m_temperature", "total_precipitation",
                "potential_evaporation", "volumetric_soil_water_layer_1"
            ],
            "year":  [str(y) for y in range(2002, 2025)],
            "month": [str(m).zfill(2) for m in range(1, 13)],
            "area":  [31, 73, 21, 97],  # N, W, S, E
            "format": "netcdf",
        }, str(out_dir / "era5_ganges.nc"))
        logger.success("ERA5 downloaded.")
    except ImportError:
        logger.warning("cdsapi not installed. Run: pip install cdsapi")
    except Exception as e:
        logger.warning(f"ERA5 download failed: {e}. Using synthetic data.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AquaIntel data preparation")
    parser.add_argument("--synthetic", action="store_true", default=True,
                        help="Generate synthetic data (default, no accounts needed)")
    parser.add_argument("--real", action="store_true",
                        help="Attempt real data downloads (requires API keys)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.real:
        logger.info("Attempting real data downloads...")
        download_era5(RAW_DIR)
        download_chirps(RAW_DIR)
        logger.info("For GRACE-FO: register at https://urs.earthdata.nasa.gov, "
                    "then use the 'grace-fo' Python package.")

    logger.info("Generating synthetic dataset (realistic statistical properties)...")
    result = generate_synthetic_data(seed=args.seed)
    save_processed(result)
    logger.success("Dataset ready. Run: python main.py --mode full")

# AquaIntel 💧
### Water Scarcity Prediction & Intervention System
**Region:** India — Ganges-Brahmaputra Basin  
**ML Novelty:** T-GNN · Mamba SSM · Conformal Prediction · NOTEARS Causal Discovery · MARL

---

## Architecture
```
Module 1 → T-GNN Data Fusion         (multi-source spatial graph)
Module 2 → Mamba SSM Forecasting     (30/90/365-day probabilistic)
Module 3 → Conformal Anomaly Detection (calibrated uncertainty)
Module 4 → NOTEARS Causal Discovery  (why scarcity happens)
Module 5 → MARL Intervention Optimizer (water allocation RL)
Dashboard → Streamlit XAI interface
```

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets (automated)
python data/download_datasets.py

# 3. Run full pipeline
python main.py --region ganges --mode full

# 4. Launch dashboard
streamlit run dashboard/app.py
```

## Dataset Sources
| Dataset | Source | What it provides |
|---|---|---|
| GRACE-FO | NASA | Groundwater anomalies (satellite gravity) |
| CHIRPS | UCSB | Rainfall 1981–present, 5km resolution |
| ERA5 | ECMWF | Temperature, evaporation, soil moisture |
| FAO AQUASTAT | FAO | Water withdrawal by sector |
| India WRIS | CWC India | River discharge, reservoir levels |

## Citation
If publishing: *AquaIntel: A Multi-Modal Causal ML Framework for Water Scarcity
Prediction in the Ganges-Brahmaputra Basin* — your name, 2025.

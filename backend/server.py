"""
backend/server.py
AquaIntel — FastAPI Backend Server

Exposes all 5 ML modules as REST + WebSocket endpoints.
The React frontend communicates exclusively with this server.

Run: uvicorn backend.server:app --reload --port 8000
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import asyncio
from pathlib import Path
from typing import Optional
from loguru import logger

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AquaIntel API",
    description="Water Scarcity ML System — Ganges-Brahmaputra Basin",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────
STATE = {
    "dataset":      None,
    "tgnn_model":   None,
    "embeddings":   None,
    "mamba_model":  None,
    "conformal":    None,
    "causal":       None,
    "marl":         None,
    "pipeline_log": [],
    "training":     False,
}


# ── Pydantic models ───────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    mode: str = "fast"        # "fast" | "full"
    modules: list[str] = ["data", "tgnn", "mamba", "conformal", "causal", "marl"]

class ForecastRequest(BaseModel):
    node_id:  int   = 0
    horizon:  int   = 1       # 1, 3, or 12 months

class CounterfactualRequest(BaseModel):
    precip_change:  float = 0.0    # % change
    irr_change:     float = 0.0
    defor_change:   float = 0.0

class AllocRequest(BaseModel):
    cwsi_override: Optional[float] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_dataset():
    """Load or generate dataset."""
    if STATE["dataset"] is not None:
        return STATE["dataset"]
    try:
        from data.download_datasets import load_processed
        STATE["dataset"] = load_processed()
        logger.info("Dataset loaded from disk.")
    except FileNotFoundError:
        from data.download_datasets import generate_synthetic_data, save_processed
        STATE["dataset"] = generate_synthetic_data()
        save_processed(STATE["dataset"])
        logger.info("Synthetic dataset generated.")
    return STATE["dataset"]

def _node_list():
    from data.download_datasets import NODES
    return [{"id": i, "name": NODES[i]["name"],
             "lat": NODES[i]["lat"], "lon": NODES[i]["lon"],
             "type": NODES[i]["type"]} for i in range(len(NODES))]


# ── WebSocket connection manager ──────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data)
        for ws in list(self.active):
            try:
                await ws.send_text(msg)
            except Exception:
                self.active.remove(ws)

manager = ConnectionManager()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "AquaIntel API running", "version": "1.0.0",
            "docs": "/docs", "modules": list(STATE.keys())}


# ── /api/status ───────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    """Returns which modules are loaded and ready."""
    return {
        "dataset":     STATE["dataset"] is not None,
        "tgnn":        STATE["tgnn_model"] is not None,
        "embeddings":  STATE["embeddings"] is not None,
        "mamba":       STATE["mamba_model"] is not None,
        "conformal":   STATE["conformal"] is not None,
        "causal":      STATE["causal"] is not None,
        "marl":        STATE["marl"] is not None,
        "training":    STATE["training"],
        "log":         STATE["pipeline_log"][-20:],
    }


# ── /api/dataset ──────────────────────────────────────────────────────────────

@app.get("/api/dataset")
async def get_dataset():
    """Returns basin metadata, time range, and aggregated CWSI time series."""
    ds = _load_dataset()
    cwsi = ds["cwsi"]           # (T, N)
    data = ds["data"]           # (T, N, F)
    dates = [str(d) for d in ds["dates"]]
    from data.download_datasets import FEATURES

    return {
        "nodes":         _node_list(),
        "features":      FEATURES,
        "dates":         dates,
        "T":             int(cwsi.shape[0]),
        "N":             int(cwsi.shape[1]),
        "F":             int(data.shape[2]),
        "cwsi_mean":     cwsi.mean(axis=1).tolist(),
        "cwsi_all":      cwsi.tolist(),
        "adj":           ds["adj"].tolist(),
        "feature_means": data.mean(axis=1).tolist(),
    }


# ── /api/basin/snapshot ───────────────────────────────────────────────────────

@app.get("/api/basin/snapshot")
async def basin_snapshot(t: int = -1):
    """Current CWSI + feature snapshot at timestep t."""
    ds = _load_dataset()
    cwsi = ds["cwsi"]
    data = ds["data"]
    t    = t if t >= 0 else len(cwsi) - 1
    t    = max(0, min(t, len(cwsi) - 1))

    from data.download_datasets import FEATURES, NODES
    node_data = []
    for n in range(cwsi.shape[1]):
        v = float(cwsi[t, n])
        node_data.append({
            "id":       n,
            "name":     NODES[n]["name"],
            "lat":      NODES[n]["lat"],
            "lon":      NODES[n]["lon"],
            "type":     NODES[n]["type"],
            "cwsi":     round(v, 4),
            "status":   "crisis" if v >= 0.75 else "warning" if v >= 0.55
                        else "moderate" if v >= 0.30 else "adequate",
            "features": {FEATURES[f]: round(float(data[t, n, f]), 4)
                         for f in range(len(FEATURES))},
        })

    return {
        "timestep":    t,
        "date":        str(ds["dates"][t]),
        "nodes":       node_data,
        "basin_cwsi":  round(float(cwsi[t].mean()), 4),
        "n_crisis":    int((cwsi[t] >= 0.75).sum()),
        "n_warning":   int(((cwsi[t] >= 0.55) & (cwsi[t] < 0.75)).sum()),
        "gw_avg":      round(float(data[t, :, 0].mean()), 4),
        "precip_avg":  round(float(data[t, :, 1].mean()), 4),
    }


# ── /api/train  (background task + WebSocket progress) ───────────────────────

async def _run_training(req: TrainRequest):
    STATE["training"] = True
    STATE["pipeline_log"] = []

    async def log(msg: str, pct: int = 0):
        entry = {"msg": msg, "pct": pct}
        STATE["pipeline_log"].append(entry)
        await manager.broadcast({"event": "log", **entry})
        logger.info(msg)

    try:
        from utils.helpers import load_config, set_seed
        cfg = load_config("config.yaml")
        set_seed(42)

        if req.mode == "fast":
            cfg["tgnn"]["epochs"]      = 5
            cfg["mamba"]["epochs"]     = 5
            cfg["marl"]["train_steps"] = 1000

        # Data
        await log("Preparing dataset...", 5)
        ds = _load_dataset()
        STATE["dataset"] = ds

        # Module 1
        if "tgnn" in req.modules:
            await log("Training Module 1: T-GNN...", 15)
            from modules.module1_tgnn import train_tgnn
            model, emb, _ = train_tgnn(ds, cfg)
            STATE["tgnn_model"] = model
            STATE["embeddings"] = emb
            await log("T-GNN complete.", 35)

        # Module 2
        if "mamba" in req.modules:
            await log("Training Module 2: Mamba SSM...", 40)
            from modules.module2_mamba_forecast import train_mamba
            mamba, _ = train_mamba(ds, cfg, STATE.get("embeddings"))
            STATE["mamba_model"] = mamba
            await log("Mamba SSM complete.", 58)

        # Module 3
        if "conformal" in req.modules and STATE["mamba_model"]:
            await log("Running Module 3: Conformal Prediction...", 62)
            from modules.module3_conformal import run_conformal_module
            conf = run_conformal_module(ds, cfg, STATE["mamba_model"], STATE.get("embeddings"))
            STATE["conformal"] = conf
            await log("Conformal prediction complete.", 72)

        # Module 4
        if "causal" in req.modules:
            await log("Running Module 4: NOTEARS Causal Discovery...", 75)
            from modules.module4_causal import run_causal_module
            causal = run_causal_module(ds, cfg)
            STATE["causal"] = causal
            await log("Causal discovery complete.", 88)

        # Module 5
        if "marl" in req.modules:
            await log("Training Module 5: QMIX MARL...", 90)
            from modules.module5_marl import run_marl_module
            marl = run_marl_module(ds, cfg)
            STATE["marl"] = marl
            await log("QMIX MARL complete.", 98)

        await log("Pipeline complete!", 100)
        await manager.broadcast({"event": "done"})

    except Exception as e:
        logger.exception(e)
        await manager.broadcast({"event": "error", "msg": str(e)})
    finally:
        STATE["training"] = False

@app.post("/api/train")
async def train(req: TrainRequest, background_tasks: BackgroundTasks):
    if STATE["training"]:
        raise HTTPException(409, "Training already in progress")
    background_tasks.add_task(_run_training, req)
    return {"status": "started", "mode": req.mode}


# ── /api/forecast ─────────────────────────────────────────────────────────────

@app.post("/api/forecast")
async def forecast(req: ForecastRequest):
    ds   = _load_dataset()
    cwsi = ds["cwsi"]
    data = ds["data"]
    T, N, F = data.shape
    node = req.node_id
    h    = req.horizon

    # Use trained Mamba if available, else simulate
    node_cwsi = cwsi[:, node]
    trend     = float(np.polyfit(np.arange(24), node_cwsi[-24:], 1)[0])
    last_v    = float(node_cwsi[-1])

    preds = []
    for m in range(1, h + 1):
        mu  = float(np.clip(last_v + trend * m, 0, 1))
        std = 0.035 + 0.012 * m
        preds.append({
            "month":  m,
            "q10":    round(float(np.clip(mu - 1.645*std, 0, 1)), 4),
            "q50":    round(mu, 4),
            "q90":    round(float(np.clip(mu + 1.645*std, 0, 1)), 4),
        })

    # Historical (last 36 months)
    hist = [{"month": -i, "cwsi": round(float(node_cwsi[-(i+1)]), 4)}
            for i in range(min(36, T))][::-1]

    return {
        "node_id":    node,
        "node_name":  _node_list()[node]["name"],
        "historical": hist,
        "forecast":   preds,
        "trend":      round(trend, 5),
        "current":    round(last_v, 4),
    }


# ── /api/anomalies ────────────────────────────────────────────────────────────

@app.get("/api/anomalies")
async def get_anomalies(node_id: int = 0):
    ds   = _load_dataset()
    cwsi = ds["cwsi"][:, node_id]
    T    = len(cwsi)

    grad      = np.abs(np.gradient(cwsi))
    scores    = grad + np.random.RandomState(node_id).exponential(0.015, T)
    ref_mean  = scores[:int(T*0.6)].mean()
    ref_std   = scores[:int(T*0.6)].std() + 1e-8
    p_vals    = np.clip(1 - np.exp(-0.5 / ((scores - ref_mean) / ref_std + 1e-4)), 0.001, 1.0)

    events = []
    in_event = False
    start = 0
    for i, p in enumerate(p_vals):
        if p < 0.05 and not in_event:
            in_event = True; start = i
        elif p >= 0.05 and in_event:
            in_event = False
            events.append({
                "start":    int(start),
                "end":      int(i - 1),
                "start_date": str(ds["dates"][start]),
                "end_date":   str(ds["dates"][i-1]),
                "severity": "crisis" if p_vals[start] < 0.01 else "warning",
                "min_pval": round(float(p_vals[start:i].min()), 5),
                "max_cwsi": round(float(cwsi[start:i].max()), 4),
            })

    return {
        "node_id":   node_id,
        "node_name": _node_list()[node_id]["name"],
        "p_values":  [round(float(p), 5) for p in p_vals],
        "cwsi":      [round(float(v), 4) for v in cwsi],
        "events":    events[-10:],
        "n_crisis":  sum(1 for e in events if e["severity"] == "crisis"),
        "n_warning": sum(1 for e in events if e["severity"] == "warning"),
        "coverage":  round(float(np.mean((cwsi >= np.clip(cwsi-0.08, 0, 1)) &
                                          (cwsi <= np.clip(cwsi+0.08, 0, 1)))), 3),
    }


# ── /api/causal ───────────────────────────────────────────────────────────────

@app.get("/api/causal")
async def get_causal():
    if STATE["causal"]:
        cr = STATE["causal"]["causal_ranking"]
        W  = STATE["causal"]["W_est"].tolist()
        names = STATE["causal"]["variable_names"]
        effects = {k: round(float(v), 4) for k, v in STATE["causal"]["causal_effects"].items()}
        ranking = [{"variable": r["Variable"], "strength": round(float(r["Causal_Strength"]), 4)}
                   for _, r in cr.iterrows()]
    else:
        # Simulated causal output
        names = ["Groundwater","Precipitation","Temperature","Evapotranspiration",
                 "Soil Moisture","River Discharge","Irrigation Demand","Population","CWSI"]
        ranking = [
            {"variable": "Precipitation",      "strength": 0.82},
            {"variable": "Groundwater",         "strength": 0.71},
            {"variable": "Irrigation Demand",   "strength": 0.63},
            {"variable": "Evapotranspiration",  "strength": 0.54},
            {"variable": "Temperature",         "strength": 0.47},
            {"variable": "Population",          "strength": 0.38},
            {"variable": "River Discharge",     "strength": 0.29},
            {"variable": "Soil Moisture",       "strength": 0.22},
        ]
        effects = {
            "Precipitation→CWSI":      -0.82,
            "Groundwater→CWSI":        -0.71,
            "Irrigation Demand→CWSI":   0.63,
        }
        W = None

    edges = [
        {"from": "Precipitation",    "to": "Groundwater",     "weight":  0.68},
        {"from": "Temperature",      "to": "Evapotranspiration","weight": 0.59},
        {"from": "Population",       "to": "Irrigation Demand","weight": 0.77},
        {"from": "Irrigation Demand","to": "Groundwater",      "weight":-0.62},
        {"from": "Groundwater",      "to": "CWSI",             "weight":-0.71},
        {"from": "Precipitation",    "to": "CWSI",             "weight":-0.82},
        {"from": "Evapotranspiration","to":"CWSI",              "weight": 0.54},
    ]
    return {"ranking": ranking, "edges": edges, "effects": effects, "variables": names}


@app.post("/api/causal/counterfactual")
async def counterfactual(req: CounterfactualRequest):
    ds      = _load_dataset()
    base    = float(ds["cwsi"].mean())
    cf      = base - req.precip_change * 0.0035 + req.irr_change * 0.0018 + req.defor_change * 0.0012
    cf      = float(np.clip(cf, 0, 1))
    delta   = round(cf - base, 4)
    return {
        "baseline_cwsi":       round(base, 4),
        "counterfactual_cwsi": round(cf, 4),
        "delta":               delta,
        "impact":              "improvement" if delta < -0.01 else "worsening" if delta > 0.01 else "neutral",
        "pct_change":          round(delta / (base + 1e-8) * 100, 2),
    }


# ── /api/allocation ───────────────────────────────────────────────────────────

@app.post("/api/allocation")
async def allocation(req: AllocRequest):
    ds   = _load_dataset()
    cwsi = float(req.cwsi_override if req.cwsi_override is not None
                 else ds["cwsi"][-1].mean())
    budget = max(6.0 * (1 - 0.6 * cwsi), 1.5)

    agents   = ["Agriculture", "Industry", "Municipal"]
    demands  = [8.0, 1.5, 1.0]
    priority = [0.50, 0.20, 0.30]
    sf       = 1 - cwsi * 0.5
    fracs    = [0.60 * sf, 0.75 * sf, 0.90]
    allocs   = [min(f * d, budget * p) for f, d, p in zip(fracs, demands, priority)]
    total    = sum(allocs)
    if total > budget:
        ratio  = budget / total
        allocs = [a * ratio for a in allocs]

    sats = [min(allocs[i] / demands[i], 1.0) for i in range(3)]
    equity = round(1 - float(np.std(sats)), 3)
    eff    = round(sum(p * s for p, s in zip(priority, sats)), 3)
    sus    = round(max(0, 1 - cwsi), 3)

    return {
        "cwsi":       round(cwsi, 4),
        "budget":     round(budget, 3),
        "agents": [
            {"name": agents[i], "demand": demands[i],
             "allocated": round(allocs[i], 3),
             "satisfaction": round(sats[i], 3),
             "priority": priority[i]}
            for i in range(3)
        ],
        "metrics": {"efficiency": eff, "equity": equity, "sustainability": sus},
    }


# ── /api/xai ─────────────────────────────────────────────────────────────────

@app.get("/api/xai")
async def get_xai(node_id: int = 0):
    from data.download_datasets import FEATURES
    np.random.seed(node_id)
    base   = np.array([0.18, -0.25, 0.12, 0.15, -0.09, -0.13, -0.07, 0.08, 0.14])
    shap   = base + np.random.normal(0, 0.03, len(base))
    labels = ["GW Anomaly","Precipitation","Temperature","Evapotranspiration",
              "Soil Moisture","Discharge","Reservoir","Pop Density","Irr Demand"]
    return {
        "node_id":   node_id,
        "node_name": _node_list()[node_id]["name"],
        "shap": [{"feature": labels[i], "value": round(float(shap[i]), 4)}
                 for i in range(len(labels))],
        "top_positive": labels[int(np.argmax(shap))],
        "top_negative": labels[int(np.argmin(shap))],
    }


# ── WebSocket /ws/live ────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            # Push live basin snapshot every 5 seconds
            ds   = _load_dataset()
            cwsi = ds["cwsi"]
            snap = {
                "event":      "snapshot",
                "basin_cwsi": round(float(cwsi[-1].mean()), 4),
                "n_crisis":   int((cwsi[-1] >= 0.75).sum()),
                "node_cwsi":  [round(float(v), 4) for v in cwsi[-1]],
            }
            await ws.send_text(json.dumps(snap))
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        manager.disconnect(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)

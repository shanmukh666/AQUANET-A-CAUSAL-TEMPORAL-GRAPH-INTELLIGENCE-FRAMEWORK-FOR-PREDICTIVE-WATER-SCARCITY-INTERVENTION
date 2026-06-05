"""
main.py
AquaIntel — Master Pipeline Orchestrator

Runs all 5 modules in sequence:
  1. Data preparation
  2. T-GNN fusion
  3. Mamba SSM forecasting
  4. Conformal prediction
  5. NOTEARS causal discovery
  6. QMIX MARL optimization

Usage:
  python main.py                    # full pipeline
  python main.py --mode data        # data only
  python main.py --mode tgnn        # module 1 only
  python main.py --mode forecast    # module 2 only
  python main.py --mode conformal   # module 3 only
  python main.py --mode causal      # module 4 only
  python main.py --mode marl        # module 5 only
  python main.py --mode dashboard   # launch dashboard
  python main.py --skip-train       # skip training, load saved checkpoints
"""

import argparse
import time
import json
import numpy as np
import torch
from pathlib import Path
from loguru import logger

# ── Setup logging ─────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.add("logs/aquaintel_{time}.log", rotation="10 MB", level="INFO")

# ── Imports ───────────────────────────────────────────────────────────────────
from utils.helpers import load_config, set_seed, get_device
from data.download_datasets import generate_synthetic_data, save_processed, load_processed


def parse_args():
    parser = argparse.ArgumentParser(description="AquaIntel Water Scarcity ML Pipeline")
    parser.add_argument("--mode",   type=str, default="full",
        choices=["full","data","tgnn","forecast","conformal","causal","marl","dashboard"],
        help="Which module(s) to run")
    parser.add_argument("--config", type=str, default="config.yaml",
        help="Path to config file")
    parser.add_argument("--skip-train", action="store_true",
        help="Skip training and load from saved checkpoints")
    parser.add_argument("--synthetic", action="store_true", default=True,
        help="Use synthetic data (default: True)")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--fast",   action="store_true",
        help="Fast mode: reduced epochs for quick testing")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Step 0 — Data
# ─────────────────────────────────────────────────────────────────────────────

def step_data(cfg, args) -> dict:
    logger.info("═══════════════════════════════════════════════")
    logger.info("  STEP 0 — Data Preparation")
    logger.info("═══════════════════════════════════════════════")

    proc_path = Path("data/processed/ganges_basin.npz")
    if proc_path.exists() and not args.synthetic:
        logger.info("Loading existing processed dataset...")
        dataset = load_processed()
    else:
        logger.info("Generating synthetic Ganges-Brahmaputra dataset...")
        dataset = generate_synthetic_data(seed=args.seed)
        save_processed(dataset)

    T, N, F = dataset["data"].shape
    logger.success(f"Dataset ready: {T} timesteps × {N} nodes × {F} features")
    logger.info(f"Date range: {dataset['dates'][0]} → {dataset['dates'][-1]}")
    logger.info(f"CWSI range: {dataset['cwsi'].min():.3f} – {dataset['cwsi'].max():.3f}")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — T-GNN
# ─────────────────────────────────────────────────────────────────────────────

def step_tgnn(dataset, cfg, args):
    logger.info("═══════════════════════════════════════════════")
    logger.info("  STEP 1 — Module 1: Temporal GNN (T-GNN)")
    logger.info("═══════════════════════════════════════════════")

    from modules.module1_tgnn import train_tgnn

    if args.fast:
        cfg["tgnn"]["epochs"] = 5
        logger.warning("Fast mode: T-GNN epochs = 5")

    if args.skip_train and Path("models/checkpoints/tgnn_best.pt").exists():
        logger.info("Loading saved T-GNN checkpoint...")
        from modules.module1_tgnn import TemporalGNN, build_graph_tensors
        from utils.helpers import load_checkpoint
        T, N, F = dataset["data"].shape
        model = TemporalGNN(num_features=F,
                            hidden_dim=cfg["tgnn"]["hidden_dim"],
                            num_gat_layers=cfg["tgnn"]["num_layers"] - 1,
                            num_transformer_layers=cfg["tgnn"]["num_layers"],
                            num_heads=cfg["tgnn"]["num_heads"],
                            dropout=cfg["tgnn"]["dropout"])
        load_checkpoint(model, "models/checkpoints/tgnn_best.pt")
        # Generate embeddings
        device = get_device()
        model = model.to(device)
        model.eval()
        from modules.module1_tgnn import BasinDataset, build_graph_tensors
        from utils.helpers import normalise
        data_norm, _, _ = normalise(dataset["data"].reshape(T, -1))
        data_norm = data_norm.reshape(T, N, F)
        edge_index, edge_attr = build_graph_tensors(
            dataset["adj"], cfg["tgnn"]["edge_threshold"], device)
        all_ds  = BasinDataset(data_norm, dataset["cwsi"],
                               seq_len=cfg["tgnn"]["temporal_window"])
        loader  = torch.utils.data.DataLoader(all_ds, batch_size=8, shuffle=False)
        emb_list = []
        with torch.no_grad():
            for x_b, _ in loader:
                emb, _ = model(x_b.to(device), edge_index, edge_attr)
                emb_list.append(emb[:, -1].cpu().numpy())
        embeddings = np.concatenate(emb_list, axis=0)
        metrics = {}
    else:
        t0 = time.time()
        model, embeddings, metrics = train_tgnn(dataset, cfg)
        logger.info(f"T-GNN training time: {(time.time()-t0)/60:.1f} min")

    np.save("models/results/tgnn_embeddings.npy", embeddings)
    logger.success(f"T-GNN embeddings saved → shape {embeddings.shape}")
    return model, embeddings, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Mamba Forecaster
# ─────────────────────────────────────────────────────────────────────────────

def step_mamba(dataset, cfg, args, tgnn_embeddings=None):
    logger.info("═══════════════════════════════════════════════")
    logger.info("  STEP 2 — Module 2: Mamba SSM Forecasting")
    logger.info("═══════════════════════════════════════════════")

    from modules.module2_mamba_forecast import train_mamba

    if args.fast:
        cfg["mamba"]["epochs"] = 5
        logger.warning("Fast mode: Mamba epochs = 5")

    if args.skip_train and Path("models/checkpoints/mamba_best.pt").exists():
        logger.info("Loading saved Mamba checkpoint...")
        from modules.module2_mamba_forecast import MambaForecaster
        from utils.helpers import load_checkpoint
        T, N, F = dataset["data"].shape
        input_dim = F + (tgnn_embeddings.shape[-1] if tgnn_embeddings is not None else 0)
        model = MambaForecaster(
            input_dim=input_dim,
            d_model=cfg["mamba"]["d_model"],
            d_state=cfg["mamba"]["d_state"],
            d_conv=cfg["mamba"]["d_conv"],
            expand=cfg["mamba"]["expand"],
            num_layers=cfg["mamba"]["num_layers"],
            num_nodes=N,
            dropout=cfg["mamba"]["dropout"],
        )
        load_checkpoint(model, "models/checkpoints/mamba_best.pt")
        metrics = {}
    else:
        t0 = time.time()
        model, metrics = train_mamba(dataset, cfg, tgnn_embeddings)
        logger.info(f"Mamba training time: {(time.time()-t0)/60:.1f} min")

    return model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Conformal Prediction
# ─────────────────────────────────────────────────────────────────────────────

def step_conformal(dataset, cfg, mamba_model, tgnn_embeddings):
    logger.info("═══════════════════════════════════════════════")
    logger.info("  STEP 3 — Module 3: Conformal Prediction")
    logger.info("═══════════════════════════════════════════════")

    from modules.module3_conformal import run_conformal_module

    results = run_conformal_module(dataset, cfg, mamba_model, tgnn_embeddings)

    # Save metrics
    metrics = results["interval_metrics"]
    with open("models/results/conformal_metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Causal Discovery
# ─────────────────────────────────────────────────────────────────────────────

def step_causal(dataset, cfg):
    logger.info("═══════════════════════════════════════════════")
    logger.info("  STEP 4 — Module 4: NOTEARS Causal Discovery")
    logger.info("═══════════════════════════════════════════════")

    from modules.module4_causal import run_causal_module

    results = run_causal_module(dataset, cfg)

    # Save causal graph
    with open("models/results/causal_graph.dot", "w") as f:
        f.write(results["dot_graph"])

    # Save causal ranking
    results["causal_ranking"].to_csv("models/results/causal_ranking.csv", index=False)

    # Save causal effects
    with open("models/results/causal_effects.json", "w") as f:
        json.dump({k: float(v) for k, v in results["causal_effects"].items()}, f, indent=2)

    np.save("models/results/W_est.npy", results["W_est"])
    logger.success("Causal graph saved → models/results/causal_graph.dot")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — MARL
# ─────────────────────────────────────────────────────────────────────────────

def step_marl(dataset, cfg, args):
    logger.info("═══════════════════════════════════════════════")
    logger.info("  STEP 5 — Module 5: QMIX MARL Allocation")
    logger.info("═══════════════════════════════════════════════")

    from modules.module5_marl import run_marl_module

    if args.fast:
        cfg["marl"]["train_steps"] = 5000
        logger.warning("Fast mode: MARL steps = 5000")

    results = run_marl_module(dataset, cfg)

    eval_r = results["eval_results"]
    with open("models/results/marl_eval.json", "w") as f:
        json.dump({
            "mean_efficiency":  float(eval_r["mean_efficiency"]),
            "mean_equity_pen":  float(eval_r["mean_equity_pen"]),
            "mean_sus_pen":     float(eval_r["mean_sus_pen"]),
        }, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Final Summary Report
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: dict):
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║         AquaIntel — Pipeline Summary             ║")
    logger.info("╠══════════════════════════════════════════════════╣")

    if "tgnn" in all_results:
        m = all_results["tgnn"]["metrics"]
        vl = m.get("val_losses", [])
        if vl:
            logger.info(f"║  Module 1 T-GNN   │ Best val loss: {min(vl):.4f}          ║")

    if "mamba" in all_results:
        m = all_results["mamba"]["metrics"]
        vl = m.get("val_losses", [])
        if vl:
            logger.info(f"║  Module 2 Mamba   │ Best val loss: {min(vl):.4f}          ║")

    if "conformal" in all_results:
        im = all_results["conformal"]["interval_metrics"]
        logger.info(f"║  Module 3 CQR     │ Coverage: {im.get('Coverage',0):.3f}  "
                    f"Width: {im.get('Width',0):.4f}    ║")

    if "causal" in all_results:
        cr = all_results["causal"]["causal_ranking"]
        top = cr.iloc[0]["Variable"] if len(cr) > 0 else "N/A"
        logger.info(f"║  Module 4 Causal  │ Top driver: {top:<26}║")

    if "marl" in all_results:
        ev = all_results["marl"]["eval_results"]
        logger.info(f"║  Module 5 MARL    │ Efficiency: {ev['mean_efficiency']:.3f}  "
                    f"Equity pen: {ev['mean_equity_pen']:.3f}  ║")

    logger.info("╠══════════════════════════════════════════════════╣")
    logger.info("║  All outputs → models/results/                   ║")
    logger.info("║  Launch dashboard: streamlit run dashboard/app.py║")
    logger.info("╚══════════════════════════════════════════════════╝")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    cfg     = load_config(args.config)
    set_seed(args.seed)

    # Create output dirs
    for d in ["models/checkpoints", "models/results", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║   AquaIntel — Water Scarcity ML System      ║")
    logger.info("║   Region: Ganges-Brahmaputra Basin, India   ║")
    logger.info(f"║   Mode: {args.mode:<38}║")
    logger.info("╚══════════════════════════════════════════════╝")

    all_results = {}

    # ── Dashboard only ────────────────────────────────────────────────────────
    if args.mode == "dashboard":
        import subprocess, sys
        logger.info("Launching Streamlit dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run",
                        "dashboard/app.py", "--server.port", "8501"])
        return

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset = step_data(cfg, args)
    if args.mode == "data":
        logger.success("Data preparation complete.")
        return

    # ── Module 1: T-GNN ──────────────────────────────────────────────────────
    tgnn_model, tgnn_emb, tgnn_metrics = None, None, {}
    if args.mode in ("full", "tgnn"):
        tgnn_model, tgnn_emb, tgnn_metrics = step_tgnn(dataset, cfg, args)
        all_results["tgnn"] = {"metrics": tgnn_metrics}
        if args.mode == "tgnn":
            print_summary(all_results); return

    # ── Module 2: Mamba ───────────────────────────────────────────────────────
    mamba_model, mamba_metrics = None, {}
    if args.mode in ("full", "forecast"):
        if tgnn_emb is None and Path("models/results/tgnn_embeddings.npy").exists():
            tgnn_emb = np.load("models/results/tgnn_embeddings.npy")
            logger.info("Loaded T-GNN embeddings from disk.")
        mamba_model, mamba_metrics = step_mamba(dataset, cfg, args, tgnn_emb)
        all_results["mamba"] = {"metrics": mamba_metrics}
        if args.mode == "forecast":
            print_summary(all_results); return

    # ── Module 3: Conformal ───────────────────────────────────────────────────
    conformal_results = {}
    if args.mode in ("full", "conformal"):
        if mamba_model is None:
            logger.warning("Mamba model not trained yet; running Module 2 first...")
            if tgnn_emb is None and Path("models/results/tgnn_embeddings.npy").exists():
                tgnn_emb = np.load("models/results/tgnn_embeddings.npy")
            mamba_model, _ = step_mamba(dataset, cfg, args, tgnn_emb)
        conformal_results = step_conformal(dataset, cfg, mamba_model, tgnn_emb)
        all_results["conformal"] = conformal_results
        if args.mode == "conformal":
            print_summary(all_results); return

    # ── Module 4: Causal ──────────────────────────────────────────────────────
    causal_results = {}
    if args.mode in ("full", "causal"):
        causal_results = step_causal(dataset, cfg)
        all_results["causal"] = causal_results
        if args.mode == "causal":
            print_summary(all_results); return

    # ── Module 5: MARL ────────────────────────────────────────────────────────
    marl_results = {}
    if args.mode in ("full", "marl"):
        marl_results = step_marl(dataset, cfg, args)
        all_results["marl"] = marl_results

    # ── Final summary ─────────────────────────────────────────────────────────
    print_summary(all_results)
    logger.success("Pipeline complete! Run: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()

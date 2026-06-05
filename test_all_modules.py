"""
tests/test_all_modules.py
AquaIntel — Unit & Integration Tests

Run: pytest tests/test_all_modules.py -v
  or: python tests/test_all_modules.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest
from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cfg():
    from utils.helpers import load_config
    c = load_config("config.yaml")
    # Override for fast testing
    c["tgnn"]["epochs"]        = 2
    c["tgnn"]["batch_size"]    = 4
    c["mamba"]["epochs"]       = 2
    c["mamba"]["batch_size"]   = 4
    c["marl"]["train_steps"]   = 200
    c["marl"]["batch_size"]    = 16
    c["marl"]["buffer_size"]   = 500
    return c

@pytest.fixture(scope="session")
def dataset():
    from data.download_datasets import generate_synthetic_data
    return generate_synthetic_data(seed=42)

@pytest.fixture(scope="session")
def small_dataset():
    """Tiny dataset for fast unit tests."""
    from data.download_datasets import generate_synthetic_data, NODES, FEATURES
    d = generate_synthetic_data(seed=0)
    # Take first 48 timesteps only
    return {
        "data":     d["data"][:48],
        "cwsi":     d["cwsi"][:48],
        "adj":      d["adj"],
        "dates":    d["dates"][:48],
        "nodes":    d["nodes"],
        "features": d["features"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestData:
    def test_dataset_shape(self, dataset):
        T, N, F = dataset["data"].shape
        assert T > 100,  f"Expected >100 timesteps, got {T}"
        assert N == 20,  f"Expected 20 nodes, got {N}"
        assert F == 9,   f"Expected 9 features, got {F}"
        logger.info(f"✓ Dataset shape: ({T}, {N}, {F})")

    def test_cwsi_range(self, dataset):
        cwsi = dataset["cwsi"]
        assert cwsi.min() >= 0.0, "CWSI below 0"
        assert cwsi.max() <= 1.0, "CWSI above 1"
        logger.info(f"✓ CWSI range: [{cwsi.min():.3f}, {cwsi.max():.3f}]")

    def test_adjacency_matrix(self, dataset):
        adj = dataset["adj"]
        N = 20
        assert adj.shape == (N, N), f"Adj shape mismatch: {adj.shape}"
        assert np.allclose(np.diag(adj), 0), "Self-loops found in adjacency matrix"
        assert adj.min() >= 0.0, "Negative edge weights"
        assert adj.max() <= 1.0, "Edge weights > 1"
        logger.info(f"✓ Adjacency matrix: {adj.shape}, {(adj>0).sum()} edges")

    def test_no_nans(self, dataset):
        assert not np.isnan(dataset["data"]).any(), "NaN in data"
        assert not np.isnan(dataset["cwsi"]).any(), "NaN in CWSI"
        assert not np.isnan(dataset["adj"]).any(),  "NaN in adjacency"
        logger.info("✓ No NaN values in dataset")

    def test_save_load(self, tmp_path, dataset):
        from data.download_datasets import save_processed, load_processed
        import os
        # Temporarily change working dir to tmp
        orig = os.getcwd()
        os.chdir(tmp_path)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("data/raw", exist_ok=True)
        save_processed(dataset)
        loaded = load_processed()
        assert loaded["data"].shape == dataset["data"].shape
        assert np.allclose(loaded["cwsi"], dataset["cwsi"])
        os.chdir(orig)
        logger.info("✓ Save/load round-trip passed")


# ─────────────────────────────────────────────────────────────────────────────
# Utils Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUtils:
    def test_normalise(self):
        from utils.helpers import normalise, denormalise
        x = np.random.randn(100, 5)
        x_n, mu, std = normalise(x)
        assert np.allclose(x_n.mean(axis=0), 0, atol=1e-6), "Normalise: mean ≠ 0"
        assert np.allclose(x_n.std(axis=0),  1, atol=1e-5), "Normalise: std ≠ 1"
        x_rec = denormalise(x_n, mu, std)
        assert np.allclose(x_rec, x, atol=1e-5), "Denormalise: reconstruction failed"
        logger.info("✓ Normalise/denormalise round-trip passed")

    def test_metrics(self):
        from utils.helpers import compute_regression_metrics, coverage_width_criterion
        y_true = np.random.rand(100)
        y_pred = y_true + np.random.normal(0, 0.05, 100)
        m = compute_regression_metrics(y_true, y_pred)
        assert "MAE" in m and "RMSE" in m and "R2" in m
        assert m["MAE"]  >= 0
        assert m["RMSE"] >= 0
        # CWC
        y_lo = y_true - 0.1
        y_hi = y_true + 0.1
        cw = coverage_width_criterion(y_true, y_lo, y_hi)
        assert abs(cw["Coverage"] - 1.0) < 0.05
        logger.info(f"✓ Metrics: MAE={m['MAE']:.4f}, R2={m['R2']:.4f}, Coverage={cw['Coverage']:.3f}")

    def test_set_seed(self):
        from utils.helpers import set_seed
        set_seed(42)
        a = np.random.randn(10)
        set_seed(42)
        b = np.random.randn(10)
        assert np.allclose(a, b), "set_seed not reproducible"
        logger.info("✓ Reproducibility seed working")

    def test_train_val_test_split(self):
        from utils.helpers import train_val_test_split
        data = np.arange(100)
        tr, val, te = train_val_test_split(data, ratios=(0.7, 0.15, 0.15))
        assert len(tr)  == 70
        assert len(val) == 15
        assert len(te)  == 15
        assert tr[-1] < val[0] < te[0], "Split is not chronological"
        logger.info("✓ Chronological train/val/test split correct")


# ─────────────────────────────────────────────────────────────────────────────
# Module 1: T-GNN Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTGNN:
    def test_model_forward(self):
        from modules.module1_tgnn import TemporalGNN, build_graph_tensors
        adj = np.random.rand(20, 20) * 0.5
        np.fill_diagonal(adj, 0)
        edge_index, edge_attr = build_graph_tensors(adj, threshold=0.1)

        model = TemporalGNN(num_features=9, hidden_dim=32,
                            num_gat_layers=1, num_transformer_layers=2,
                            num_heads=4, dropout=0.0)
        x = torch.randn(2, 12, 20, 9)   # (B=2, T=12, N=20, F=9)
        with torch.no_grad():
            emb, pred = model(x, edge_index, edge_attr)

        assert emb.shape  == (2, 12, 20, 32), f"Embedding shape wrong: {emb.shape}"
        assert pred.shape == (2, 12, 20),     f"Prediction shape wrong: {pred.shape}"
        assert (pred >= 0).all() and (pred <= 1).all(), "CWSI pred outside [0,1]"
        logger.info(f"✓ T-GNN forward: emb={emb.shape}, pred={pred.shape}")

    def test_graph_construction(self):
        from modules.module1_tgnn import build_graph_tensors
        adj = np.eye(20) * 0  # no self-loops
        adj[0, 1] = 0.8
        adj[1, 2] = 0.6
        adj[3, 4] = 0.05  # below threshold
        edge_index, edge_attr = build_graph_tensors(adj, threshold=0.1)
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == 2   # only 2 edges above threshold
        logger.info(f"✓ Graph construction: {edge_index.shape[1]} edges from adj")

    def test_basin_dataset(self, small_dataset):
        from modules.module1_tgnn import BasinDataset
        ds = BasinDataset(small_dataset["data"], small_dataset["cwsi"],
                          seq_len=12, stride=1)
        assert len(ds) > 0, "Dataset is empty"
        x, y = ds[0]
        assert x.shape == (12, 20, 9), f"X shape: {x.shape}"
        assert y.shape == (12, 20),    f"Y shape: {y.shape}"
        logger.info(f"✓ BasinDataset: {len(ds)} samples, x={x.shape}")

    def test_tgnn_training(self, small_dataset, cfg):
        from modules.module1_tgnn import train_tgnn
        model, emb, metrics = train_tgnn(small_dataset, cfg)
        assert model is not None
        assert len(metrics["train_losses"]) == cfg["tgnn"]["epochs"]
        assert all(np.isfinite(metrics["train_losses"])), "Train loss has NaN/Inf"
        logger.info(f"✓ T-GNN training: {cfg['tgnn']['epochs']} epochs, "
                    f"final loss={metrics['train_losses'][-1]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Module 2: Mamba Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMamba:
    def test_mamba_block_forward(self):
        from modules.module2_mamba_forecast import MambaBlock
        blk = MambaBlock(d_model=64, d_state=8, d_conv=4, expand=2, dropout=0.0)
        x = torch.randn(2, 24, 64)   # (B, T, d_model)
        with torch.no_grad():
            out = blk(x)
        assert out.shape == x.shape, f"MambaBlock output shape wrong: {out.shape}"
        assert torch.isfinite(out).all(), "MambaBlock output has NaN/Inf"
        logger.info(f"✓ MambaBlock forward: {out.shape}")

    def test_spatial_attention(self):
        from modules.module2_mamba_forecast import SpatialAttention
        attn = SpatialAttention(d_model=64, num_heads=4, dropout=0.0)
        x    = torch.randn(4, 20, 64)  # (B, N, d_model)
        with torch.no_grad():
            out = attn(x)
        assert out.shape == x.shape
        logger.info(f"✓ SpatialAttention forward: {out.shape}")

    def test_mamba_forecaster_forward(self):
        from modules.module2_mamba_forecast import MambaForecaster
        model = MambaForecaster(input_dim=9, d_model=64, d_state=8,
                                d_conv=4, expand=2, num_layers=2,
                                num_nodes=20, num_heads=4, dropout=0.0,
                                forecast_horizons=[1, 3, 12])
        x = torch.randn(2, 24, 20, 9)   # (B, T, N, F)
        with torch.no_grad():
            pred = model(x)
        assert pred.shape == (2, 20, 3, 3), f"Mamba pred shape: {pred.shape}"
        # Check quantile ordering: q10 ≤ q50 ≤ q90
        assert (pred[..., 0] <= pred[..., 1] + 1e-4).all(), "q10 > q50"
        assert (pred[..., 1] <= pred[..., 2] + 1e-4).all(), "q50 > q90"
        assert (pred >= 0).all() and (pred <= 1).all(), "Preds outside [0,1]"
        logger.info(f"✓ MambaForecaster forward: {pred.shape}, quantiles monotonic")

    def test_quantile_loss(self):
        from modules.module2_mamba_forecast import quantile_loss
        pred   = torch.rand(4, 20, 3, 3)
        target = torch.rand(4, 20, 3)
        loss   = quantile_loss(pred, target)
        assert loss.item() >= 0, "Quantile loss is negative"
        assert torch.isfinite(loss), "Quantile loss is NaN/Inf"
        logger.info(f"✓ Quantile loss: {loss.item():.4f}")

    def test_mamba_training(self, small_dataset, cfg):
        from modules.module2_mamba_forecast import train_mamba
        model, metrics = train_mamba(small_dataset, cfg, tgnn_embeddings=None)
        assert model is not None
        assert all(np.isfinite(metrics["train_losses"]))
        logger.info(f"✓ Mamba training: final loss={metrics['train_losses'][-1]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Module 3: Conformal Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConformal:
    def test_cqr_calibration(self):
        from modules.module3_conformal import CQRWrapper
        np.random.seed(42)
        n = 500
        y_true  = np.random.rand(n)
        y_lower = y_true - 0.1 - np.random.rand(n) * 0.05
        y_upper = y_true + 0.1 + np.random.rand(n) * 0.05

        cqr = CQRWrapper(alpha=0.10)
        q_hat = cqr.calibrate(y_lower, y_upper, y_true)
        assert np.isfinite(q_hat), "q_hat is not finite"

        conf_lo, conf_hi = cqr.predict_intervals(y_lower[:100], y_upper[:100])
        metrics = cqr.evaluate(conf_lo, conf_hi, y_true[:100])
        assert metrics["Coverage"] >= 0.85, f"Coverage too low: {metrics['Coverage']:.3f}"
        logger.info(f"✓ CQR: q_hat={q_hat:.4f}, coverage={metrics['Coverage']:.3f}")

    def test_anomaly_detector(self):
        from modules.module3_conformal import ConformedAnomalyDetector
        np.random.seed(0)
        X_normal = np.random.randn(200, 20 * 9)
        X_cal    = np.random.randn(100, 20 * 9)
        X_test   = np.vstack([
            np.random.randn(50, 20 * 9),
            np.random.randn(10, 20 * 9) * 5 + 10  # outliers
        ])
        det = ConformedAnomalyDetector(contamination=0.05)
        det.fit(X_normal, X_cal)
        result = det.predict(X_test)
        assert "p_value"    in result
        assert "is_anomaly" in result
        assert "severity"   in result
        # The 10 outliers should mostly be flagged
        outlier_pvals = result["p_value"][50:]
        assert (outlier_pvals < 0.10).mean() > 0.5, "Outliers not detected"
        logger.info(f"✓ Anomaly detector: {result['is_anomaly'].sum()} anomalies in {len(X_test)} samples")

    def test_risk_classifier(self):
        from modules.module3_conformal import RiskClassifier
        np.random.seed(1)
        T, N, H = 100, 20, 32
        emb  = np.random.randn(T, N, H)
        cwsi = np.random.rand(T, N)

        clf = RiskClassifier()
        clf.fit(emb, cwsi, cal_fraction=0.3)
        preds = clf.predict(emb[:10])
        assert "predicted_class"  in preds
        assert "prediction_sets"  in preds
        assert preds["predicted_class"].shape == (10, N)
        logger.info(f"✓ RiskClassifier: {preds['predicted_class'].shape}")

    def test_cwsi_to_label(self):
        from modules.module3_conformal import RiskClassifier
        cwsi = np.array([0.10, 0.40, 0.65, 0.80])
        lbls = RiskClassifier.cwsi_to_label(cwsi)
        assert list(lbls) == [0, 1, 2, 3], f"Labels wrong: {lbls}"
        logger.info("✓ CWSI → risk label mapping correct")


# ─────────────────────────────────────────────────────────────────────────────
# Module 4: Causal Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCausal:
    def test_notears_tiny(self):
        """Test NOTEARS converges on a known 4-variable DAG."""
        from modules.module4_causal import NOTEARS
        np.random.seed(42)
        # Ground truth: x1 → x2 → x4, x3 → x4
        n, d = 300, 4
        x1 = np.random.randn(n)
        x2 = 0.8 * x1 + np.random.randn(n) * 0.3
        x3 = np.random.randn(n)
        x4 = 0.6 * x2 + 0.5 * x3 + np.random.randn(n) * 0.3
        X  = np.column_stack([x1, x2, x3, x4])
        # Normalise
        X  = (X - X.mean(0)) / (X.std(0) + 1e-8)

        nt = NOTEARS(lambda1=0.05, max_iter=30, w_threshold=0.2)
        W  = nt.fit(X)
        assert W.shape == (d, d), f"W shape: {W.shape}"
        assert np.diag(W).sum() == 0, "Self-loops in W"
        # x1→x2 edge should be present
        assert W[0, 1] != 0 or W[1, 0] != 0, "Expected x1↔x2 edge not found"
        logger.info(f"✓ NOTEARS: {(W!=0).sum()} edges discovered")

    def test_causal_graph_analyzer(self):
        from modules.module4_causal import CausalGraphAnalyzer
        W = np.zeros((5, 5))
        W[0, 2] = 0.7   # 0 → 2
        W[1, 2] = 0.5   # 1 → 2
        W[2, 4] = 0.8   # 2 → 4 (target)
        W[3, 4] = 0.4   # 3 → 4

        names = ["precip", "gw", "soil", "temp", "cwsi"]
        analyzer = CausalGraphAnalyzer(W, names)
        ancestors = analyzer.get_ancestors(4)  # ancestors of cwsi
        assert "precip" in ancestors or "gw" in ancestors, "Ancestors not found"
        logger.info(f"✓ Causal ancestors of cwsi: {list(ancestors.keys())}")

        df = analyzer.causal_summary("cwsi")
        assert len(df) > 0
        logger.info(f"✓ Causal summary:\n{df.to_string(index=False)}")

    def test_counterfactual(self):
        from modules.module4_causal import CausalGraphAnalyzer
        np.random.seed(5)
        d = 4
        W = np.zeros((d, d))
        W[0, 3] = -0.6   # precip → cwsi (negative: more rain = less stress)
        W[1, 3] =  0.5   # temp   → cwsi
        W[2, 3] =  0.4   # irr    → cwsi
        names = ["precip", "temp", "irr", "cwsi"]
        analyzer = CausalGraphAnalyzer(W, names)
        X = np.random.randn(100, d)
        cf = analyzer.counterfactual(X, treatment_idx=0, new_value=2.0, target_idx=3)
        assert cf.shape == (100,), f"CF shape: {cf.shape}"
        logger.info(f"✓ Counterfactual: mean CF cwsi = {cf.mean():.4f}")

    def test_causal_module_runs(self, small_dataset, cfg):
        from modules.module4_causal import run_causal_module
        results = run_causal_module(small_dataset, cfg)
        assert "W_est"          in results
        assert "causal_ranking" in results
        assert "dot_graph"      in results
        assert len(results["causal_ranking"]) > 0
        logger.info(f"✓ Causal module: {results['W_est'].shape} DAG, "
                    f"top driver = {results['causal_ranking'].iloc[0]['Variable']}")


# ─────────────────────────────────────────────────────────────────────────────
# Module 5: MARL Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMARL:
    def test_environment_reset(self, small_dataset, cfg):
        from modules.module5_marl import WaterAllocationEnv
        env = WaterAllocationEnv(small_dataset, cfg)
        obs = env.reset(t_start=12)
        assert len(obs) == env.NUM_AGENTS
        assert all(o.shape == (env.LOCAL_OBS_DIM,) for o in obs)
        assert np.isfinite(env.water_budget) and env.water_budget > 0
        logger.info(f"✓ Env reset: {len(obs)} agents, budget={env.water_budget:.2f}")

    def test_environment_step(self, small_dataset, cfg):
        from modules.module5_marl import WaterAllocationEnv
        env = WaterAllocationEnv(small_dataset, cfg)
        obs = env.reset(t_start=12)
        actions = [1, 2, 3]   # Low/Medium/High for each agent
        next_obs, state, rewards, done, info = env.step(actions)
        assert len(next_obs) == env.NUM_AGENTS
        assert state.shape == (env.GLOBAL_STATE_DIM,)
        assert len(rewards) == env.NUM_AGENTS
        assert all(np.isfinite(r) for r in rewards)
        assert "allocations" in info
        assert sum(info["allocations"]) <= env.water_budget + 1e-6
        logger.info(f"✓ Env step: rewards={[f'{r:.3f}' for r in rewards]}, "
                    f"alloc={[f'{a:.2f}' for a in info['allocations']]}")

    def test_agent_network_forward(self):
        from modules.module5_marl import AgentNetwork
        net = AgentNetwork(obs_dim=11, action_dim=5, hidden_dim=32)
        obs = torch.randn(4, 11)    # batch of 4
        h   = net.init_hidden(4)
        q, h_new = net(obs, h)
        assert q.shape     == (4, 5),  f"Q shape: {q.shape}"
        assert h_new.shape == (4, 32), f"H shape: {h_new.shape}"
        assert torch.isfinite(q).all()
        logger.info(f"✓ AgentNetwork forward: q={q.shape}")

    def test_qmix_mixer_forward(self):
        from modules.module5_marl import QMIXMixer
        mixer = QMIXMixer(num_agents=3, state_dim=33, embed_dim=16)
        agent_qs = torch.rand(4, 1, 3)    # (B, T=1, n_agents)
        state    = torch.randn(4, 1, 33)  # (B, T=1, state_dim)
        q_tot    = mixer(agent_qs, state)
        assert q_tot.shape == (4, 1, 1), f"Q_tot shape: {q_tot.shape}"
        # Monotonicity: increasing any agent Q should increase Q_tot
        agent_qs2 = agent_qs.clone()
        agent_qs2[:, :, 0] += 10.0
        q_tot2 = mixer(agent_qs2, state)
        assert (q_tot2 >= q_tot - 1e-4).all(), "QMIX violates monotonicity!"
        logger.info(f"✓ QMIX mixer: q_tot={q_tot.shape}, monotonicity ✓")

    def test_replay_buffer(self):
        from modules.module5_marl import ReplayBuffer
        buf = ReplayBuffer(capacity=100)
        for _ in range(50):
            buf.push(
                np.random.randn(3, 11),  # obs
                np.random.randn(33),     # global state
                np.array([0, 1, 2]),     # actions
                np.array([0.1, 0.2, 0.3]),  # rewards
                np.random.randn(3, 11),  # next obs
                np.random.randn(33),     # next state
                0.0,                     # done
            )
        assert len(buf) == 50
        batch = buf.sample(16)
        assert len(batch.obs) == 16
        logger.info(f"✓ ReplayBuffer: {len(buf)} transitions, sample batch of 16")

    def test_marl_training_short(self, small_dataset, cfg):
        from modules.module5_marl import run_marl_module
        results = run_marl_module(small_dataset, cfg)
        assert "trainer"       in results
        assert "train_results" in results
        assert "eval_results"  in results
        ev = results["eval_results"]
        assert 0 <= ev["mean_efficiency"] <= 1.1
        logger.info(f"✓ MARL training: efficiency={ev['mean_efficiency']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Integration Test
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_smoke(self, small_dataset, cfg):
        """
        Smoke test: run all modules on tiny dataset with minimal epochs.
        Checks that the full pipeline executes without errors.
        """
        logger.info("Running full pipeline smoke test...")

        # Module 1
        from modules.module1_tgnn import train_tgnn
        tgnn_model, embeddings, _ = train_tgnn(small_dataset, cfg)
        assert embeddings is not None
        logger.info(f"  ✓ Module 1 T-GNN: embeddings={embeddings.shape}")

        # Module 2
        from modules.module2_mamba_forecast import train_mamba
        mamba_model, _ = train_mamba(small_dataset, cfg, embeddings)
        assert mamba_model is not None
        logger.info("  ✓ Module 2 Mamba: model trained")

        # Module 3
        from modules.module3_conformal import run_conformal_module
        conf = run_conformal_module(small_dataset, cfg, mamba_model, embeddings)
        assert conf["interval_metrics"]["Coverage"] > 0
        logger.info(f"  ✓ Module 3 Conformal: coverage={conf['interval_metrics']['Coverage']:.3f}")

        # Module 4
        from modules.module4_causal import run_causal_module
        causal = run_causal_module(small_dataset, cfg)
        assert causal["W_est"] is not None
        logger.info(f"  ✓ Module 4 Causal: {(causal['W_est']!=0).sum()} edges")

        # Module 5
        from modules.module5_marl import run_marl_module
        marl = run_marl_module(small_dataset, cfg)
        assert marl["eval_results"]["mean_efficiency"] >= 0
        logger.info(f"  ✓ Module 5 MARL: efficiency={marl['eval_results']['mean_efficiency']:.3f}")

        logger.info("✅ Full pipeline smoke test PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short",
         "--no-header", "-rN"],
        capture_output=False
    )
    sys.exit(result.returncode)

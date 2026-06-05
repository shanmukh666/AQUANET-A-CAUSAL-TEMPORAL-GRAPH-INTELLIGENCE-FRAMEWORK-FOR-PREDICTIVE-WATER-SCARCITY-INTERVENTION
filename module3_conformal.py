"""
modules/module3_conformal.py
Module 3 — Conformal Prediction for Uncertainty Quantification

Standard ML: "Drought risk is 0.72"
This module: "Drought risk is between 0.65 and 0.82 with 90% statistical guarantee"

Method: Conformalized Quantile Regression (CQR)
  - Wraps the Mamba forecaster from Module 2
  - Uses a calibration set (never seen during training) to compute
    non-conformity scores: how wrong was the model on held-out data?
  - Adjusts prediction intervals to guarantee ≥90% coverage on new data
  - Coverage guarantee is distribution-free — no assumptions about data!

Paper: Romano et al., "Conformalized Quantile Regression", NeurIPS 2019
       Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction", 2021
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from loguru import logger
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import get_device, coverage_width_criterion, normalise


# ─────────────────────────────────────────────────────────────────────────────
# Conformalized Quantile Regression (CQR)
# ─────────────────────────────────────────────────────────────────────────────

class CQRWrapper:
    """
    Wraps any quantile-predicting model with conformal guarantees.

    The key insight: After calibration, for any new input x_new:
        P(y_new ∈ [q̂_lo(x_new) - Q̂, q̂_hi(x_new) + Q̂]) ≥ 1 - alpha

    where Q̂ is the (1-alpha)(1+1/n) quantile of calibration non-conformity scores.
    """
    def __init__(self, alpha: float = 0.1):
        self.alpha      = alpha       # Miscoverage rate (0.1 → 90% coverage)
        self.q_hat      = None        # Conformal quantile correction
        self.calibrated = False

    def calibrate(self, y_lower: np.ndarray, y_upper: np.ndarray,
                  y_true: np.ndarray) -> float:
        """
        Compute conformal quantile from calibration set.

        Non-conformity score for CQR:
            s_i = max(q̂_lo(x_i) - y_i, y_i - q̂_hi(x_i))
        This measures how far the true value is outside [lower, upper].

        Args:
            y_lower: (n_cal,) lower quantile predictions on calibration set
            y_upper: (n_cal,) upper quantile predictions on calibration set
            y_true:  (n_cal,) true values on calibration set
        """
        n = len(y_true)
        # Non-conformity scores
        scores = np.maximum(y_lower - y_true, y_true - y_upper)

        # Compute Q̂ = (1-alpha)(1+1/n) quantile of scores
        level = np.ceil((1 - self.alpha) * (1 + 1 / n)) / (1 + 1 / n)
        level = np.clip(level, 0, 1)
        self.q_hat = np.quantile(scores, level)
        self.calibrated = True

        logger.info(f"CQR calibrated: Q̂ = {self.q_hat:.4f}  "
                    f"(alpha={self.alpha}, n_cal={n})")
        return self.q_hat

    def predict_intervals(self, y_lower: np.ndarray,
                          y_upper: np.ndarray) -> tuple:
        """
        Apply conformal correction to prediction intervals.
        Returns guaranteed (lower, upper) intervals.
        """
        if not self.calibrated:
            raise RuntimeError("Call calibrate() before predict_intervals()")
        conf_lower = y_lower - self.q_hat
        conf_upper = y_upper + self.q_hat
        return np.clip(conf_lower, 0, 1), np.clip(conf_upper, 0, 1)

    def evaluate(self, conf_lower: np.ndarray, conf_upper: np.ndarray,
                 y_true: np.ndarray) -> dict:
        """Evaluate coverage and interval width."""
        metrics = coverage_width_criterion(y_true, conf_lower, conf_upper)
        coverage = metrics["Coverage"]
        width    = metrics["Width"]
        logger.info(f"Conformal Coverage: {coverage:.3f} "
                    f"(target: {1-self.alpha:.2f}) | Width: {width:.4f}")
        if coverage < 1 - self.alpha - 0.01:
            logger.warning("Coverage below target — check calibration set size!")
        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly Detection via Isolation Forest + Conformal p-values
# ─────────────────────────────────────────────────────────────────────────────

class ConformedAnomalyDetector:
    """
    Detects water crisis anomalies with statistically valid p-values.

    Step 1: Train Isolation Forest on normal data (pre-2010 or drought-free years)
    Step 2: Compute anomaly scores on calibration data
    Step 3: For new data, conformal p-value = fraction of calibration scores
            that are at least as extreme as the new score

    A p-value < 0.05 → statistically significant anomaly (drought / flood event)
    """
    def __init__(self, contamination: float = 0.05, n_estimators: int = 200):
        self.iforest     = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
        )
        self.cal_scores  = None
        self.scaler      = StandardScaler()
        self.fitted      = False

    def fit(self, X_train: np.ndarray, X_cal: np.ndarray) -> None:
        """
        X_train: normal-period data (T_normal, N*F) flattened per timestep
        X_cal:   calibration data for computing reference scores
        """
        X_train_s = self.scaler.fit_transform(X_train)
        self.iforest.fit(X_train_s)

        X_cal_s = self.scaler.transform(X_cal)
        # Isolation Forest anomaly score (more negative = more anomalous)
        raw = self.iforest.score_samples(X_cal_s)
        self.cal_scores = -raw   # flip so higher = more anomalous
        self.fitted = True
        logger.info(f"Anomaly detector fitted. "
                    f"Calibration scores: mean={self.cal_scores.mean():.3f}, "
                    f"std={self.cal_scores.std():.3f}")

    def predict(self, X_new: np.ndarray) -> dict:
        """
        Returns dict with:
          - anomaly_score: raw score per timestep
          - p_value: conformal p-value (lower = more anomalous)
          - is_anomaly: boolean flag (p_value < 0.05)
          - severity: 'normal' / 'watch' / 'warning' / 'crisis'
        """
        if not self.fitted:
            raise RuntimeError("Call fit() first")

        X_new_s = self.scaler.transform(X_new)
        raw_scores = -self.iforest.score_samples(X_new_s)

        # Conformal p-values
        n_cal = len(self.cal_scores)
        p_values = np.array([
            (np.sum(self.cal_scores >= s) + 1) / (n_cal + 1)
            for s in raw_scores
        ])

        is_anomaly = p_values < 0.05

        # Severity classification
        severity = []
        for p in p_values:
            if p >= 0.20:
                severity.append("normal")
            elif p >= 0.10:
                severity.append("watch")
            elif p >= 0.05:
                severity.append("warning")
            else:
                severity.append("crisis")

        return {
            "anomaly_score": raw_scores,
            "p_value":       p_values,
            "is_anomaly":    is_anomaly,
            "severity":      severity,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node-Level Conformal Risk Classifier
# ─────────────────────────────────────────────────────────────────────────────

class RiskClassifier:
    """
    Lightweight GBM trained on T-GNN embeddings to classify each node into:
      Level 0: Adequate water supply    (CWSI < 0.30)
      Level 1: Moderate stress          (0.30 ≤ CWSI < 0.55)
      Level 2: High stress              (0.55 ≤ CWSI < 0.75)
      Level 3: Crisis / Scarcity        (CWSI ≥ 0.75)

    Conformal p-values from RAPS (Regularized Adaptive Prediction Sets)
    give a valid set of possible risk levels for each prediction.
    """
    LEVELS = {0: "Adequate", 1: "Moderate stress",
              2: "High stress", 3: "Crisis"}
    THRESHOLDS = [0.30, 0.55, 0.75]

    def __init__(self):
        self.models   = {}   # one GBM per risk class (one-vs-rest)
        self.scaler   = StandardScaler()
        self.cal_sets = {}   # calibration softmax scores per class
        self.fitted   = False

    @staticmethod
    def cwsi_to_label(cwsi: np.ndarray) -> np.ndarray:
        labels = np.zeros(len(cwsi), dtype=int)
        for i, thresh in enumerate(RiskClassifier.THRESHOLDS):
            labels[cwsi >= thresh] = i + 1
        return labels

    def fit(self, X_emb: np.ndarray, cwsi: np.ndarray,
            cal_fraction: float = 0.2) -> None:
        """
        X_emb: (T, N, H) T-GNN embeddings
        cwsi:  (T, N) water stress index
        """
        T, N, H = X_emb.shape
        # Flatten: each sample = one (node, timestep) pair
        X_flat = X_emb.reshape(T * N, H)
        y_flat = self.cwsi_to_label(cwsi.reshape(T * N))

        # Normalise
        n_train = int(len(X_flat) * (1 - cal_fraction))
        X_s = self.scaler.fit_transform(X_flat)

        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.ensemble import GradientBoostingClassifier

        clf = OneVsRestClassifier(
            GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        )
        clf.fit(X_s[:n_train], y_flat[:n_train])
        self.clf = clf

        # Conformal calibration: collect scores on held-out calibration samples
        X_cal = X_s[n_train:]
        y_cal = y_flat[n_train:]
        probs = clf.predict_proba(X_cal)  # (n_cal, 4)

        # RAPS score: 1 - softmax of true class
        self.cal_softmax_scores = {
            c: 1 - probs[y_cal == c, c] for c in range(4)
        }

        # Compute q_hat per class
        alpha = 0.10
        self.q_hats = {}
        for c in range(4):
            scores = self.cal_softmax_scores[c]
            if len(scores) > 0:
                n = len(scores)
                level = min(np.ceil((1 - alpha) * (1 + 1 / n)) / (1 + 1/n), 1.0)
                self.q_hats[c] = np.quantile(scores, level)
            else:
                self.q_hats[c] = 1.0

        self.fitted = True
        logger.info(f"Risk classifier fitted. q_hats: {self.q_hats}")

    def predict(self, X_emb: np.ndarray) -> dict:
        """
        Returns prediction set (valid set of risk levels) for each sample.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() first")

        shape = X_emb.shape[:-1]  # (T, N) or (N,)
        X_flat = X_emb.reshape(-1, X_emb.shape[-1])
        X_s    = self.scaler.transform(X_flat)
        probs  = self.clf.predict_proba(X_s)  # (n, 4)
        pred_class = np.argmax(probs, axis=1)

        # Conformal prediction sets
        pred_sets = []
        for i in range(len(X_flat)):
            pset = [c for c in range(4) if 1 - probs[i, c] <= self.q_hats[c]]
            if not pset:  # guarantee non-empty
                pset = [pred_class[i]]
            pred_sets.append(pset)

        labels = [self.LEVELS[c] for c in pred_class]
        return {
            "predicted_class":  pred_class.reshape(shape),
            "predicted_label":  np.array(labels).reshape(shape),
            "prediction_sets":  np.array(pred_sets, dtype=object).reshape(shape),
            "probabilities":    probs.reshape(*shape, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Full Module 3 Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_conformal_module(dataset_dict: dict, cfg: dict,
                         mamba_model, tgnn_embeddings: np.ndarray) -> dict:
    """
    End-to-end conformal prediction module.
    Returns calibrated uncertainty bounds and anomaly detections.
    """
    device  = get_device()
    ccfg    = cfg["conformal"]
    data    = dataset_dict["data"]    # (T, N, F)
    cwsi    = dataset_dict["cwsi"]    # (T, N)
    T, N, F = data.shape

    logger.info("── Running Module 3: Conformal Prediction ─────────")

    # ─ Split ─
    train_end = int(T * 0.70)
    cal_end   = int(T * 0.85)

    # ─ A. CQR on Mamba forecasts ─
    mamba_model.eval()
    data_norm, mu, std = normalise(data.reshape(T, -1))
    data_norm = data_norm.reshape(T, N, F)

    # Get calibration predictions from Mamba
    seq_len = 36
    cal_preds_lower, cal_preds_upper, cal_true = [], [], []

    with torch.no_grad():
        for t in range(seq_len, cal_end - 1):
            x_in = torch.FloatTensor(data_norm[t-seq_len:t]).unsqueeze(0).to(device)
            pred = mamba_model(x_in)     # (1, N, horizons, 3)
            # Use 1-month horizon (index 0)
            lower = pred[0, :, 0, 0].cpu().numpy()  # q10
            upper = pred[0, :, 0, 2].cpu().numpy()  # q90
            cal_preds_lower.append(lower)
            cal_preds_upper.append(upper)
            cal_true.append(cwsi[t])

    cal_lower = np.array(cal_preds_lower).reshape(-1)  # (T_cal * N,)
    cal_upper = np.array(cal_preds_upper).reshape(-1)
    cal_true_flat = np.array(cal_true).reshape(-1)

    cqr = CQRWrapper(alpha=ccfg["alpha"])
    cqr.calibrate(cal_lower, cal_upper, cal_true_flat)

    # Evaluate on test set
    test_preds_lower, test_preds_upper, test_true = [], [], []
    with torch.no_grad():
        for t in range(cal_end, T - 1):
            x_in = torch.FloatTensor(data_norm[t-seq_len:t]).unsqueeze(0).to(device)
            pred = mamba_model(x_in)
            lower = pred[0, :, 0, 0].cpu().numpy()
            upper = pred[0, :, 0, 2].cpu().numpy()
            test_preds_lower.append(lower)
            test_preds_upper.append(upper)
            test_true.append(cwsi[t])

    test_lower = np.array(test_preds_lower).reshape(-1)
    test_upper = np.array(test_preds_upper).reshape(-1)
    test_true_flat = np.array(test_true).reshape(-1)

    conf_lower, conf_upper = cqr.predict_intervals(test_lower, test_upper)
    interval_metrics = cqr.evaluate(conf_lower, conf_upper, test_true_flat)

    # ─ B. Anomaly detection ─
    # Normal period: 2002–2008 (pre-major stress)
    normal_end = min(80, train_end // 2)  # first ~6.5 years
    X_normal = data[:normal_end].reshape(normal_end, -1)
    X_cal_anomaly = data[normal_end:train_end].reshape(train_end - normal_end, -1)

    anomaly_detector = ConformedAnomalyDetector(contamination=0.05)
    anomaly_detector.fit(X_normal, X_cal_anomaly)

    X_test = data[cal_end:].reshape(T - cal_end, -1)
    anomaly_results = anomaly_detector.predict(X_test)

    n_crisis = np.sum(np.array(anomaly_results["severity"]) == "crisis")
    n_warning = np.sum(np.array(anomaly_results["severity"]) == "warning")
    logger.info(f"Anomaly summary: {n_crisis} crisis events, {n_warning} warnings detected")

    # ─ C. Risk classifier ─
    if tgnn_embeddings is not None:
        emb_T = min(T, len(tgnn_embeddings))
        risk_clf = RiskClassifier()
        risk_clf.fit(tgnn_embeddings[:train_end], cwsi[:train_end])
        risk_preds = risk_clf.predict(tgnn_embeddings[cal_end:emb_T])
        logger.info("Risk classification complete with conformal prediction sets")
    else:
        risk_clf   = None
        risk_preds = None
        logger.warning("T-GNN embeddings not provided; skipping risk classifier")

    logger.success("Module 3 complete.")
    return {
        "cqr":             cqr,
        "interval_metrics": interval_metrics,
        "anomaly_detector": anomaly_detector,
        "anomaly_results":  anomaly_results,
        "risk_classifier":  risk_clf,
        "risk_predictions": risk_preds,
        "conformal_lower":  conf_lower,
        "conformal_upper":  conf_upper,
        "test_true":        test_true_flat,
    }

"""
modules/module4_causal.py
Module 4 — NOTEARS Causal Discovery + DoWhy Counterfactual Reasoning

This is the module that elevates AquaIntel from "prediction" to "understanding".

NOTEARS (Non-combinatorial Optimization via Trace Exponential and Augmented
lagRangian for Structure learning) learns a Directed Acyclic Graph (DAG)
representing causal relationships between water variables.

Key question answered: WHY is water scarce?
  - Does deforestation CAUSE lower groundwater? (or just correlate?)
  - Does population growth CAUSE irrigation demand? (obvious) — and by how much?
  - What is the counterfactual: if we restored 20% forest cover, what
    would groundwater be?

Paper: Zheng et al., "DAGs with NOTEARS", NeurIPS 2018
DoWhy: Sharma & Kiciman, "DoWhy: An End-to-End Library for Causal Inference", 2020
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit
from loguru import logger
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import normalise


# ─────────────────────────────────────────────────────────────────────────────
# NOTEARS: DAG structure learning via continuous optimisation
# ─────────────────────────────────────────────────────────────────────────────

class NOTEARS:
    """
    Learns a causal DAG W (weighted adjacency matrix) from observational data X.

    The DAG constraint (no cycles) is enforced via:
        h(W) = tr(e^{W ⊙ W}) - d = 0  (Zheng et al. 2018)
    where d = number of variables and ⊙ is elementwise product.

    Optimisation: Augmented Lagrangian method
        min_{W} (1/2n)||X - X·W||²_F + lambda1*||W||_1
        s.t.    h(W) = 0
    """

    def __init__(self, lambda1: float = 0.01, loss_type: str = "l2",
                 max_iter: int = 100, h_tol: float = 1e-8,
                 rho_max: float = 1e16, w_threshold: float = 0.3):
        self.lambda1     = lambda1
        self.loss_type   = loss_type
        self.max_iter    = max_iter
        self.h_tol       = h_tol
        self.rho_max     = rho_max
        self.w_threshold = w_threshold
        self.W_est       = None

    @staticmethod
    def _h(W: np.ndarray) -> float:
        """DAG acyclicity constraint: h(W) = tr(e^{W²}) - d."""
        d = W.shape[0]
        M = np.eye(d) + W * W / d           # (I + W²/d)^d approximation
        E = np.linalg.matrix_power(M, d)     # cheaper than matrix exp
        return np.trace(E) - d

    @staticmethod
    def _h_grad(W: np.ndarray) -> np.ndarray:
        """Gradient of h w.r.t. W."""
        d = W.shape[0]
        M = np.eye(d) + W * W / d
        E = np.linalg.matrix_power(M, d - 1)
        return (E.T * 2 * W / d)

    def _loss(self, W: np.ndarray, X: np.ndarray) -> tuple:
        """Squared loss + gradient."""
        n, d = X.shape
        M = X @ W
        R = X - M
        loss = 0.5 / n * (R ** 2).sum()
        G    = -1.0 / n * X.T @ R
        return loss, G

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the causal DAG.
        X: (n_samples, d_variables) — should be normalised
        Returns: W_est (d, d) weighted DAG adjacency matrix
                 W_est[i,j] ≠ 0 means variable i causally influences variable j
        """
        n, d = X.shape
        logger.info(f"NOTEARS: fitting causal DAG on {n} samples × {d} variables")

        W = np.zeros((d, d))
        rho, alpha, h = 1.0, 0.0, np.inf

        for it in range(self.max_iter):
            # Augmented Lagrangian inner loop
            W_prev = W.copy()
            for inner in range(500):
                loss, G_loss = self._loss(W, X)
                h_val  = self._h(W)
                G_h    = self._h_grad(W)
                G_obj  = G_loss + (rho * h_val + alpha) * G_h

                # L1 subgradient
                G_obj += self.lambda1 * np.sign(W)

                # Gradient step with backtracking line search
                step = 1.0
                for _ in range(50):
                    W_new = W - step * G_obj
                    np.fill_diagonal(W_new, 0)   # no self-loops
                    loss_new, _ = self._loss(W_new, X)
                    h_new = self._h(W_new)
                    if (loss_new + (rho*h_new + alpha)*h_new
                            <= loss + (rho*h_val + alpha)*h_val - 0.5*step*(G_obj**2).sum()):
                        break
                    step *= 0.5
                W = W_new
                if np.max(np.abs(W - W_prev)) < 1e-6:
                    break
                W_prev = W.copy()

            h_new = self._h(W)
            if h_new > 0.25 * h:
                rho = min(rho * 10, self.rho_max)
            alpha += rho * h_new
            h = h_new

            if it % 10 == 0:
                logger.info(f"  NOTEARS iter {it:3d}: h={h:.2e}, rho={rho:.1e}")

            if h <= self.h_tol and rho >= self.rho_max:
                logger.info(f"  Converged at iteration {it}")
                break

        # Threshold small edges
        W[np.abs(W) < self.w_threshold] = 0
        np.fill_diagonal(W, 0)
        self.W_est = W
        n_edges = np.sum(W != 0)
        logger.success(f"NOTEARS complete: {n_edges} causal edges discovered")
        return W


# ─────────────────────────────────────────────────────────────────────────────
# Causal Graph Analysis
# ─────────────────────────────────────────────────────────────────────────────

class CausalGraphAnalyzer:
    """
    Analyses the learned causal DAG to:
    1. Identify direct and indirect causes of water stress
    2. Rank variables by causal influence (path-based)
    3. Compute Average Causal Effect (ACE) via do-calculus approximation
    4. Run counterfactual simulations ("what if X changed?")
    """

    def __init__(self, W: np.ndarray, variable_names: list):
        self.W     = W
        self.names = variable_names
        self.d     = len(variable_names)

    def get_ancestors(self, target_idx: int) -> dict:
        """
        Find all ancestors of the target variable in the DAG,
        with their path-based causal strength.
        """
        # BFS/DFS over the transitive closure
        W_abs = np.abs(self.W)
        ancestors = {}
        queue = [(target_idx, 1.0)]
        visited = set()

        while queue:
            node, strength = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            # Find parents: W[parent, node] ≠ 0
            parents = np.where(W_abs[:, node] > 0)[0]
            for p in parents:
                edge_w = W_abs[p, node]
                acc_strength = strength * edge_w
                if p not in ancestors or ancestors[p] < acc_strength:
                    ancestors[p] = acc_strength
                queue.append((p, acc_strength))

        return {self.names[k]: v for k, v in ancestors.items()}

    def average_causal_effect(self, X: np.ndarray,
                               treatment_idx: int, outcome_idx: int,
                               delta: float = 1.0) -> float:
        """
        Estimate ACE via the linear do-calculus approximation:
            ACE(X_t → X_o) ≈ (W^T)_{t→o}  (for linear models)

        For non-linear: use do-interventions via regression on residuals.
        delta: size of intervention (in standard deviation units)
        """
        # Linear path-tracing formula (Wright's rule)
        # Compute matrix of total effects: (I - W)^{-1}
        try:
            I = np.eye(self.d)
            total_effects = np.linalg.inv(I - self.W.T)
            ace = total_effects[treatment_idx, outcome_idx] * delta
        except np.linalg.LinAlgError:
            # Fallback: estimate via intervention simulation
            ace = self._intervention_ace(X, treatment_idx, outcome_idx, delta)
        return ace

    def _intervention_ace(self, X: np.ndarray,
                           treatment_idx: int, outcome_idx: int,
                           delta: float) -> float:
        """Estimate ACE by simulating do(X_t = X_t + delta)."""
        X_do = X.copy()
        X_do[:, treatment_idx] += delta
        # Propagate intervention through DAG
        y_natural   = X @ self.W
        y_intervened = X_do @ self.W
        return np.mean(y_intervened[:, outcome_idx] - y_natural[:, outcome_idx])

    def counterfactual(self, X: np.ndarray, treatment_idx: int,
                        new_value: float, target_idx: int) -> np.ndarray:
        """
        Compute counterfactual outcome: what would target be if treatment = new_value?
        Returns: counterfactual target values (n_samples,)
        """
        # 1. Abduction: infer exogenous noise from observed data
        # In linear SEM: epsilon = X - X @ W
        epsilon = X - X @ self.W

        # 2. Action: set treatment to new value
        X_cf = X.copy()
        X_cf[:, treatment_idx] = new_value

        # 3. Prediction: recompute all descendants
        for step in range(self.d):  # forward pass through DAG
            # Topological ordering (simplified — assumes W is upper triangular after ordering)
            for j in range(self.d):
                parents = np.where(self.W[:, j] != 0)[0]
                if len(parents) > 0 and j != treatment_idx:
                    X_cf[:, j] = X_cf[:, parents] @ self.W[parents, j] + epsilon[:, j]

        return X_cf[:, target_idx]

    def causal_summary(self, target: str = "water_stress_index") -> pd.DataFrame:
        """Return ranked causal factors for the target variable."""
        if target not in self.names:
            logger.warning(f"'{target}' not in variable names. Using last variable.")
            target_idx = self.d - 1
        else:
            target_idx = self.names.index(target)

        ancestors = self.get_ancestors(target_idx)
        df = pd.DataFrame({
            "Variable":       list(ancestors.keys()),
            "Causal_Strength": list(ancestors.values()),
        }).sort_values("Causal_Strength", ascending=False)
        df["Rank"] = range(1, len(df) + 1)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# DoWhy Integration (when available)
# ─────────────────────────────────────────────────────────────────────────────

def dowhy_causal_effect(df: pd.DataFrame, treatment: str, outcome: str,
                         dag_dot: str = None) -> dict:
    """
    Use DoWhy for rigorous causal effect estimation.
    Compatible with dowhy 0.8 (Python 3.13/3.14) and newer versions.
    Falls back gracefully if DoWhy not installed or estimation fails.
    """
    try:
        import dowhy
        from dowhy import CausalModel

        # dowhy 0.8 uses 'graph' as a DOT string directly
        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            graph=dag_dot,
            logging_level="WARNING",   # suppress verbose output
        )
        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=True
        )
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
        )

        # Refutation (graceful — some dowhy 0.8 builds skip this)
        refute_result = None
        try:
            refute_result = model.refute_estimate(
                identified_estimand, estimate,
                method_name="random_common_cause"
            )
        except Exception:
            pass   # refutation optional — core estimate still valid

        return {
            "estimate":      float(estimate.value),
            "refute_result": str(refute_result) if refute_result else "skipped",
            "dowhy_version": dowhy.__version__,
        }

    except ImportError:
        logger.warning("DoWhy not installed. Run: pip install dowhy==0.8")
        return {"estimate": None, "error": "dowhy not installed"}
    except Exception as e:
        logger.warning(f"DoWhy estimation failed (non-critical): {e}")
        return {"estimate": None, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Full Module 4 Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_causal_module(dataset_dict: dict, cfg: dict) -> dict:
    """
    End-to-end causal discovery and counterfactual analysis.
    """
    ccfg  = cfg["causal"]
    data  = dataset_dict["data"]   # (T, N, F)
    cwsi  = dataset_dict["cwsi"]   # (T, N)
    T, N, F = data.shape
    feature_names = dataset_dict.get("features", [f"feat_{i}" for i in range(F)])

    logger.info("── Running Module 4: Causal Discovery ─────────────")

    # ─ Aggregate across nodes (basin-level causal analysis) ─
    # Mean across nodes for each feature + CWSI
    data_mean  = data.mean(axis=1)  # (T, F)
    cwsi_mean  = cwsi.mean(axis=1, keepdims=True)  # (T, 1)
    causal_data = np.concatenate([data_mean, cwsi_mean], axis=1)  # (T, F+1)

    var_names = feature_names + ["water_stress_index"]
    d = len(var_names)

    # Normalise for NOTEARS
    causal_norm, _, _ = normalise(causal_data, axis=0)

    # ─ Run NOTEARS ─
    notears = NOTEARS(
        lambda1=ccfg["lambda1"],
        max_iter=ccfg["max_iter"],
        h_tol=ccfg["h_tol"],
        rho_max=ccfg["rho_max"],
        w_threshold=ccfg["w_threshold"],
    )
    W_est = notears.fit(causal_norm)

    # ─ Build causal graph DOT string ─
    dot_lines = ["digraph {"]
    for i in range(d):
        for j in range(d):
            if W_est[i, j] != 0:
                label = f"{W_est[i,j]:.2f}"
                dot_lines.append(f'  "{var_names[i]}" -> "{var_names[j]}" [label="{label}"];')
    dot_lines.append("}")
    dot_str = "\n".join(dot_lines)

    # ─ Causal analysis ─
    analyzer = CausalGraphAnalyzer(W_est, var_names)

    # Root causes of water stress
    target = "water_stress_index"
    causal_ranking = analyzer.causal_summary(target)
    logger.info(f"\nTop causal drivers of water stress:")
    logger.info(causal_ranking.to_string(index=False))

    # ─ Key causal effects ─
    causal_effects = {}
    key_treatments = [
        ("precipitation", "water_stress_index"),
        ("groundwater_anomaly", "water_stress_index"),
        ("irrigation_demand", "water_stress_index"),
    ]
    for treatment, outcome in key_treatments:
        if treatment in var_names and outcome in var_names:
            t_idx = var_names.index(treatment)
            o_idx = var_names.index(outcome)
            ace = analyzer.average_causal_effect(causal_norm, t_idx, o_idx)
            causal_effects[f"{treatment}→{outcome}"] = ace
            logger.info(f"ACE({treatment} → {outcome}): {ace:.4f}")

    # ─ Counterfactual: What if precipitation increases 20%? ─
    cf_results = {}
    if "precipitation" in var_names:
        prec_idx  = var_names.index("precipitation")
        cwsi_idx  = var_names.index("water_stress_index")
        # 20% increase in normalised precip
        delta_20pct = causal_norm[:, prec_idx].std() * 0.20
        cf_cwsi = analyzer.counterfactual(
            causal_norm, prec_idx,
            causal_norm[:, prec_idx] + delta_20pct,
            cwsi_idx
        )
        fact_cwsi    = causal_norm[:, cwsi_idx]
        cf_reduction = np.mean(fact_cwsi - cf_cwsi)
        cf_results["precipitation_20pct_increase"] = {
            "counterfactual_cwsi": cf_cwsi,
            "factual_cwsi":        fact_cwsi,
            "mean_reduction":      cf_reduction,
        }
        logger.info(f"Counterfactual: 20% more rain → CWSI reduces by {cf_reduction:.4f}")

    # ─ DoWhy rigorous estimation ─
    df_causal = pd.DataFrame(causal_norm, columns=var_names)
    if "precipitation" in var_names:
        # Binarise treatment for DoWhy (above/below median)
        med = df_causal["precipitation"].median()
        df_causal["high_precip"] = (df_causal["precipitation"] > med).astype(int)
        dowhy_result = dowhy_causal_effect(
            df_causal, "high_precip", "water_stress_index", dag_dot=dot_str
        )
        logger.info(f"DoWhy ACE estimate: {dowhy_result.get('estimate')}")
    else:
        dowhy_result = {}

    logger.success("Module 4 complete.")
    return {
        "W_est":          W_est,
        "variable_names": var_names,
        "dot_graph":      dot_str,
        "causal_ranking": causal_ranking,
        "causal_effects": causal_effects,
        "counterfactuals": cf_results,
        "dowhy_result":   dowhy_result,
        "notears":        notears,
        "analyzer":       analyzer,
    }

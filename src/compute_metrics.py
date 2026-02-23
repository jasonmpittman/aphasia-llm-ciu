# src/compute_metrics.py

from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.2.1"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

"""
Compute evaluation metrics for CIU token classification predictions.

Produces four output artefacts:

  1. summary_per_run.csv      — One row per (model, mode, seed): point estimates
                                + bootstrapped 95 % CIs for F1 and Kappa.
  2. summary_aggregated.csv   — One row per (model, mode): mean ± std across
                                seeds for every metric.
  3. summary_per_severity.csv — Per-run metrics disaggregated by severity
                                stratum (mild / moderate / severe).
  4. mcnemar_tests.csv        — Pairwise McNemar's test results between every
                                (model, mode) pair on the pooled eval set.

All stdout is also routed through the structured logger to log/.
"""

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.inter_rater import cohens_kappa

from utils import setup_logger, save_run_metadata

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def _point_estimates(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    """Return a dict of scalar metric estimates for a single (true, pred) pair."""
    if len(y_true) == 0:
        return dict(accuracy=None, precision=None, recall=None, f1=None, kappa=None)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    table = confusion_matrix(y_true, y_pred)
    try:
        kappa_result = cohens_kappa(table)
        kappa = float(kappa_result.kappa)
    except Exception:
        kappa = None

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, kappa=kappa)


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_iterations: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Return (lower, upper) bootstrap confidence interval for a scalar metric.

    Args:
        y_true:       Ground-truth labels.
        y_pred:       Predicted labels.
        metric_fn:    Callable(y_true, y_pred) -> float.
        n_iterations: Number of bootstrap resamples.
        ci:           Confidence level (default 0.95).
        seed:         RNG seed for reproducibility.

    Returns:
        (lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    scores: List[float] = []
    n = len(y_true)

    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))

    alpha = 1.0 - ci
    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return lower, upper


def _kappa_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Wrapper so kappa can be passed to _bootstrap_ci."""
    table = confusion_matrix(y_true, y_pred)
    try:
        return float(cohens_kappa(table).kappa)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# McNemar's test helpers
# ---------------------------------------------------------------------------

def _mcnemar_table(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> np.ndarray:
    """
    Build the 2×2 contingency table for McNemar's test comparing two classifiers.

    Cells:
        [0,0] both correct
        [0,1] A correct, B wrong
        [1,0] A wrong, B correct
        [1,1] both wrong
    """
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    n00 = int(np.sum( correct_a &  correct_b))
    n01 = int(np.sum( correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a &  correct_b))
    n11 = int(np.sum(~correct_a & ~correct_b))

    return np.array([[n00, n01], [n10, n11]])


def run_mcnemar_tests(
    pivot: pd.DataFrame,
    logger,
) -> pd.DataFrame:
    """
    Run pairwise McNemar's tests between every (model_name, mode) combination.

    Args:
        pivot: DataFrame with columns [token_key, (model_name, mode)_pred, ciu_label].
               token_key identifies each word-level token uniquely.
        logger: Logger instance.

    Returns:
        DataFrame with columns:
            model_a, mode_a, model_b, mode_b,
            n00, n01, n10, n11, statistic, p_value, significant_0.05
    """
    rows = []
    condition_cols = [c for c in pivot.columns if c not in ("token_key", "ciu_label")]

    for col_a, col_b in combinations(condition_cols, 2):
        y_true   = pivot["ciu_label"].values
        y_pred_a = pivot[col_a].values
        y_pred_b = pivot[col_b].values

        table = _mcnemar_table(y_true, y_pred_a, y_pred_b)

        try:
            result = mcnemar(table, exact=False, correction=True)
            stat   = float(result.statistic)
            pval   = float(result.pvalue)
        except Exception as exc:
            logger.warning("McNemar failed for %s vs %s: %s", col_a, col_b, exc)
            stat, pval = float("nan"), float("nan")

        model_a, mode_a = col_a.split("__", 1)
        model_b, mode_b = col_b.split("__", 1)

        rows.append(
            dict(
                model_a=model_a, mode_a=mode_a,
                model_b=model_b, mode_b=mode_b,
                n00=table[0, 0], n01=table[0, 1],
                n10=table[1, 0], n11=table[1, 1],
                statistic=stat,
                p_value=pval,
                significant_0_05=pval < 0.05 if not np.isnan(pval) else None,
            )
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-run metrics (step 3)
# ---------------------------------------------------------------------------

def compute_per_run_metrics(
    df: pd.DataFrame,
    n_bootstrap: int,
    logger,
) -> pd.DataFrame:
    """
    Compute point estimates + bootstrapped CIs for each (model_name, mode, seed).

    Filters to word_label == 1 before scoring (consistent with prior approach).
    """
    rows = []

    for (model_name, mode, seed), g in df.groupby(["model_name", "mode", "seed"]):
        g_word = g[g["word_label"] == 1].copy()
        y_true = g_word["ciu_label"].values
        y_pred = g_word["pred_ciu_label"].values

        pts = _point_estimates(y_true, y_pred)

        # Bootstrapped CIs for F1 and Kappa
        f1_lo, f1_hi = _bootstrap_ci(
            y_true, y_pred,
            lambda yt, yp: f1_score(yt, yp, zero_division=0),
            n_iterations=n_bootstrap,
            seed=int(seed) if seed is not None else 0,
        )
        kappa_lo, kappa_hi = _bootstrap_ci(
            y_true, y_pred,
            _kappa_from_arrays,
            n_iterations=n_bootstrap,
            seed=int(seed) if seed is not None else 0,
        )

        logger.info(
            "Per-run | model=%s mode=%s seed=%s | F1=%.3f [%.3f, %.3f] Kappa=%.3f [%.3f, %.3f]",
            model_name, mode, seed,
            pts["f1"] or 0, f1_lo, f1_hi,
            pts["kappa"] or 0, kappa_lo, kappa_hi,
        )
        logger.debug("\n%s", classification_report(y_true, y_pred, digits=3))

        rows.append(
            dict(
                model_name=model_name,
                mode=mode,
                seed=seed,
                n_tokens=len(g_word),
                **pts,
                f1_ci_lower=f1_lo,
                f1_ci_upper=f1_hi,
                kappa_ci_lower=kappa_lo,
                kappa_ci_upper=kappa_hi,
            )
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-seed aggregation (step 3 continued)
# ---------------------------------------------------------------------------

def aggregate_across_seeds(per_run: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Collapse per-run rows into mean ± std across seeds for each (model, mode).
    """
    metric_cols = ["accuracy", "precision", "recall", "f1", "kappa"]
    rows = []

    for (model_name, mode), g in per_run.groupby(["model_name", "mode"]):
        row: Dict = dict(model_name=model_name, mode=mode, n_seeds=len(g))
        for col in metric_cols:
            vals = g[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else None
            row[f"{col}_std"]  = float(vals.std())  if len(vals) > 1 else None
        logger.info(
            "Aggregated | model=%s mode=%s | F1 %.3f ± %.3f  Kappa %.3f ± %.3f",
            model_name, mode,
            row.get("f1_mean") or 0,
            row.get("f1_std")  or 0,
            row.get("kappa_mean") or 0,
            row.get("kappa_std")  or 0,
        )
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-severity breakdown (step 4)
# ---------------------------------------------------------------------------

def compute_per_severity_metrics(
    df: pd.DataFrame,
    logger,
) -> pd.DataFrame:
    """
    Compute metrics for each (model_name, mode, seed, severity) stratum.

    Requires a 'severity' column in df.  Rows where severity is null are
    grouped under the label 'unknown'.
    """
    if "severity" not in df.columns:
        logger.warning("No 'severity' column found — skipping per-severity breakdown.")
        return pd.DataFrame()

    df = df.copy()
    df["severity"] = df["severity"].fillna("unknown")

    rows = []
    for (model_name, mode, seed, severity), g in df.groupby(
        ["model_name", "mode", "seed", "severity"]
    ):
        g_word = g[g["word_label"] == 1].copy()
        y_true = g_word["ciu_label"].values
        y_pred = g_word["pred_ciu_label"].values

        pts = _point_estimates(y_true, y_pred)

        logger.info(
            "Per-severity | model=%s mode=%s seed=%s severity=%s | "
            "n=%d F1=%.3f Kappa=%.3f",
            model_name, mode, seed, severity,
            len(g_word),
            pts["f1"] or 0,
            pts["kappa"] or 0,
        )

        rows.append(
            dict(
                model_name=model_name,
                mode=mode,
                seed=seed,
                severity=severity,
                n_tokens=len(g_word),
                **pts,
            )
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@app.command()
def main(
    merged_path: Path = typer.Option(
        Path("results/parsed/llm_predictions_hf.parquet"),
        help="Parquet with ground-truth and LLM predictions merged.",
    ),
    out_dir: Path = typer.Option(
        Path("results/metrics"),
        help="Directory for all output CSVs.",
    ),
    n_bootstrap: int = typer.Option(
        1000,
        help="Number of bootstrap resamples for confidence intervals.",
    ),
    log_dir: Path = typer.Option(
        Path("log"),
        help="Directory for log files.",
    ),
) -> None:
    """
    Evaluate CIU predictions and write four metric artefacts to out_dir.
    """
    logger = setup_logger("compute_metrics", log_dir=log_dir)
    logger.info("Loading predictions from %s", merged_path)

    df = pd.read_parquet(merged_path)
    logger.info("Loaded %d rows across %d columns.", len(df), len(df.columns))

    # Inject a dummy seed column if upstream scripts haven't added one yet,
    # so the groupby logic is always consistent.
    if "seed" not in df.columns:
        logger.warning("No 'seed' column in predictions — defaulting to seed=0.")
        df["seed"] = 0

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Per-run metrics + bootstrapped CIs                               #
    # ------------------------------------------------------------------ #
    logger.info("Computing per-run metrics with %d bootstrap iterations ...", n_bootstrap)
    per_run = compute_per_run_metrics(df, n_bootstrap=n_bootstrap, logger=logger)
    per_run_path = out_dir / "summary_per_run.csv"
    per_run.to_csv(per_run_path, index=False)
    logger.info("Wrote per-run summary to %s", per_run_path)

    # ------------------------------------------------------------------ #
    # 2. Cross-seed aggregation                                           #
    # ------------------------------------------------------------------ #
    logger.info("Aggregating across seeds ...")
    aggregated = aggregate_across_seeds(per_run, logger=logger)
    agg_path = out_dir / "summary_aggregated.csv"
    aggregated.to_csv(agg_path, index=False)
    logger.info("Wrote aggregated summary to %s", agg_path)

    # ------------------------------------------------------------------ #
    # 3. Per-severity breakdown                                           #
    # ------------------------------------------------------------------ #
    logger.info("Computing per-severity metrics ...")
    per_severity = compute_per_severity_metrics(df, logger=logger)
    if not per_severity.empty:
        sev_path = out_dir / "summary_per_severity.csv"
        per_severity.to_csv(sev_path, index=False)
        logger.info("Wrote per-severity summary to %s", sev_path)

    # ------------------------------------------------------------------ #
    # 4. Pairwise McNemar's tests                                         #
    # ------------------------------------------------------------------ #
    logger.info("Running pairwise McNemar's tests ...")

    # Build a pivot so each (model, mode) prediction is its own column.
    # We pool across seeds here — McNemar operates on the full token set.
    g_word_all = df[df["word_label"] == 1].copy()
    g_word_all["token_key"] = (
        g_word_all["transcript_id"].astype(str)
        + "__"
        + g_word_all["token_index"].astype(str)
    )
    g_word_all["condition"] = (
        g_word_all["model_name"] + "__" + g_word_all["mode"]
    )

    try:
        pivot = g_word_all.pivot_table(
            index="token_key",
            columns="condition",
            values="pred_ciu_label",
            aggfunc="first",
        ).reset_index()

        # Merge in ground truth
        truth = (
            g_word_all[["token_key", "ciu_label"]]
            .drop_duplicates("token_key")
        )
        pivot = pivot.merge(truth, on="token_key")
        pivot.columns.name = None

        mcnemar_df = run_mcnemar_tests(pivot, logger=logger)
        mcnemar_path = out_dir / "mcnemar_tests.csv"
        mcnemar_df.to_csv(mcnemar_path, index=False)
        logger.info("Wrote McNemar test results to %s", mcnemar_path)

    except Exception as exc:
        logger.warning("McNemar pivot failed — skipping: %s", exc)

    # ------------------------------------------------------------------ #
    # Run metadata sidecar                                                 #
    # ------------------------------------------------------------------ #
    save_run_metadata(
        out_dir / "compute_metrics_metadata.json",
        merged_path=str(merged_path),
        n_bootstrap=n_bootstrap,
        out_dir=str(out_dir),
    )

    logger.info("compute_metrics complete.")


if __name__ == "__main__":
    app()

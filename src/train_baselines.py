# src/train_baselines.py

from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.2.1"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

"""
Train and evaluate a classic ML baseline (LinearSVC) for CIU vs NON-CIU
classification on token-level features.

The baseline is intentionally frozen — trained and evaluated before any LLM
results are inspected — so it serves as an unbiased reference point.

Features
--------
  - token_text : TF-IDF unigram–trigram representation
  - severity   : one-hot encoded severity stratum

Outputs
-------
  models/baselines/linear_svc_baseline.joblib   Serialised pipeline
  models/baselines/baseline_metrics.json        Classification report + kappa
  models/baselines/baseline_metadata.json       Run metadata sidecar
"""

from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import typer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from statsmodels.stats.inter_rater import cohens_kappa

from utils import set_global_seed, setup_logger, save_run_metadata

app = typer.Typer(add_completion=False)


@app.command()
def main(
    input_path: Path = typer.Option(
        Path("data/labeled/ciu_tokens_normalized.parquet"),
        help="Token-level dataset.",
    ),
    eval_ids_path: Path = typer.Option(
        Path("data/splits/eval_ids.txt"),
        help="Transcript IDs used for evaluation.",
    ),
    out_dir: Path = typer.Option(
        Path("models/baselines"),
        help="Where to save baseline model and metrics.",
    ),
    test_size: float = typer.Option(
        0.2,
        help="Fraction of eval-set tokens held out for testing.",
    ),
    seed: int = typer.Option(2025, help="Random seed."),
    log_dir: Path = typer.Option(Path("log"), help="Directory for log files."),
) -> None:
    """
    Train a LinearSVC baseline on the eval split and save the model + metrics.

    Note: the baseline uses only the eval-split transcripts (not prompt-support
    transcripts) consistent with the LLM evaluation design.  A stratified
    80/20 train/test split is applied within that set.
    """
    logger = setup_logger("train_baselines", log_dir=log_dir)
    logger.info("Starting baseline training — seed=%d", seed)

    set_global_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Data loading                                                         #
    # ------------------------------------------------------------------ #
    logger.info("Loading tokens from %s", input_path)
    df = pd.read_parquet(input_path)

    eval_ids = set(eval_ids_path.read_text().splitlines())
    logger.info("Eval transcript IDs loaded: %d", len(eval_ids))

    df_eval = df[df["transcript_id"].isin(eval_ids)].copy()
    df_eval = df_eval[df_eval["word_label"] == 1].reset_index(drop=True)
    logger.info(
        "Eval word tokens: %d  |  CIU positive rate: %.1f%%",
        len(df_eval),
        100.0 * df_eval["ciu_label"].mean(),
    )

    # ------------------------------------------------------------------ #
    # Feature construction                                                 #
    # ------------------------------------------------------------------ #
    X_text = df_eval["token_text"]
    X_meta = df_eval[["severity"]]
    y      = df_eval["ciu_label"]

    X = pd.concat([X_text, X_meta], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    logger.info(
        "Train/test split — train: %d  test: %d",
        len(X_train), len(X_test),
    )

    # ------------------------------------------------------------------ #
    # Pipeline                                                             #
    # ------------------------------------------------------------------ #
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 3), min_df=2), "token_text"),
            ("meta", OneHotEncoder(handle_unknown="ignore"), ["severity"]),
        ]
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LinearSVC(random_state=seed)),
        ]
    )

    logger.info("Fitting LinearSVC pipeline ...")
    clf.fit(X_train, y_train)
    logger.info("Fitting complete.")

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #
    y_pred = clf.predict(X_test)

    acc  = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec  = float(recall_score(y_test, y_pred, zero_division=0))
    f1   = float(f1_score(y_test, y_pred, zero_division=0))

    table = confusion_matrix(y_test, y_pred)
    try:
        kappa = float(cohens_kappa(table).kappa)
    except Exception:
        kappa = None

    report_str = classification_report(y_test, y_pred, digits=3)
    logger.info(
        "Baseline results — accuracy=%.3f  precision=%.3f  recall=%.3f  "
        "F1=%.3f  kappa=%s",
        acc, prec, rec, f1,
        f"{kappa:.3f}" if kappa is not None else "n/a",
    )
    logger.debug("Full classification report:\n%s", report_str)

    # ------------------------------------------------------------------ #
    # Persist model + metrics                                              #
    # ------------------------------------------------------------------ #
    model_path = out_dir / "linear_svc_baseline.joblib"
    joblib.dump(clf, model_path)
    logger.info("Saved baseline model to %s", model_path)

    metrics = {
        "model": "LinearSVC",
        "features": "tfidf_unigram_trigram + severity_onehot",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "kappa":     kappa,
        "classification_report": report_str,
    }

    metrics_path = out_dir / "baseline_metrics.json"
    with metrics_path.open("w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Saved baseline metrics to %s", metrics_path)

    save_run_metadata(
        out_dir / "baseline_metadata.json",
        input_path=str(input_path),
        eval_ids_path=str(eval_ids_path),
        seed=seed,
        test_size=test_size,
        n_eval_tokens=len(df_eval),
        n_train=len(X_train),
        n_test=len(X_test),
        f1=f1,
        kappa=kappa,
    )

    logger.info("train_baselines complete.")


if __name__ == "__main__":
    app()

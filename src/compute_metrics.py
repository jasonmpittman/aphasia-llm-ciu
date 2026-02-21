# src/compute_metrics.py
from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2025"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from pathlib import Path

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
from statsmodels.stats.inter_rater import cohens_kappa

app = typer.Typer(add_completion=False)


@app.command()
def main(
    merged_path: Path = typer.Option(
        Path("results/parsed/llm_predictions_hf.parquet"),
        help="Parquet with ground-truth and LLM predictions merged.",
    ),
    out_path: Path = typer.Option(
        Path("results/metrics/summary.csv"),
        help="CSV with aggregated metrics.",
    ),
) -> None:
    df = pd.read_parquet(merged_path)

    rows = []
    for (model_name, mode), g in df.groupby(["model_name", "mode"]):
        g_word = g[g["word_label"] == 1].copy()
        y_true = g_word["ciu_label"].values
        y_pred = g_word["pred_ciu_label"].values

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        table = confusion_matrix(y_true, y_pred)
        kappa = cohens_kappa(table)

        rows.append(
            {
                "model_name": model_name,
                "mode": mode,
                "n_tokens": len(g_word),
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "kappa": float(kappa.kappa) if kappa is not None else None,
            }
        )

        print(f"\n=== {model_name} | {mode} ===")
        print(classification_report(y_true, y_pred, digits=3))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote metrics to {out_path}")


if __name__ == "__main__":
    app()

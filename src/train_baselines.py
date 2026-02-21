# src/train_baselines.py
from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2025"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

"""
Train classic ML baselines for CIU vs NON-CIU on token-level features.
"""

from pathlib import Path

import joblib
import pandas as pd
import typer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from utils import set_global_seed

app = typer.Typer(add_completion=False)


@app.command()
def main(
    input_path: Path = typer.Option(
        Path("data/labeled/ciu_tokens_normalized.parquet"), help="Token-level dataset."
    ),
    eval_ids_path: Path = typer.Option(
        Path("data/splits/eval_ids.txt"), help="Transcript IDs used for evaluation."
    ),
    out_dir: Path = typer.Option(
        Path("models/baselines"), help="Where to save baseline models."
    ),
    seed: int = typer.Option(2025, help="Random seed."),
) -> None:
    set_global_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    eval_ids = set(eval_ids_path.read_text().splitlines())

    df_eval = df[df["transcript_id"].isin(eval_ids)].copy()
    df_eval = df_eval[df_eval["word_label"] == 1].reset_index(drop=True)

    X_text = df_eval["token_text"]
    X_meta = df_eval[["severity"]]
    y = df_eval["ciu_label"]

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

    X = pd.concat([X_text, X_meta], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Baseline LinearSVC classification report (CIU vs NON-CIU):")
    print(classification_report(y_test, y_pred, digits=3))

    model_path = out_dir / "linear_svc_baseline.joblib"
    joblib.dump(clf, model_path)
    print(f"Saved baseline model to {model_path}")


if __name__ == "__main__":
    app()

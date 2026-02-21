# src/data_prep.py
from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2025"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

"""
Normalize the labeled token dataset from the prior CIU study into a canonical format.

Input:  data/labeled/ciu_tokens.csv
Output: data/labeled/ciu_tokens_normalized.parquet
"""

from pathlib import Path

import pandas as pd
import typer

from utils import set_global_seed

app = typer.Typer(add_completion=False)


@app.command()
def main(
    input_path: Path = typer.Option(
        Path("data/labeled/ciu_tokens.csv"),
        help="CSV with labeled tokens from prior study.",
    ),
    output_path: Path = typer.Option(
        Path("data/labeled/ciu_tokens_normalized.parquet"),
        help="Output path for normalized token table.",
    ),
    seed: int = typer.Option(2025, help="Random seed (for any shuffling)."),
) -> None:
    """
    Load labeled tokens, validate invariants, and write a normalized parquet file.
    """
    set_global_seed(seed)

    df = pd.read_csv(input_path)

    expected_cols = {
        "transcript_id",
        "token_index",
        "token_text",
        "word_label",
        "ciu_label",
        "speaker_id",
        "severity",
    }
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in labeled CSV: {missing}")

    bad_rows = df[(df["ciu_label"] == 1) & (df["word_label"] == 0)]
    if not bad_rows.empty:
        raise ValueError(
            f"Found {len(bad_rows)} rows with ciu_label=1 but word_label=0. "
            f"Please fix before proceeding."
        )

    df = df.sort_values(["transcript_id", "token_index"]).reset_index(drop=True)
    df["word_label"] = df["word_label"].astype(int)
    df["ciu_label"] = df["ciu_label"].astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    print(f"Wrote normalized labeled tokens to {output_path}")


if __name__ == "__main__":
    app()

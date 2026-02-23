# src/data_prep.py

from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.2.1"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

"""
Normalize the labeled token dataset from the prior CIU study into a canonical
format.

Input:  data/labeled/ciu_tokens.csv
Output: data/labeled/ciu_tokens_normalized.parquet
"""

from pathlib import Path

import pandas as pd
import typer

from utils import set_global_seed, setup_logger, save_run_metadata

app = typer.Typer(add_completion=False)

EXPECTED_COLS = {
    "transcript_id",
    "token_index",
    "token_text",
    "word_label",
    "ciu_label",
    "speaker_id",
    "severity",
}


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
    log_dir: Path = typer.Option(Path("log"), help="Directory for log files."),
) -> None:
    """
    Load labeled tokens, validate invariants, and write a normalized parquet file.
    """
    logger = setup_logger("data_prep", log_dir=log_dir)
    logger.info("Starting data preparation — input=%s", input_path)

    set_global_seed(seed)

    logger.info("Loading CSV from %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows, %d columns.", len(df), len(df.columns))

    # ------------------------------------------------------------------ #
    # Schema validation                                                    #
    # ------------------------------------------------------------------ #
    missing_cols = EXPECTED_COLS.difference(df.columns)
    if missing_cols:
        logger.error("Missing expected columns: %s", missing_cols)
        raise ValueError(f"Missing expected columns in labeled CSV: {missing_cols}")
    logger.info("Schema validation passed.")

    # ------------------------------------------------------------------ #
    # Invariant checks                                                     #
    # ------------------------------------------------------------------ #
    bad_rows = df[(df["ciu_label"] == 1) & (df["word_label"] == 0)]
    if not bad_rows.empty:
        logger.error(
            "Found %d rows with ciu_label=1 but word_label=0. "
            "Sample transcript IDs: %s",
            len(bad_rows),
            bad_rows["transcript_id"].unique()[:5].tolist(),
        )
        raise ValueError(
            f"Found {len(bad_rows)} rows with ciu_label=1 but word_label=0. "
            f"Please fix before proceeding."
        )
    logger.info("Invariant check passed — no CIU tokens with word_label=0.")

    # ------------------------------------------------------------------ #
    # Normalisation                                                        #
    # ------------------------------------------------------------------ #
    df = df.sort_values(["transcript_id", "token_index"]).reset_index(drop=True)
    df["word_label"] = df["word_label"].astype(int)
    df["ciu_label"]  = df["ciu_label"].astype(int)

    n_transcripts = df["transcript_id"].nunique()
    n_words       = int((df["word_label"] == 1).sum())
    n_ciu         = int((df["ciu_label"]  == 1).sum())
    logger.info(
        "Dataset summary: %d tokens | %d transcripts | %d words | %d CIU tokens (%.1f%% of words)",
        len(df), n_transcripts, n_words, n_ciu,
        100.0 * n_ciu / n_words if n_words > 0 else 0.0,
    )

    # ------------------------------------------------------------------ #
    # Write output                                                         #
    # ------------------------------------------------------------------ #
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    logger.info("Wrote normalized labeled tokens to %s", output_path)

    save_run_metadata(
        output_path.parent / "data_prep_metadata.json",
        input_path=str(input_path),
        output_path=str(output_path),
        seed=seed,
        n_rows=len(df),
        n_transcripts=n_transcripts,
        n_words=n_words,
        n_ciu=n_ciu,
    )

    logger.info("data_prep complete.")


if __name__ == "__main__":
    app()

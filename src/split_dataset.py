# src/split_dataset.py

from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.2.1"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

"""
Create stratified prompt-support and evaluation splits at transcript level.

Stratification is performed by severity so that the distribution of severity
strata is approximately preserved in both the prompt-support set and the eval
set.  The split is deterministic given a fixed seed and prompt_n.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import typer

from utils import set_global_seed, setup_logger, save_run_metadata

app = typer.Typer(add_completion=False)


def stratified_split_transcripts(
    df: pd.DataFrame,
    prompt_n: int,
    seed: int,
    logger,
) -> Tuple[List[str], List[str]]:
    """
    Split transcript IDs into prompt-support and eval sets, stratified by
    severity.

    Args:
        df:       Full normalized token DataFrame (requires transcript_id and
                  severity columns).
        prompt_n: Target size of the prompt-support set.  Actual size may
                  differ slightly due to per-stratum rounding.
        seed:     RNG seed for reproducibility.
        logger:   Logger instance.

    Returns:
        (prompt_ids, eval_ids) — both sorted lists of transcript ID strings.
    """
    set_global_seed(seed)

    meta        = df[["transcript_id", "severity"]].drop_duplicates()
    transcripts = np.array(meta["transcript_id"].tolist())
    severities  = np.array(meta["severity"].tolist())
    n_total     = len(transcripts)

    logger.info(
        "Total transcripts: %d  |  target prompt_n: %d  |  seed: %d",
        n_total, prompt_n, seed,
    )

    unique_sev = np.unique(severities)
    logger.info("Severity strata: %s", unique_sev.tolist())

    prompt_ids: List[str] = []

    for sev in unique_sev:
        mask    = severities == sev
        sev_ids = transcripts[mask]
        n_sev   = len(sev_ids)

        n_prompt_sev = max(1, int(round(prompt_n * (n_sev / n_total))))
        n_chosen     = min(n_prompt_sev, n_sev)
        chosen       = np.random.choice(sev_ids, size=n_chosen, replace=False)
        prompt_ids.extend(chosen.tolist())

        logger.info(
            "  Severity %-12s : %d transcripts total  →  %d selected for prompt set",
            sev, n_sev, n_chosen,
        )

    prompt_ids = sorted(set(prompt_ids))
    eval_ids   = sorted(t for t in transcripts.tolist() if t not in prompt_ids)

    logger.info(
        "Split complete — prompt: %d  eval: %d  (total: %d)",
        len(prompt_ids), len(eval_ids), len(prompt_ids) + len(eval_ids),
    )

    return prompt_ids, eval_ids


@app.command()
def main(
    input_path: Path = typer.Option(
        Path("data/labeled/ciu_tokens_normalized.parquet"),
        help="Input normalized token file.",
    ),
    prompt_n: int = typer.Option(
        30,
        help="Approximate number of transcripts in prompt-support set.",
    ),
    out_dir: Path = typer.Option(
        Path("data/splits"),
        help="Output directory for split ID lists.",
    ),
    seed: int = typer.Option(2025, help="Random seed."),
    log_dir: Path = typer.Option(Path("log"), help="Directory for log files."),
) -> None:
    """
    Produce prompt_ids.txt and eval_ids.txt under out_dir via stratified split.

    The split is frozen after this step — run_all.sh calls this once and all
    subsequent inference and fine-tuning steps consume the same split.
    """
    logger = setup_logger("split_dataset", log_dir=log_dir)
    logger.info("Loading normalized tokens from %s", input_path)

    df = pd.read_parquet(input_path)
    logger.info("Loaded %d rows.", len(df))

    prompt_ids, eval_ids = stratified_split_transcripts(
        df, prompt_n=prompt_n, seed=seed, logger=logger
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt_ids.txt").write_text("\n".join(prompt_ids))
    (out_dir / "eval_ids.txt").write_text("\n".join(eval_ids))
    logger.info(
        "Wrote prompt_ids.txt (%d IDs) and eval_ids.txt (%d IDs) to %s",
        len(prompt_ids), len(eval_ids), out_dir,
    )

    save_run_metadata(
        out_dir / "split_metadata.json",
        input_path=str(input_path),
        prompt_n_target=prompt_n,
        prompt_n_actual=len(prompt_ids),
        eval_n=len(eval_ids),
        seed=seed,
    )

    logger.info("split_dataset complete.")


if __name__ == "__main__":
    app()

# src/split_dataset.py
from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2025"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

"""
Create prompt-support and evaluation splits at transcript level.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import typer

from utils import set_global_seed

app = typer.Typer(add_completion=False)


def stratified_split_transcripts(
    df: pd.DataFrame, prompt_n: int, seed: int
) -> tuple[List[str], List[str]]:
    set_global_seed(seed)

    meta = df[["transcript_id", "severity"]].drop_duplicates()
    transcripts = meta["transcript_id"].tolist()
    severities = meta["severity"].tolist()

    transcripts = np.array(transcripts)
    severities = np.array(severities)

    unique_sev = np.unique(severities)
    prompt_ids: List[str] = []

    for sev in unique_sev:
        mask = severities == sev
        sev_ids = transcripts[mask]
        n_sev = len(sev_ids)
        if n_sev == 0:
            continue

        n_prompt_sev = max(1, int(round(prompt_n * (n_sev / len(transcripts)))))
        chosen = np.random.choice(sev_ids, size=min(n_prompt_sev, n_sev), replace=False)
        prompt_ids.extend(chosen.tolist())

    prompt_ids = sorted(set(prompt_ids))
    eval_ids = sorted(t for t in transcripts.tolist() if t not in prompt_ids)
    return prompt_ids, eval_ids


@app.command()
def main(
    input_path: Path = typer.Option(
        Path("data/labeled/ciu_tokens_normalized.parquet"),
        help="Input normalized token file.",
    ),
    prompt_n: int = typer.Option(30, help="Approximate number of transcripts in prompt set."),
    out_dir: Path = typer.Option(Path("data/splits"), help="Output directory for split ID lists."),
    seed: int = typer.Option(2025, help="Random seed."),
) -> None:
    df = pd.read_parquet(input_path)
    prompt_ids, eval_ids = stratified_split_transcripts(df, prompt_n=prompt_n, seed=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt_ids.txt").write_text("\n".join(prompt_ids))
    (out_dir / "eval_ids.txt").write_text("\n".join(eval_ids))

    print(f"Prompt transcripts: {len(prompt_ids)}, Eval transcripts: {len(eval_ids)}")


if __name__ == "__main__":
    app()

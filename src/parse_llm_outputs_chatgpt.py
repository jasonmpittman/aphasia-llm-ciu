# src/parse_llm_outputs_chatgpt.py
from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2025"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

"""
Parse manual ChatGPT outputs (plain text files with JSON arrays) and merge with ground-truth.
"""

import json
from pathlib import Path
from typing import List

import pandas as pd
import typer
from tqdm import tqdm

app = typer.Typer(add_completion=False)


def extract_json_array(text: str) -> List[dict]:
    text = text.strip()
    first = text.find("[")
    last = text.rfind("]")
    if first == -1 or last == -1:
        raise ValueError("Could not find JSON array in response.")
    json_str = text[first : last + 1]
    return json.loads(json_str)


@app.command()
def main(
    labeled_path: Path = typer.Option(
        Path("data/labeled/ciu_tokens_normalized.parquet"),
        help="Ground-truth labels.",
    ),
    raw_dir: Path = typer.Option(
        Path("results/raw/chatgpt/z_shot_local"),
        help="Directory of plain-text files with ChatGPT responses.",
    ),
    out_path: Path = typer.Option(
        Path("results/parsed/llm_predictions_chatgpt.parquet"),
        help="Output parquet with predictions+labels.",
    ),
    model_name: str = typer.Option(
        "chatgpt-webui", help="Model label to store in output."
    ),
    mode: str = typer.Option(
        "z_shot_local", help="Prompting mode label to store in output."
    ),
) -> None:
    labeled = pd.read_parquet(labeled_path)
    labeled = labeled.sort_values(["transcript_id", "token_index"]).reset_index(drop=True)

    rows = []
    raw_dir = Path(raw_dir)

    for txt_file in tqdm(list(raw_dir.glob("*.txt")), desc="Parsing ChatGPT outputs"):
        group_id = txt_file.stem
        resp_text = txt_file.read_text()

        # group_id could be "CR_001" or "CR_001__utt-3"; we only need transcript_id
        transcript_id = group_id.split("__utt-")[0]

        try:
            preds = extract_json_array(resp_text)
        except Exception as e:
            print(f"Failed to parse {txt_file}: {e}")
            continue

        for p in preds:
            rows.append(
                {
                    "transcript_id": transcript_id,
                    "model_name": model_name,
                    "mode": mode,
                    "pred_index": p["index"],
                    "pred_token": p["token"],
                    "pred_word_label": int(p["word_label"]),
                    "pred_ciu_label": int(p["ciu_label"]),
                }
            )

    pred_df = pd.DataFrame(rows)

    merged = labeled.merge(
        pred_df,
        left_on=["transcript_id", "token_index"],
        right_on=["transcript_id", "pred_index"],
        how="inner",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path)
    print(f"Wrote merged ChatGPT predictions+labels to {out_path}")


if __name__ == "__main__":
    app()

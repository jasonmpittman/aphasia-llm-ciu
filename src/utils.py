# src/utils.py
from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2025"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from typing import List, Optional
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


@dataclass
class Config:
    """Simple wrapper around a dict-based config."""
    data: Dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(data=data)

    def __getitem__(self, item: str) -> Any:
        return self.data[item]


def set_global_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def get_model_config(cfg: "Config", model_key: str) -> dict:
    models = cfg["models"]
    if model_key not in models:
        raise KeyError(f"Unknown model_key '{model_key}'. Available: {list(models.keys())}")
    return models[model_key]


def build_few_shot_block(
    df,
    prompt_ids: List[str],
    n_examples: int = 3,
    seed: Optional[int] = None,
    group_by_utterance: bool = True,
) -> Tuple[str, List[dict]]:
    """
    Build a human-readable few-shot block from the prompt-support set AND
    return metadata about which examples were chosen.

    - df: full labeled token DataFrame (not filtered to eval).
    - prompt_ids: transcript_ids that belong to the prompt-support set P.
    - n_examples: how many utterances (or transcripts) to sample.
    - seed: RNG seed for reproducibility.
    - group_by_utterance:
        * True  -> group by (transcript_id, utterance_id) if available.
        * False -> group by transcript_id only.

    Returns:
        (text_block, metadata)

        text_block: string to drop into {{few_shot_examples}}.
        metadata: list of dicts like:
          {
            "transcript_id": ...,
            "utterance_id": ...,
            "group_id": ...,
            "n_tokens": ...,
          }
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    df_p = df[df["transcript_id"].isin(prompt_ids)].copy()
    if df_p.empty:
        print("[build_few_shot_block] WARNING: prompt_ids produced an empty subset.")
        return "", []

    df_p = df_p.sort_values(["transcript_id", "token_index"]).reset_index(drop=True)

    if group_by_utterance and "utterance_id" in df_p.columns:
        group_cols = ["transcript_id", "utterance_id"]
    else:
        group_cols = ["transcript_id"]

    groups = list(df_p.groupby(group_cols))
    if not groups:
        print("[build_few_shot_block] WARNING: no groups found for few-shot construction.")
        return "", []

    n = min(n_examples, len(groups))
    idxs = rng.choice(len(groups), size=n, replace=False)

    pieces = []
    metadata: List[dict] = []

    for idx in idxs:
        keys, g = groups[idx]
        if isinstance(keys, tuple):
            transcript_id = keys[0]
            utterance_id = keys[1] if len(keys) > 1 else None
        else:
            transcript_id = keys
            utterance_id = None

        tokens = g["token_text"].tolist()
        word_labels = g["word_label"].astype(int).tolist()
        ciu_labels = g["ciu_label"].astype(int).tolist()

        group_id = (
            f"{transcript_id}__utt-{utterance_id}" if utterance_id is not None else transcript_id
        )

        token_block = "\n".join(f"{i}: {tok}" for i, tok in enumerate(tokens))

        records = [
            {
                "index": i,
                "token": tok,
                "word_label": wl,
                "ciu_label": cl,
            }
            for i, (tok, wl, cl) in enumerate(zip(tokens, word_labels, ciu_labels))
        ]
        labels_json = json.dumps(records, indent=2)

        piece = (
            f"Example Utterance ID: {group_id}\n"
            f"Tokens:\n{token_block}\n\n"
            f"Labels:\n{labels_json}"
        )
        pieces.append(piece)

        metadata.append(
            {
                "transcript_id": transcript_id,
                "utterance_id": utterance_id,
                "group_id": group_id,
                "n_tokens": len(tokens),
            }
        )

    return "\n\n".join(pieces), metadata
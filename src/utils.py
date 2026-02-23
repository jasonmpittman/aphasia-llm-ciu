# src/utils.py

from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.2.1"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from typing import List, Optional, Tuple
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(
    name: str,
    log_dir: Path = Path("log"),
    level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Create a logger that writes to both the console (INFO+) and a timestamped
    file under log_dir (DEBUG+).

    Each call creates a new handler set, so it is safe to call once per script
    entry-point.  Subsequent calls with the same *name* within the same process
    return the already-configured logger without adding duplicate handlers.

    Args:
        name:    Logger name – use the script stem, e.g. "run_llm_inference".
        log_dir: Directory for log files (created if absent).
        level:   Root log level (default DEBUG captures everything).

    Returns:
        Configured :class:`logging.Logger`.

    Example::

        logger = setup_logger("run_llm_inference")
        logger.info("Starting inference run")
        logger.warning("Model output was truncated for group %s", group_id)
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called more than once in the same process
    if logger.handlers:
        return logger

    logger.setLevel(level)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = log_dir / f"{name}_{timestamp}.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    # File handler — DEBUG and above
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Log file: %s", log_file)
    return logger


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

def save_run_metadata(path: str | Path, **kwargs: Any) -> None:
    """
    Write a JSON sidecar that describes the conditions under which an output
    artifact was produced.

    Automatically includes a UTC timestamp and the versions of key libraries
    (transformers, peft, torch, pandas, sklearn) if they are importable.

    Args:
        path:    Destination path for the JSON file.
        **kwargs: Arbitrary key/value pairs to include (e.g. model_key, mode,
                  seed, adapter_dir).

    Example::

        save_run_metadata(
            out_dir / "run_metadata.json",
            model_key=model_key,
            mode=mode,
            seed=seed,
            use_lora=use_lora,
        )
    """
    def _ver(pkg: str) -> str:
        try:
            import importlib.metadata as im
            return im.version(pkg)
        except Exception:
            return "unavailable"

    meta: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "library_versions": {
            "transformers": _ver("transformers"),
            "peft":         _ver("peft"),
            "torch":        _ver("torch"),
            "pandas":       _ver("pandas"),
            "scikit-learn": _ver("scikit-learn"),
            "statsmodels":  _ver("statsmodels"),
        },
    }
    meta.update(kwargs)
    save_json(meta, path)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def get_model_config(cfg: "Config", model_key: str) -> dict:
    models = cfg["models"]
    if model_key not in models:
        raise KeyError(
            f"Unknown model_key '{model_key}'. Available: {list(models.keys())}"
        )
    return models[model_key]


# ---------------------------------------------------------------------------
# Few-shot construction
# ---------------------------------------------------------------------------

def build_few_shot_block(
    df,
    prompt_ids: List[str],
    n_examples: int = 3,
    seed: Optional[int] = None,
    group_by_utterance: bool = True,
    strategy: str = "random",
) -> Tuple[str, List[dict]]:
    """
    Build a human-readable few-shot block from the prompt-support set AND
    return metadata about which examples were chosen.

    Args:
        df:                 Full labeled token DataFrame (not filtered to eval).
        prompt_ids:         Transcript IDs that belong to the prompt-support set P.
        n_examples:         How many utterances (or transcripts) to sample.
        seed:               RNG seed for reproducibility.
        group_by_utterance: If True, group by (transcript_id, utterance_id);
                            otherwise group by transcript_id only.
        strategy:           Sampling strategy – "random" (default) or
                            "severity_stratified".  The latter samples
                            proportionally from each severity stratum so that
                            the few-shot block mirrors the severity distribution
                            of the prompt-support set.

    Returns:
        (text_block, metadata)

        text_block: String to drop into ``{{few_shot_examples}}``.
        metadata:   List of dicts, each with keys
                    ``transcript_id``, ``utterance_id``, ``group_id``,
                    ``n_tokens``.
    """
    rng = np.random.default_rng(seed)

    df_p = df[df["transcript_id"].isin(prompt_ids)].copy()
    if df_p.empty:
        logging.warning("[build_few_shot_block] prompt_ids produced an empty subset.")
        return "", []

    df_p = df_p.sort_values(["transcript_id", "token_index"]).reset_index(drop=True)

    if group_by_utterance and "utterance_id" in df_p.columns:
        group_cols = ["transcript_id", "utterance_id"]
    else:
        group_cols = ["transcript_id"]

    groups = list(df_p.groupby(group_cols))
    if not groups:
        logging.warning("[build_few_shot_block] No groups found for few-shot construction.")
        return "", []

    # ------------------------------------------------------------------ #
    # Sampling strategies                                                  #
    # ------------------------------------------------------------------ #
    if strategy == "severity_stratified" and "severity" in df_p.columns:
        # Build a severity label per group (take the first row's severity)
        group_severities = []
        for keys, g in groups:
            sev = g["severity"].iloc[0]
            group_severities.append(sev)

        unique_sev = np.unique(group_severities)
        chosen_idxs: List[int] = []
        for sev in unique_sev:
            sev_idxs = [i for i, s in enumerate(group_severities) if s == sev]
            n_sev = max(1, int(round(n_examples * len(sev_idxs) / len(groups))))
            chosen = rng.choice(sev_idxs, size=min(n_sev, len(sev_idxs)), replace=False)
            chosen_idxs.extend(chosen.tolist())
        # Trim to n_examples in case rounding over-selected
        rng.shuffle(chosen_idxs)
        chosen_idxs = chosen_idxs[:n_examples]
    else:
        # Default: uniform random
        n = min(n_examples, len(groups))
        chosen_idxs = rng.choice(len(groups), size=n, replace=False).tolist()

    # ------------------------------------------------------------------ #
    # Build text blocks                                                    #
    # ------------------------------------------------------------------ #
    pieces: List[str] = []
    metadata: List[dict] = []

    for idx in chosen_idxs:
        keys, g = groups[idx]
        if isinstance(keys, tuple):
            transcript_id = keys[0]
            utterance_id = keys[1] if len(keys) > 1 else None
        else:
            transcript_id = keys
            utterance_id = None

        tokens      = g["token_text"].tolist()
        word_labels = g["word_label"].astype(int).tolist()
        ciu_labels  = g["ciu_label"].astype(int).tolist()

        group_id = (
            f"{transcript_id}__utt-{utterance_id}"
            if utterance_id is not None
            else transcript_id
        )

        token_block = "\n".join(f"{i}: {tok}" for i, tok in enumerate(tokens))
        records = [
            {
                "index":      i,
                "token":      tok,
                "word_label": wl,
                "ciu_label":  cl,
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
                "utterance_id":  utterance_id,
                "group_id":      group_id,
                "n_tokens":      len(tokens),
            }
        )

    return "\n\n".join(pieces), metadata

# src/parse_llm_outputs_chatgpt.py

from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.2.1"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

"""
Parse manual ChatGPT outputs (plain .txt files containing JSON arrays) and
merge with ground-truth labels.

Applies the same robustness logic as parse_llm_outputs_hf.py:

  - Multi-strategy JSON extraction (bracket scan → regex → truncation repair)
  - Required-field validation
  - Token-count mismatch detection with configurable drop / truncate strategy
  - Parse quality report written alongside the merged parquet
  - Run metadata sidecar

Failure categories are identical to the HF parser so quality reports across
both pipelines are directly comparable.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
from tqdm import tqdm

from utils import setup_logger, save_run_metadata, save_json

# Re-use the shared parsing primitives from the HF parser
from parse_llm_outputs_hf import (
    ParseFailure,
    ParseQualityReport,
    extract_json_array,
    validate_records,
    handle_mismatch,
    FAILURE_TOKEN_MISMATCH,
)

app = typer.Typer(add_completion=False)


def parse_chatgpt_directory(
    raw_dir: Path,
    labeled: pd.DataFrame,
    model_name: str,
    mode: str,
    seed: Optional[int],
    mismatch_strategy: str,
    report: ParseQualityReport,
    logger,
) -> List[dict]:
    """
    Walk *raw_dir* for *.txt response files, parse each, and return merged rows.

    File naming convention: <group_id>.txt, where group_id is either a plain
    transcript_id (e.g. "CR_001") or "CR_001__utt-3".
    """
    txt_files = sorted(raw_dir.glob("*.txt"))
    report.total_files = len(txt_files)
    logger.info("Found %d ChatGPT response files in %s", report.total_files, raw_dir)

    rows: List[dict] = []

    for txt_file in tqdm(txt_files, desc="Parsing ChatGPT outputs"):
        group_id      = txt_file.stem
        transcript_id = group_id.split("__utt-")[0]
        utterance_id  = (
            group_id.split("__utt-")[1]
            if "__utt-" in group_id
            else None
        )

        resp_text = txt_file.read_text(encoding="utf-8")

        # ---------------------------------------------------------------- #
        # Extract + validate JSON array                                     #
        # ---------------------------------------------------------------- #
        preds, failure_type = extract_json_array(resp_text)

        if failure_type:
            report.failures.append(ParseFailure(
                json_file=str(txt_file),
                transcript_id=transcript_id,
                group_id=group_id,
                model_name=model_name,
                mode=mode,
                failure_type=failure_type,
                detail=resp_text[:200],
            ))
            logger.warning(
                "Parse failure [%s] — file=%s group=%s",
                failure_type, txt_file.name, group_id,
            )
            continue

        preds, val_failure = validate_records(preds)
        if val_failure:
            report.failures.append(ParseFailure(
                json_file=str(txt_file),
                transcript_id=transcript_id,
                group_id=group_id,
                model_name=model_name,
                mode=mode,
                failure_type=val_failure,
                detail=val_failure,
            ))
            logger.warning(
                "Validation failure [%s] — file=%s group=%s",
                val_failure, txt_file.name, group_id,
            )
            continue

        # ---------------------------------------------------------------- #
        # Token-count mismatch check                                       #
        # ---------------------------------------------------------------- #
        if utterance_id is not None:
            gt_group = labeled[
                (labeled["transcript_id"] == transcript_id) &
                (labeled["utterance_id"]   == utterance_id)
            ]
        else:
            gt_group = labeled[labeled["transcript_id"] == transcript_id]

        if len(preds) != len(gt_group):
            preds = handle_mismatch(
                preds=preds,
                gt_tokens=gt_group,
                strategy=mismatch_strategy,
                report=report,
                group_id=group_id,
                logger=logger,
            )
            if preds is None:
                report.failures.append(ParseFailure(
                    json_file=str(txt_file),
                    transcript_id=transcript_id,
                    group_id=group_id,
                    model_name=model_name,
                    mode=mode,
                    failure_type=FAILURE_TOKEN_MISMATCH,
                    detail=f"strategy=drop",
                ))
                continue

        # ---------------------------------------------------------------- #
        # Build rows                                                        #
        # ---------------------------------------------------------------- #
        report.success += 1
        for p in preds:
            rows.append(
                {
                    "transcript_id":   transcript_id,
                    "utterance_id":    utterance_id,
                    "group_id":        group_id,
                    "model_name":      model_name,
                    "mode":            mode,
                    "seed":            seed,
                    "pred_index":      int(p["index"]),
                    "pred_token":      p["token"],
                    "pred_word_label": int(p["word_label"]),
                    "pred_ciu_label":  int(p["ciu_label"]),
                }
            )

    return rows


@app.command()
def main(
    labeled_path: Path = typer.Option(
        Path("data/labeled/ciu_tokens_normalized.parquet"),
        help="Ground-truth labels.",
    ),
    raw_dir: Path = typer.Option(
        Path("results/raw/chatgpt/z_shot_local"),
        help="Directory of .txt files with ChatGPT responses.",
    ),
    out_path: Path = typer.Option(
        Path("results/parsed/llm_predictions_chatgpt.parquet"),
        help="Output parquet with predictions merged with ground-truth labels.",
    ),
    model_name: str = typer.Option(
        "chatgpt-webui",
        help="Model label to embed in output rows.",
    ),
    mode: str = typer.Option(
        "z_shot_local",
        help="Prompting mode label to embed in output rows.",
    ),
    seed: Optional[int] = typer.Option(
        None,
        help="Seed label to embed in output rows (informational only).",
    ),
    mismatch_strategy: str = typer.Option(
        "drop",
        help="How to handle token-count mismatches: 'drop' or 'truncate'.",
    ),
    log_dir: Path = typer.Option(
        Path("log"), help="Directory for log files."
    ),
) -> None:
    logger = setup_logger("parse_llm_outputs_chatgpt", log_dir=log_dir)
    logger.info(
        "Parsing ChatGPT outputs — raw_dir=%s  model=%s  mode=%s",
        raw_dir, model_name, mode,
    )

    labeled = pd.read_parquet(labeled_path)
    labeled = labeled.sort_values(["transcript_id", "token_index"]).reset_index(drop=True)
    logger.info("Loaded %d ground-truth token rows.", len(labeled))

    report = ParseQualityReport()

    rows = parse_chatgpt_directory(
        raw_dir=raw_dir,
        labeled=labeled,
        model_name=model_name,
        mode=mode,
        seed=seed,
        mismatch_strategy=mismatch_strategy,
        report=report,
        logger=logger,
    )

    pred_df = pd.DataFrame(rows)

    if pred_df.empty:
        logger.error(
            "No predictions were successfully parsed from %s. "
            "Check the parse quality report for details.",
            raw_dir,
        )
        raise typer.Exit(code=1)

    merged = labeled.merge(
        pred_df,
        left_on=["transcript_id", "token_index"],
        right_on=["transcript_id", "pred_index"],
        how="inner",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path)
    logger.info(
        "Wrote %d merged rows (%d unique transcripts) to %s",
        len(merged),
        merged["transcript_id"].nunique(),
        out_path,
    )

    # ------------------------------------------------------------------ #
    # Parse quality report                                                 #
    # ------------------------------------------------------------------ #
    quality_dict     = report.to_dict()
    quality_csv_path = out_path.parent / (out_path.stem + "_parse_quality.csv")
    quality_json_path= out_path.parent / (out_path.stem + "_parse_quality.json")

    pd.DataFrame(report.failures).to_csv(quality_csv_path, index=False) \
        if report.failures else \
        pd.DataFrame(
            columns=["json_file", "transcript_id", "group_id",
                     "model_name", "mode", "failure_type", "detail"]
        ).to_csv(quality_csv_path, index=False)

    save_json(quality_dict, quality_json_path)

    logger.info(
        "Parse quality: %d/%d files parsed successfully (failure rate=%.1f%%). "
        "Token mismatches: %d  Tokens dropped: %d  Tokens truncated: %d",
        report.success,
        report.total_files,
        report.failure_rate() * 100,
        report.token_mismatches,
        report.tokens_dropped,
        report.tokens_truncated,
    )

    if report.failure_rate() > 0.10:
        logger.warning(
            "Failure rate %.1f%% exceeds 10%% threshold — review parse quality "
            "report before interpreting metrics.",
            report.failure_rate() * 100,
        )

    for ftype, count in report.failure_counts().items():
        logger.info("  Failure type %-30s : %d", ftype, count)

    # ------------------------------------------------------------------ #
    # Run metadata sidecar                                                 #
    # ------------------------------------------------------------------ #
    save_run_metadata(
        out_path.parent / (out_path.stem + "_metadata.json"),
        raw_dir=str(raw_dir),
        labeled_path=str(labeled_path),
        model_name=model_name,
        mode=mode,
        seed=seed,
        mismatch_strategy=mismatch_strategy,
        total_files=report.total_files,
        parse_success=report.success,
        failure_rate=report.failure_rate(),
    )

    logger.info("parse_llm_outputs_chatgpt complete.")


if __name__ == "__main__":
    app()

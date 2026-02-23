# src/parse_llm_outputs_hf.py

from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.2.1"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

"""
Parse HF LLM output JSON wrappers and merge with ground-truth labels.

Robustness improvements over v0.1.0:

  - Every parse failure is categorised and logged rather than silently skipped.
  - Token-count mismatches between predicted and ground-truth sequences are
    detected, logged, and handled via a configurable fallback strategy.
  - A parse quality report (parse_quality.csv + parse_quality.json) is written
    alongside the merged parquet so failure rates are always visible in results.
  - All output artefacts include a run-metadata sidecar.

Failure categories
------------------
  no_json_array       Response contained no recognisable JSON array.
  json_decode_error   A JSON array was found but json.loads() raised an error.
  missing_fields      Parsed records were missing required keys.
  token_count_mismatch  Predicted token count differs from ground-truth count.
  empty_response      response_text was empty or whitespace-only.

Fallback strategies (--mismatch-strategy)
-----------------------------------------
  drop   (default) Exclude the group from the merged output entirely.
           Safe for metric computation — the group simply contributes no rows.
  truncate  Align on the shorter of (pred, truth) by token_index.
           Useful for exploratory analysis; note in paper if used.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import typer
from tqdm import tqdm

from utils import setup_logger, save_run_metadata, save_json

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------

FAILURE_EMPTY          = "empty_response"
FAILURE_NO_ARRAY       = "no_json_array"
FAILURE_JSON_DECODE    = "json_decode_error"
FAILURE_MISSING_FIELDS = "missing_fields"
FAILURE_TOKEN_MISMATCH = "token_count_mismatch"

REQUIRED_FIELDS = {"index", "token", "word_label", "ciu_label"}


@dataclass
class ParseFailure:
    json_file: str
    transcript_id: Optional[str]
    group_id: Optional[str]
    model_name: Optional[str]
    mode: Optional[str]
    failure_type: str
    detail: str


@dataclass
class ParseQualityReport:
    total_files: int = 0
    success: int = 0
    failures: List[ParseFailure] = field(default_factory=list)
    token_mismatches: int = 0     # files with mismatch that were dropped/truncated
    tokens_dropped: int = 0       # individual tokens lost due to mismatch+drop
    tokens_truncated: int = 0     # individual tokens lost due to mismatch+truncate

    # Counts by failure type
    def failure_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for f in self.failures:
            counts[f.failure_type] = counts.get(f.failure_type, 0) + 1
        return counts

    def failure_rate(self) -> float:
        if self.total_files == 0:
            return 0.0
        return len(self.failures) / self.total_files

    def to_dict(self) -> dict:
        return {
            "total_files": self.total_files,
            "success": self.success,
            "n_failures": len(self.failures),
            "failure_rate": round(self.failure_rate(), 4),
            "token_mismatches": self.token_mismatches,
            "tokens_dropped": self.tokens_dropped,
            "tokens_truncated": self.tokens_truncated,
            "failure_counts": self.failure_counts(),
            "failures": [
                {
                    "json_file":      f.json_file,
                    "transcript_id":  f.transcript_id,
                    "group_id":       f.group_id,
                    "model_name":     f.model_name,
                    "mode":           f.mode,
                    "failure_type":   f.failure_type,
                    "detail":         f.detail,
                }
                for f in self.failures
            ],
        }


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

# Matches the outermost [...] block, tolerating surrounding text/markdown fences
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def extract_json_array(text: str) -> Tuple[List[dict], str]:
    """
    Attempt to extract and parse the first JSON array from *text*.

    Returns:
        (records, failure_type)

        On success: (list_of_dicts, "")
        On failure: ([], FAILURE_*)

    Strategy:
        1. Quick bracket scan (original approach) — handles clean outputs.
        2. Regex fallback — handles outputs where the array is embedded in
           markdown fences or prose.
        3. Attempt to repair common truncation patterns (trailing comma before
           EOF) by appending "]}".
    """
    text = text.strip()

    if not text:
        return [], FAILURE_EMPTY

    # -- Strategy 1: bracket scan --
    first = text.find("[")
    last  = text.rfind("]")
    if first != -1 and last != -1 and last > first:
        candidate = text[first : last + 1]
        try:
            return json.loads(candidate), ""
        except json.JSONDecodeError:
            pass  # fall through to strategy 2

    # -- Strategy 2: regex extraction --
    match = _ARRAY_RE.search(text)
    if match:
        try:
            return json.loads(match.group()), ""
        except json.JSONDecodeError:
            pass

    # -- Strategy 3: truncation repair --
    # Models sometimes cut off mid-array; try appending closing tokens.
    for suffix in ("]", "}]", "\n}]"):
        candidate = text[text.find("["):] + suffix if "[" in text else ""
        if candidate:
            try:
                return json.loads(candidate), ""
            except json.JSONDecodeError:
                continue

    # Give up — decide between no_array and decode_error
    if "[" not in text:
        return [], FAILURE_NO_ARRAY
    return [], FAILURE_JSON_DECODE


def validate_records(records: List[dict]) -> Tuple[List[dict], str]:
    """
    Validate that every record contains the required fields.

    Returns (valid_records, failure_type).  failure_type is "" if all records
    are valid, else FAILURE_MISSING_FIELDS.
    """
    clean = []
    for rec in records:
        if not REQUIRED_FIELDS.issubset(rec.keys()):
            missing = REQUIRED_FIELDS - rec.keys()
            return [], f"{FAILURE_MISSING_FIELDS}: {missing}"
        clean.append(rec)
    return clean, ""


# ---------------------------------------------------------------------------
# Token-count mismatch handling
# ---------------------------------------------------------------------------

def handle_mismatch(
    preds: List[dict],
    gt_tokens: pd.DataFrame,
    strategy: str,
    report: ParseQualityReport,
    group_id: str,
    logger,
) -> Optional[List[dict]]:
    """
    Resolve a token-count mismatch between *preds* and *gt_tokens*.

    Returns the (possibly truncated) pred list, or None if the group should
    be dropped.
    """
    n_pred = len(preds)
    n_gt   = len(gt_tokens)
    delta  = abs(n_pred - n_gt)

    report.token_mismatches += 1
    logger.warning(
        "Token count mismatch for group %s: pred=%d gt=%d (delta=%d) — strategy=%s",
        group_id, n_pred, n_gt, delta, strategy,
    )

    if strategy == "drop":
        report.tokens_dropped += n_pred
        return None

    if strategy == "truncate":
        n = min(n_pred, n_gt)
        truncated = preds[:n]
        report.tokens_truncated += delta
        logger.debug("Truncated group %s to %d tokens.", group_id, n)
        return truncated

    # Unknown strategy — default to drop
    logger.error("Unknown mismatch strategy '%s' — defaulting to drop.", strategy)
    report.tokens_dropped += n_pred
    return None


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_directory(
    raw_dir: Path,
    labeled: pd.DataFrame,
    mismatch_strategy: str,
    report: ParseQualityReport,
    logger,
) -> List[dict]:
    """
    Walk *raw_dir* recursively, parse every *.json wrapper file, and return
    a list of merged row dicts ready for DataFrame construction.
    """
    json_files = sorted(raw_dir.rglob("*.json"))

    # Exclude any metadata/sidecar files written by earlier pipeline steps
    json_files = [
        f for f in json_files
        if not any(
            excl in f.name
            for excl in ("few_shot_examples_metadata", "run_metadata", "parse_quality")
        )
    ]

    report.total_files = len(json_files)
    logger.info("Found %d raw output files in %s", report.total_files, raw_dir)

    rows: List[dict] = []

    for json_file in tqdm(json_files, desc="Parsing HF outputs"):
        # ---------------------------------------------------------------- #
        # Load wrapper                                                      #
        # ---------------------------------------------------------------- #
        try:
            wrapper = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as exc:
            report.failures.append(ParseFailure(
                json_file=str(json_file),
                transcript_id=None, group_id=None,
                model_name=None, mode=None,
                failure_type=FAILURE_JSON_DECODE,
                detail=f"Wrapper load failed: {exc}",
            ))
            logger.warning("Could not load wrapper %s: %s", json_file.name, exc)
            continue

        tid        = wrapper.get("transcript_id")
        uid        = wrapper.get("utterance_id")
        group_id   = wrapper.get("group_id", tid)
        model_name = wrapper.get("model_name", "hf_model")
        model_key  = wrapper.get("model_key", "unknown")
        mode       = wrapper.get("mode", "unknown")
        seed       = wrapper.get("seed", None)
        resp_text  = wrapper.get("response_text", "")

        # ---------------------------------------------------------------- #
        # Extract and validate JSON array from model response               #
        # ---------------------------------------------------------------- #
        preds, failure_type = extract_json_array(resp_text)

        if failure_type:
            report.failures.append(ParseFailure(
                json_file=str(json_file),
                transcript_id=tid, group_id=group_id,
                model_name=model_name, mode=mode,
                failure_type=failure_type,
                detail=resp_text[:200],  # first 200 chars for diagnostics
            ))
            logger.warning(
                "Parse failure [%s] — file=%s group=%s",
                failure_type, json_file.name, group_id,
            )
            continue

        preds, val_failure = validate_records(preds)
        if val_failure:
            report.failures.append(ParseFailure(
                json_file=str(json_file),
                transcript_id=tid, group_id=group_id,
                model_name=model_name, mode=mode,
                failure_type=FAILURE_MISSING_FIELDS,
                detail=val_failure,
            ))
            logger.warning(
                "Validation failure [%s] — file=%s group=%s",
                val_failure, json_file.name, group_id,
            )
            continue

        # ---------------------------------------------------------------- #
        # Token-count mismatch check                                       #
        # ---------------------------------------------------------------- #
        if uid is not None:
            gt_group = labeled[
                (labeled["transcript_id"] == tid) &
                (labeled["utterance_id"]   == uid)
            ]
        else:
            gt_group = labeled[labeled["transcript_id"] == tid]

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
                    json_file=str(json_file),
                    transcript_id=tid, group_id=group_id,
                    model_name=model_name, mode=mode,
                    failure_type=FAILURE_TOKEN_MISMATCH,
                    detail=(
                        f"pred={len(preds) if preds else '?'} "
                        f"gt={len(gt_group)} strategy=drop"
                    ),
                ))
                continue

        # ---------------------------------------------------------------- #
        # Build rows                                                        #
        # ---------------------------------------------------------------- #
        report.success += 1
        for p in preds:
            rows.append(
                {
                    "transcript_id":  tid,
                    "utterance_id":   uid,
                    "group_id":       group_id,
                    "model_name":     model_name,
                    "model_key":      model_key,
                    "mode":           mode,
                    "seed":           seed,
                    "pred_index":     int(p["index"]),
                    "pred_token":     p["token"],
                    "pred_word_label":int(p["word_label"]),
                    "pred_ciu_label": int(p["ciu_label"]),
                }
            )

    return rows


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@app.command()
def main(
    labeled_path: Path = typer.Option(
        Path("data/labeled/ciu_tokens_normalized.parquet"),
        help="Ground-truth labels.",
    ),
    raw_dir: Path = typer.Option(
        Path("results/raw/hf_local"),
        help="Root directory containing HF JSON wrappers.",
    ),
    out_path: Path = typer.Option(
        Path("results/parsed/llm_predictions_hf.parquet"),
        help="Output parquet with predictions merged with ground-truth labels.",
    ),
    mismatch_strategy: str = typer.Option(
        "drop",
        help="How to handle token-count mismatches: 'drop' or 'truncate'.",
    ),
    log_dir: Path = typer.Option(
        Path("log"),
        help="Directory for log files.",
    ),
) -> None:
    logger = setup_logger("parse_llm_outputs_hf", log_dir=log_dir)
    logger.info("Loading ground-truth labels from %s", labeled_path)

    labeled = pd.read_parquet(labeled_path)
    labeled = labeled.sort_values(["transcript_id", "token_index"]).reset_index(drop=True)
    logger.info("Loaded %d ground-truth token rows.", len(labeled))

    report = ParseQualityReport()

    rows = parse_directory(
        raw_dir=raw_dir,
        labeled=labeled,
        mismatch_strategy=mismatch_strategy,
        report=report,
        logger=logger,
    )

    # ------------------------------------------------------------------ #
    # Merge predictions with ground truth                                  #
    # ------------------------------------------------------------------ #
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
    quality_dict = report.to_dict()

    quality_csv_path  = out_path.parent / (out_path.stem + "_parse_quality.csv")
    quality_json_path = out_path.parent / (out_path.stem + "_parse_quality.json")

    pd.DataFrame(report.failures).to_csv(quality_csv_path, index=False) \
        if report.failures else \
        pd.DataFrame(columns=["json_file","transcript_id","group_id",
                               "model_name","mode","failure_type","detail"]
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
            "Failure rate %.1f%% exceeds 10%% threshold — review parse_quality "
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
        mismatch_strategy=mismatch_strategy,
        total_files=report.total_files,
        parse_success=report.success,
        failure_rate=report.failure_rate(),
    )

    logger.info("parse_llm_outputs_hf complete.")


if __name__ == "__main__":
    app()

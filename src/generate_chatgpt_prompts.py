# src/generate_chatgpt_prompts.py

from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.2.3"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

"""
Generate one prompt text file per utterance (or transcript) for manual use
in the ChatGPT web UI.

For modes starting with 'few_shot', a few-shot block is auto-generated from
the prompt-support set using the specified --few-shot-strategy, mirroring the
behaviour of run_llm_inference.py so both pipelines receive equivalent prompts.

Outputs
-------
  results/prompts/chatgpt/<mode>/<group_id>.txt
  results/prompts/chatgpt/<mode>/few_shot_examples_metadata.json  (few-shot only)
  results/prompts/chatgpt/<mode>/prompt_metadata.json
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import typer
from jinja2 import Template
from tqdm import tqdm

from utils import (
    Config,
    set_global_seed,
    build_few_shot_block,
    save_json,
    setup_logger,
    save_run_metadata,
)

app = typer.Typer(add_completion=False)


def load_prompts_yaml(path: Path) -> Dict[str, str]:
    import yaml
    with path.open("r") as f:
        data = yaml.safe_load(f)
    system  = data["system"]
    prompts = data["prompts"]
    return {"system": system, **prompts}


def build_token_block(tokens: List[str], start_index: int = 0) -> str:
    """Render tokens as a two-column INDEX | TOKEN table, matching the HF prompt format."""
    return "\n".join(
        f"{start_index + i:<6}| {tok}"
        for i, tok in enumerate(tokens)
    )


def choose_grouping_cols(df: pd.DataFrame, logger) -> Tuple[List[str], bool]:
    if "utterance_id" in df.columns:
        logger.info("Grouping by (transcript_id, utterance_id).")
        return ["transcript_id", "utterance_id"], True
    else:
        logger.info("No 'utterance_id' column — grouping by transcript_id only.")
        return ["transcript_id"], False


@app.command()
def main(
    config_path: Path = typer.Option(
        Path("config.yaml"), help="Config file."
    ),
    mode: str = typer.Option(
        "z_shot_local",
        help="Prompting mode: z_shot_local | few_shot_local | few_shot_global",
    ),
    out_dir: Path = typer.Option(
        Path("results/prompts/chatgpt"),
        help="Root directory for prompt text files.",
    ),
    n_few_shot: int = typer.Option(
        3,
        help="Number of few-shot examples when using few_shot_* modes.",
    ),
    few_shot_strategy: str = typer.Option(
        "random",
        help=(
            "Few-shot example selection strategy: "
            "'random' (default) or 'severity_stratified'."
        ),
    ),
    seed: int = typer.Option(2025, help="Random seed."),
    log_dir: Path = typer.Option(Path("log"), help="Directory for log files."),
) -> None:
    """
    Write one .txt prompt file per group to results/prompts/chatgpt/<mode>/.
    """
    logger = setup_logger(
        f"generate_chatgpt_prompts__{mode}__seed{seed}",
        log_dir=log_dir,
    )
    logger.info(
        "Generating ChatGPT prompts — mode=%s  seed=%d  few_shot_strategy=%s",
        mode, seed, few_shot_strategy,
    )

    set_global_seed(seed)
    cfg = Config.load(config_path)

    prompts_dict  = load_prompts_yaml(Path("prompts/ciu_prompts.yaml"))
    system_prompt = prompts_dict["system"]
    user_template = Template(prompts_dict[mode])

    labeled_path    = Path(cfg["data"]["labeled_normalized"])
    eval_ids_path   = Path(cfg["data"]["eval_ids"])
    prompt_ids_path = Path(cfg["data"]["prompt_ids"])

    df_all   = pd.read_parquet(labeled_path)
    eval_ids = set(eval_ids_path.read_text().splitlines())

    df_eval = df_all[df_all["transcript_id"].isin(eval_ids)].copy()
    df_eval = df_eval.sort_values(["transcript_id", "token_index"]).reset_index(drop=True)
    logger.info(
        "Eval set: %d tokens across %d transcripts.",
        len(df_eval), df_eval["transcript_id"].nunique(),
    )

    group_cols, has_utter = choose_grouping_cols(df_eval, logger)

    out_mode_dir = out_dir / mode
    out_mode_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Few-shot block                                                       #
    # ------------------------------------------------------------------ #
    few_shot_text = ""
    if mode.startswith("few_shot") and prompt_ids_path.exists():
        prompt_ids = prompt_ids_path.read_text().splitlines()
        few_shot_text, few_shot_meta = build_few_shot_block(
            df_all,
            prompt_ids=prompt_ids,
            n_examples=n_few_shot,
            seed=seed,
            group_by_utterance=True,
            strategy=few_shot_strategy,
        )
        logger.info(
            "Built few-shot block: %d examples  strategy=%s",
            len(few_shot_meta), few_shot_strategy,
        )
        if few_shot_meta:
            save_json(few_shot_meta, out_mode_dir / "few_shot_examples_metadata.json")
            logger.debug("Few-shot metadata saved.")

    # ------------------------------------------------------------------ #
    # Prompt generation loop                                               #
    # ------------------------------------------------------------------ #
    n_written = 0
    n_groups  = df_eval.groupby(group_cols).ngroups

    for group_vals, g in tqdm(
        df_eval.groupby(group_cols),
        desc=f"Generating prompts | {mode}",
        total=n_groups,
    ):
        if isinstance(group_vals, tuple):
            transcript_id = group_vals[0]
            utterance_id  = group_vals[1] if len(group_vals) > 1 else None
        else:
            transcript_id = group_vals
            utterance_id  = None

        tokens      = g["token_text"].tolist()
        token_start = int(g["token_index"].min()) if "token_index" in g.columns else 0
        token_end   = int(g["token_index"].max()) if "token_index" in g.columns else len(tokens) - 1
        token_block = build_token_block(tokens, start_index=token_start)
        group_id    = (
            f"{transcript_id}__utt-{utterance_id}"
            if has_utter and utterance_id is not None
            else transcript_id
        )

        rendered_user = user_template.render(
            utterance_id=group_id,
            transcript_id=transcript_id,
            token_block=token_block,
            token_count=len(tokens),
            token_start=token_start,
            token_end=token_end,
            few_shot_examples=few_shot_text,
        )

        prompt_text = (
            "SYSTEM MESSAGE:\n"
            f"{system_prompt}\n\n"
            "USER MESSAGE:\n"
            f"{rendered_user}\n"
        )

        (out_mode_dir / f"{group_id}.txt").write_text(prompt_text, encoding="utf-8")
        n_written += 1

    logger.info("Wrote %d prompt files to %s", n_written, out_mode_dir)

    # ------------------------------------------------------------------ #
    # Run metadata sidecar                                                 #
    # ------------------------------------------------------------------ #
    save_run_metadata(
        out_mode_dir / "prompt_metadata.json",
        mode=mode,
        seed=seed,
        n_few_shot=n_few_shot,
        few_shot_strategy=few_shot_strategy,
        n_prompts_written=n_written,
        out_dir=str(out_mode_dir),
    )

    logger.info("generate_chatgpt_prompts complete.")


if __name__ == "__main__":
    app()

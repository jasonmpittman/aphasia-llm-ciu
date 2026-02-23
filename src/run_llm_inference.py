# src/run_llm_inference.py

from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.2.1"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

"""
Run a local Hugging Face LLM (optionally with LoRA adapters) on the evaluation
split and save raw per-group outputs as JSON.

Changes from v0.1.0
-------------------
- All print() calls replaced with structured logger (setup_logger).
- seed written into every output JSON wrapper so downstream parsers and
  compute_metrics can group by seed without relying on directory paths.
- --few-shot-strategy flag added (random | severity_stratified) to support
  Ablation B in run_all.sh.
- save_run_metadata() sidecar written to the output directory on completion.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import typer
from jinja2 import Template
from tqdm import tqdm

from utils import (
    Config,
    save_json,
    save_run_metadata,
    set_global_seed,
    get_model_config,
    build_few_shot_block,
    setup_logger,
)

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompts_yaml(path: Path) -> Dict[str, str]:
    import yaml
    with path.open("r") as f:
        data = yaml.safe_load(f)
    system  = data["system"]
    prompts = data["prompts"]
    return {"system": system, **prompts}


def build_token_block(tokens: List[str]) -> str:
    return "\n".join(f"{i}: {tok}" for i, tok in enumerate(tokens))


def load_hf_model_and_tokenizer(
    model_name: str,
    max_new_tokens: int,
    use_lora: bool,
    adapter_dir: Optional[Path],
    logger,
):
    """
    Load a local HF CausalLM + tokenizer, optionally with LoRA adapters.
    Targets Apple Silicon (MPS) or CPU; CUDA will be used automatically if
    available via the pipeline default device selection.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info("Device selected: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    if use_lora:
        if adapter_dir is None:
            raise ValueError("use_lora=True but no adapter_dir provided.")
        from peft import PeftModel
        logger.info("Loading LoRA adapters from: %s", adapter_dir)
        model = PeftModel.from_pretrained(model, str(adapter_dir))

    model = model.to(device)

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=0.001,
        top_p=1.0,
    )

    return text_gen, tokenizer


def choose_grouping_cols(df: pd.DataFrame, logger) -> Tuple[List[str], bool]:
    if "utterance_id" in df.columns:
        logger.info("Grouping by (transcript_id, utterance_id).")
        return ["transcript_id", "utterance_id"], True
    else:
        logger.info("No 'utterance_id' column — grouping by transcript_id only.")
        return ["transcript_id"], False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@app.command()
def main(
    config_path: Path = typer.Option(
        Path("config.yaml"), help="Config file."
    ),
    model_key: str = typer.Option(
        "phi3-mini",
        help="Key in config.yaml under 'models' to use for this run.",
    ),
    mode: str = typer.Option(
        "z_shot_local",
        help="Prompting mode: z_shot_local | few_shot_local | few_shot_global",
    ),
    out_root: Path = typer.Option(
        Path("results/raw/hf_local"),
        help="Root directory for raw HF model responses.",
    ),
    use_lora: bool = typer.Option(
        False,
        help="If true, load LoRA adapters for this model (PEFT).",
    ),
    adapter_dir: Optional[Path] = typer.Option(
        None,
        help=(
            "Directory containing LoRA adapters. "
            "Defaults to models/llm/<model_key>-ciu-lora if --use-lora is set."
        ),
    ),
    n_few_shot: int = typer.Option(
        3,
        help="Number of few-shot examples when using few_shot_* modes.",
    ),
    few_shot_strategy: str = typer.Option(
        "random",
        help=(
            "Few-shot example selection strategy: "
            "'random' (default) or 'severity_stratified'. "
            "Used by Ablation B."
        ),
    ),
    seed: int = typer.Option(2025, help="Random seed."),
    log_dir: Path = typer.Option(
        Path("log"), help="Directory for log files."
    ),
) -> None:
    """
    Run a local HF LLM on the evaluation split and save raw outputs.

    For modes starting with 'few_shot', a few-shot block is auto-generated
    from the prompt-support set using the specified --few-shot-strategy.
    Each output JSON wrapper includes the seed so downstream scripts can
    group results by seed without path parsing.
    """
    logger = setup_logger(
        f"run_llm_inference__{model_key}__{mode}__seed{seed}",
        log_dir=log_dir,
    )
    logger.info(
        "Starting inference — model_key=%s  mode=%s  seed=%d  use_lora=%s  "
        "few_shot_strategy=%s",
        model_key, mode, seed, use_lora, few_shot_strategy,
    )

    set_global_seed(seed)
    cfg = Config.load(config_path)

    model_cfg     = get_model_config(cfg, model_key)
    model_name    = model_cfg["model_name"]
    max_new_tokens = int(model_cfg.get("max_new_tokens", 2048))

    if use_lora and adapter_dir is None:
        adapter_dir = Path("models/llm") / f"{model_key}-ciu-lora"
        logger.info("--use-lora set; defaulting adapter_dir to %s", adapter_dir)

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

    out_dir = out_root / model_key / mode / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

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
            save_json(few_shot_meta, out_dir / "few_shot_examples_metadata.json")
            logger.debug("Few-shot metadata saved.")

    # ------------------------------------------------------------------ #
    # Load model                                                           #
    # ------------------------------------------------------------------ #
    logger.info("Loading model %s ...", model_name)
    text_gen, _ = load_hf_model_and_tokenizer(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        use_lora=use_lora,
        adapter_dir=adapter_dir,
        logger=logger,
    )
    logger.info("Model loaded.")

    # ------------------------------------------------------------------ #
    # Inference loop                                                       #
    # ------------------------------------------------------------------ #
    n_groups  = df_eval.groupby(group_cols).ngroups
    n_success = 0
    n_error   = 0

    for group_vals, g in tqdm(
        df_eval.groupby(group_cols),
        desc=f"{model_key} | {mode} | seed{seed}",
        total=n_groups,
    ):
        if isinstance(group_vals, tuple):
            transcript_id = group_vals[0]
            utterance_id  = group_vals[1] if len(group_vals) > 1 else None
        else:
            transcript_id = group_vals
            utterance_id  = None

        tokens      = g["token_text"].tolist()
        token_block = build_token_block(tokens)
        group_id    = (
            f"{transcript_id}__utt-{utterance_id}"
            if has_utter and utterance_id is not None
            else transcript_id
        )

        rendered_user = user_template.render(
            utterance_id=group_id,
            transcript_id=transcript_id,
            token_block=token_block,
            few_shot_examples=few_shot_text,
        )

        prompt = (
            "SYSTEM MESSAGE:\n"
            f"{system_prompt}\n\n"
            "USER MESSAGE:\n"
            f"{rendered_user}\n"
        )

        try:
            gen = text_gen(prompt)[0]["generated_text"]
            n_success += 1
        except Exception as exc:
            logger.error("Inference failed for group %s: %s", group_id, exc)
            n_error += 1
            continue

        save_json(
            {
                "transcript_id": transcript_id,
                "utterance_id":  utterance_id,
                "group_id":      group_id,
                "mode":          mode,
                "model_name":    model_name,
                "model_key":     model_key,
                "seed":          seed,
                "use_lora":      use_lora,
                "adapter_dir":   str(adapter_dir) if adapter_dir is not None else None,
                "few_shot_strategy": few_shot_strategy,
                "n_few_shot":    n_few_shot,
                "response_text": gen,
            },
            out_dir / f"{group_id}.json",
        )

    logger.info(
        "Inference complete — success=%d  errors=%d  total=%d",
        n_success, n_error, n_groups,
    )

    if n_error > 0:
        logger.warning(
            "%d groups failed inference and were skipped. "
            "Review log for details.",
            n_error,
        )

    # ------------------------------------------------------------------ #
    # Run metadata sidecar                                                 #
    # ------------------------------------------------------------------ #
    save_run_metadata(
        out_dir / "run_metadata.json",
        model_key=model_key,
        model_name=model_name,
        mode=mode,
        seed=seed,
        use_lora=use_lora,
        adapter_dir=str(adapter_dir) if adapter_dir else None,
        n_few_shot=n_few_shot,
        few_shot_strategy=few_shot_strategy,
        n_groups=n_groups,
        n_success=n_success,
        n_error=n_error,
    )

    logger.info("Outputs written to %s", out_dir)


if __name__ == "__main__":
    app()

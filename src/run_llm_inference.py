# src/run_llm_inference.py

from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.2.3"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

"""
Run a local Hugging Face LLM (optionally with LoRA adapters) on the evaluation
split and save raw per-chunk outputs as JSON.

Chunking strategy
-----------------
Transcripts are split into non-overlapping, contiguous chunks of at most
--chunk-size tokens before being passed to the model.  This keeps every input
well within the context window of resource-constrained 7-8B models (max_seq
768-1024).  Token indices in each chunk reflect the *original* token_index
values from the dataset, so the downstream parser can reassemble chunks into
full transcripts and align predictions to ground truth without offset
arithmetic.

One JSON wrapper file is written per chunk (not per transcript), named
<group_id>__chunk<NNN>.json.  The wrapper carries enough metadata for the
parser to reconstruct chunk order and provenance.

Changes from v0.2.1
-------------------
- Chunking via chunk_transcript() / build_token_block_from_chunk() from utils.
- --chunk-size CLI flag (default 50).
- seed written into every output JSON wrapper (carried forward from v0.2.1).
- --few-shot-strategy flag (carried forward from v0.2.1).
- save_run_metadata() sidecar (carried forward from v0.2.1).
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
    chunk_transcript,
    build_token_block_from_chunk,
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
    return {"system": data["system"], **data["prompts"]}


def load_hf_model_and_tokenizer(
    model_name: str,
    max_new_tokens: int,
    use_lora: bool,
    adapter_dir: Optional[Path],
    logger,
):
    """
    Load a local HF CausalLM + tokenizer, optionally with LoRA adapters.
    Device priority: CUDA > MPS > CPU.
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
        do_sample=False,       # greedy decoding — deterministic, numerically stable,
                               # and correct for reproducible scientific inference.
                               # temperature/top_p are irrelevant when do_sample=False.
        return_full_text=False,
    )
    return text_gen, tokenizer


def choose_grouping_cols(df: pd.DataFrame, logger) -> Tuple[List[str], bool]:
    if "utterance_id" in df.columns:
        logger.info("Grouping by (transcript_id, utterance_id).")
        return ["transcript_id", "utterance_id"], True
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
        help="Key in config.yaml under 'models'.",
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
        help="Load LoRA adapters for this model (PEFT).",
    ),
    adapter_dir: Optional[Path] = typer.Option(
        None,
        help=(
            "LoRA adapter directory. "
            "Defaults to models/llm/<model_key>-ciu-lora if --use-lora is set."
        ),
    ),
    n_few_shot: int = typer.Option(
        3,
        help="Number of few-shot examples for few_shot_* modes.",
    ),
    few_shot_strategy: str = typer.Option(
        "random",
        help="Few-shot selection strategy: 'random' or 'severity_stratified'.",
    ),
    chunk_size: int = typer.Option(
        50,
        help=(
            "Maximum tokens per chunk. Transcripts longer than this are split "
            "into contiguous non-overlapping chunks before inference. "
            "Set to 0 to disable chunking (not recommended for 7-8B models)."
        ),
    ),
    seed: int = typer.Option(2025, help="Random seed."),
    log_dir: Path = typer.Option(Path("log"), help="Directory for log files."),
) -> None:
    """
    Run a local HF LLM on the evaluation split and save raw chunk outputs.

    Each transcript is chunked into segments of at most --chunk-size tokens.
    One JSON wrapper is written per chunk, carrying enough metadata for the
    downstream parser to reassemble predictions into full-transcript sequences.
    """
    logger = setup_logger(
        f"run_llm_inference__{model_key}__{mode}__seed{seed}",
        log_dir=log_dir,
    )
    logger.info(
        "Starting inference — model_key=%s  mode=%s  seed=%d  "
        "use_lora=%s  chunk_size=%d  few_shot_strategy=%s",
        model_key, mode, seed, use_lora, chunk_size, few_shot_strategy,
    )

    set_global_seed(seed)
    cfg = Config.load(config_path)

    model_cfg      = get_model_config(cfg, model_key)
    model_name     = model_cfg["model_name"]
    max_new_tokens = int(model_cfg.get("max_new_tokens", 1500))

    # Chunk settings — CLI flags take precedence; config.yaml inference block
    # provides defaults so values are consistent across all scripts.
    inference_cfg   = cfg.data.get("inference", {})
    eff_chunk_size  = chunk_size if chunk_size > 0 else int(inference_cfg.get("chunk_size", 32))
    eff_min_chunk   = int(inference_cfg.get("min_chunk_size", 10))

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
    # Inference loop — iterate groups, chunk each, run per chunk          #
    # ------------------------------------------------------------------ #
    n_groups       = df_eval.groupby(group_cols).ngroups
    n_chunks_total = 0
    n_success      = 0
    n_error        = 0

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

        group_id = (
            f"{transcript_id}__utt-{utterance_id}"
            if has_utter and utterance_id is not None
            else transcript_id
        )

        # ---- Chunk the group ----------------------------------------- #
        effective_chunk_size = eff_chunk_size if eff_chunk_size > 0 else len(g)
        chunks = chunk_transcript(
            group_df=g,
            chunk_size=effective_chunk_size,
            transcript_id=transcript_id,
            utterance_id=utterance_id,
            min_chunk_size=eff_min_chunk,
        )

        if len(chunks) > 1:
            logger.debug(
                "Chunked %s (%d tokens) into %d chunks of ≤%d.",
                group_id, len(g), len(chunks), effective_chunk_size,
            )

        n_chunks_total += len(chunks)

        # ---- Run inference on each chunk ----------------------------- #
        for chunk in chunks:
            token_block = build_token_block_from_chunk(chunk)

            rendered_user = user_template.render(
                utterance_id=chunk["chunk_id"],
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
                logger.error(
                    "Inference failed for chunk %s: %s",
                    chunk["chunk_id"], exc,
                )
                n_error += 1
                continue

            save_json(
                {
                    # Chunk provenance — used by parser to reassemble
                    "transcript_id":  transcript_id,
                    "utterance_id":   utterance_id,
                    "group_id":       group_id,
                    "chunk_id":       chunk["chunk_id"],
                    "chunk_index":    chunk["chunk_index"],
                    "n_chunks":       chunk["n_chunks"],
                    "token_start":    chunk["token_start"],
                    "token_end":      chunk["token_end"],
                    "token_indices":  chunk["token_indices"],
                    # Run metadata
                    "mode":           mode,
                    "model_name":     model_name,
                    "model_key":      model_key,
                    "seed":           seed,
                    "use_lora":       use_lora,
                    "adapter_dir":    str(adapter_dir) if adapter_dir else None,
                    "few_shot_strategy": few_shot_strategy,
                    "n_few_shot":     n_few_shot,
                    "chunk_size":     chunk_size,
                    # Model output
                    "response_text":  gen,
                },
                out_dir / f"{chunk['chunk_id']}.json",
            )

    logger.info(
        "Inference complete — groups=%d  chunks=%d  success=%d  errors=%d",
        n_groups, n_chunks_total, n_success, n_error,
    )
    if n_error > 0:
        logger.warning(
            "%d chunks failed inference and were skipped. "
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
        chunk_size=chunk_size,
        n_groups=n_groups,
        n_chunks_total=n_chunks_total,
        n_success=n_success,
        n_error=n_error,
    )

    logger.info("Outputs written to %s", out_dir)


if __name__ == "__main__":
    app()

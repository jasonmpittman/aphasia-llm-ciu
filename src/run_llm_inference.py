# src/run_llm_inference.py
from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2025"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import typer
from jinja2 import Template
from tqdm import tqdm

from utils import (
    Config,
    save_json,
    set_global_seed,
    get_model_config,
    build_few_shot_block,
)

app = typer.Typer(add_completion=False)


def load_prompts_yaml(path: Path) -> Dict[str, str]:
    import yaml

    with path.open("r") as f:
        data = yaml.safe_load(f)
    system = data["system"]
    prompts = data["prompts"]
    return {"system": system, **prompts}


def build_token_block(tokens: List[str]) -> str:
    return "\n".join(f"{i}: {tok}" for i, tok in enumerate(tokens))


def load_hf_model_and_tokenizer(
    model_name: str,
    max_new_tokens: int,
    use_lora: bool,
    adapter_dir: Optional[Path] = None,
):
    """
    Load a local HF CausalLM model + tokenizer, optionally with LoRA adapters.
    Designed to run on Apple Silicon (MPS) or CPU.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[run_llm_inference] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    if use_lora:
        if adapter_dir is None:
            raise ValueError("use_lora=True but no adapter_dir provided.")
        from peft import PeftModel

        print(f"[run_llm_inference] Loading LoRA adapters from: {adapter_dir}")
        model = PeftModel.from_pretrained(model, str(adapter_dir))

    if device == "mps":
        model = model.to("mps")
    else:
        model = model.to("cpu")

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=0.001,
        top_p=1.0,
    )

    return text_gen, tokenizer


def choose_grouping_cols(df: pd.DataFrame) -> Tuple[List[str], bool]:
    if "utterance_id" in df.columns:
        print("HF inference: grouping by (transcript_id, utterance_id).")
        return ["transcript_id", "utterance_id"], True
    else:
        print("HF inference: no 'utterance_id'; grouping by transcript_id only.")
        return ["transcript_id"], False


@app.command()
def main(
    config_path: Path = typer.Option(Path("config.yaml"), help="Config file."),
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
            "If not set and --use-lora is true, defaults to models/llm/<model_key>-ciu-lora."
        ),
    ),
    n_few_shot: int = typer.Option(
        3,
        help="Number of few-shot examples to include when using few_shot_* modes.",
    ),
    seed: int = typer.Option(2025, help="Random seed."),
) -> None:
    """
    Run a local Hugging Face LLM (optionally with LoRA) on the evaluation split and save raw outputs.

    For modes starting with 'few_shot', a few-shot block is auto-generated from the prompt-support set.
    """
    set_global_seed(seed)
    cfg = Config.load(config_path)

    model_cfg = get_model_config(cfg, model_key)
    model_name = model_cfg["model_name"]
    max_new_tokens = int(model_cfg.get("max_new_tokens", 2048))

    if use_lora and adapter_dir is None:
        adapter_dir = Path("models/llm") / f"{model_key}-ciu-lora"
        print(f"[run_llm_inference] --use-lora set; defaulting adapter_dir to {adapter_dir}")

    prompts_dict = load_prompts_yaml(Path("prompts/ciu_prompts.yaml"))
    system_prompt = prompts_dict["system"]
    user_template = Template(prompts_dict[mode])

    labeled_path = Path(cfg["data"]["labeled_normalized"])
    eval_ids_path = Path(cfg["data"]["eval_ids"])
    prompt_ids_path = Path(cfg["data"]["prompt_ids"])

    df_all = pd.read_parquet(labeled_path)

    eval_ids = set(eval_ids_path.read_text().splitlines())
    df_eval = df_all[df_all["transcript_id"].isin(eval_ids)].copy()
    df_eval = df_eval.sort_values(["transcript_id", "token_index"]).reset_index(drop=True)

    group_cols, has_utter = choose_grouping_cols(df_eval)

    # Output directory for this model+mode
    out_dir = out_root / model_key / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build few-shot examples (and log which ones) if needed
    few_shot_text = ""
    if mode.startswith("few_shot") and prompt_ids_path.exists():
        prompt_ids = prompt_ids_path.read_text().splitlines()
        few_shot_text, few_shot_meta = build_few_shot_block(
            df_all,
            prompt_ids=prompt_ids,
            n_examples=n_few_shot,
            seed=seed,
            group_by_utterance=True,
        )
        print(
            f"[run_llm_inference] Built few-shot block with up to {n_few_shot} examples "
            f"from prompt-support set."
        )
        if few_shot_meta:
            save_json(
                few_shot_meta,
                out_dir / "few_shot_examples_metadata.json",
            )
            print(
                f"[run_llm_inference] Logged few-shot metadata to "
                f"{out_dir / 'few_shot_examples_metadata.json'}"
            )

    print(
        f"[run_llm_inference] Running model_key='{model_key}' "
        f"({model_name}) | mode={mode} | use_lora={use_lora}"
    )
    text_gen, _ = load_hf_model_and_tokenizer(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        use_lora=use_lora,
        adapter_dir=adapter_dir,
    )

    for group_vals, g in tqdm(
        df_eval.groupby(group_cols), desc=f"Running HF LLM: {model_key} | {mode}"
    ):
        if isinstance(group_vals, tuple):
            transcript_id = group_vals[0]
            utterance_id = group_vals[1] if len(group_vals) > 1 else None
        else:
            transcript_id = group_vals
            utterance_id = None

        tokens = g["token_text"].tolist()
        token_block = build_token_block(tokens)

        group_id = (
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

        gen = text_gen(prompt)[0]["generated_text"]

        save_json(
            {
                "transcript_id": transcript_id,
                "utterance_id": utterance_id,
                "group_id": group_id,
                "mode": mode,
                "model_name": model_name,
                "model_key": model_key,
                "use_lora": use_lora,
                "adapter_dir": str(adapter_dir) if adapter_dir is not None else None,
                "response_text": gen,
            },
            out_dir / f"{group_id}.json",
        )

    print(f"[run_llm_inference] Completed. Outputs in {out_dir}")


if __name__ == "__main__":
    app()

# src/generate_chatgpt_prompts.py
from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2025"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import typer
from jinja2 import Template
from tqdm import tqdm

from utils import Config, set_global_seed, build_few_shot_block, save_json

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


def choose_grouping_cols(df: pd.DataFrame) -> Tuple[List[str], bool]:
    if "utterance_id" in df.columns:
        print("ChatGPT prompts: grouping by (transcript_id, utterance_id).")
        return ["transcript_id", "utterance_id"], True
    else:
        print("ChatGPT prompts: no 'utterance_id'; grouping by transcript_id only.")
        return ["transcript_id"], False


@app.command()
def main(
    config_path: Path = typer.Option(Path("config.yaml"), help="Config file."),
    mode: str = typer.Option(
        "z_shot_local",
        help="Prompting mode: z_shot_local | few_shot_local | few_shot_global",
    ),
    out_dir: Path = typer.Option(
        Path("results/prompts/chatgpt"),
        help="Where to save prompt text files.",
    ),
    n_few_shot: int = typer.Option(
        3,
        help="Number of few-shot examples to include when using few_shot_* modes.",
    ),
    seed: int = typer.Option(2025, help="Random seed."),
) -> None:
    """
    Generate one prompt text file per utterance (if available) or per transcript,
    for manual use in the ChatGPT web UI.

    For modes starting with 'few_shot', a few-shot block is auto-generated from the
    prompt-support set and logged with metadata.
    """
    set_global_seed(seed)
    cfg = Config.load(config_path)

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

    out_mode_dir = out_dir / mode
    out_mode_dir.mkdir(parents=True, exist_ok=True)

    # Few-shot block + metadata
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
            f"[generate_chatgpt_prompts] Built few-shot block with up to {n_few_shot} examples "
            f"from prompt-support set."
        )
        if few_shot_meta:
            save_json(
                few_shot_meta,
                out_mode_dir / "few_shot_examples_metadata.json",
            )
            print(
                "[generate_chatgpt_prompts] Logged few-shot metadata to "
                f"{out_mode_dir / 'few_shot_examples_metadata.json'}"
            )

    for group_vals, g in tqdm(df_eval.groupby(group_cols), desc="Generating ChatGPT prompts"):
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

        prompt_text = (
            "SYSTEM MESSAGE:\n"
            f"{system_prompt}\n\n"
            "USER MESSAGE:\n"
            f"{rendered_user}\n"
        )

        (out_mode_dir / f"{group_id}.txt").write_text(prompt_text)

    print(f"[generate_chatgpt_prompts] Prompts written to {out_mode_dir}")


if __name__ == "__main__":
    app()

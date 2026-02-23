# src/finetune_llm.py

from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.2.1"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

"""
LoRA / QLoRA fine-tuning of a causal LM on CIU token classification.

Changes from v0.1.0
-------------------
- All print() calls replaced with structured logger (setup_logger).
- save_run_metadata() sidecar written to adapter_dir on completion so every
  fine-tuned checkpoint is self-describing.
- CUDA device support added alongside MPS / CPU.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import typer
from datasets import Dataset
from jinja2 import Template
from tqdm import tqdm

from utils import Config, set_global_seed, get_model_config, setup_logger, save_run_metadata

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    model_name: str
    output_dir: Path
    use_qlora: bool
    max_seq_length: int
    num_train_epochs: int
    per_device_train_batch_size: int
    learning_rate: float
    seed: int


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


def build_gold_json(
    tokens: List[str],
    word_labels: List[int],
    ciu_labels: List[int],
) -> str:
    import json
    records = [
        {"index": i, "token": tok, "word_label": int(w), "ciu_label": int(c)}
        for i, (tok, w, c) in enumerate(zip(tokens, word_labels, ciu_labels))
    ]
    return json.dumps(records, indent=2)


def make_sft_examples(
    df: pd.DataFrame,
    prompt_ids: List[str],
    prompts_path: Path,
    mode: str,
    logger,
) -> List[Dict[str, str]]:
    """
    Build SFT-style training examples at utterance or transcript level.

    Text format:
        SYSTEM MESSAGE
        USER MESSAGE
        ASSISTANT MESSAGE (gold JSON labels)
    """
    prompts_dict  = load_prompts_yaml(prompts_path)
    system_prompt = prompts_dict["system"]
    user_template = Template(prompts_dict[mode])

    df = df[df["transcript_id"].isin(prompt_ids)].copy()
    df = df.sort_values(["transcript_id", "token_index"]).reset_index(drop=True)

    if "utterance_id" in df.columns:
        group_cols = ["transcript_id", "utterance_id"]
        logger.info("Grouping SFT examples by (transcript_id, utterance_id).")
    else:
        group_cols = ["transcript_id"]
        logger.info("Grouping SFT examples by transcript_id only.")

    examples: List[Dict[str, str]] = []

    for group_vals, g in tqdm(df.groupby(group_cols), desc="Building SFT examples"):
        if isinstance(group_vals, tuple):
            transcript_id = group_vals[0]
            utterance_id  = group_vals[1] if len(group_vals) > 1 else None
        else:
            transcript_id = group_vals
            utterance_id  = None

        tokens      = g["token_text"].tolist()
        word_labels = g["word_label"].tolist()
        ciu_labels  = g["ciu_label"].tolist()
        token_block = build_token_block(tokens)

        group_id = (
            f"{transcript_id}__utt-{utterance_id}"
            if utterance_id is not None
            else transcript_id
        )

        user_prompt = user_template.render(
            utterance_id=group_id,
            transcript_id=transcript_id,
            token_block=token_block,
            few_shot_examples="",
        )

        gold_json = build_gold_json(tokens, word_labels, ciu_labels)

        full_text = (
            "SYSTEM MESSAGE:\n"
            f"{system_prompt}\n\n"
            "USER MESSAGE:\n"
            f"{user_prompt}\n\n"
            "ASSISTANT MESSAGE:\n"
            f"{gold_json}\n"
        )

        examples.append(
            {
                "transcript_id": transcript_id,
                "utterance_id":  utterance_id,
                "text":          full_text,
            }
        )

    logger.info("Built %d SFT training examples.", len(examples))
    return examples


def get_train_config(
    cfg: Config,
    model_key: str,
    output_dir: Path,
    seed: int,
) -> TrainConfig:
    model_cfg = get_model_config(cfg, model_key)
    ft = model_cfg.get("finetune", {})
    return TrainConfig(
        model_name=model_cfg["model_name"],
        output_dir=output_dir,
        use_qlora=bool(ft.get("use_qlora", False)),
        max_seq_length=int(ft.get("max_seq_length", 1024)),
        num_train_epochs=int(ft.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(ft.get("batch_size", 1)),
        learning_rate=float(ft.get("learning_rate", 2e-4)),
        seed=seed,
    )


def load_model_and_tokenizer(train_cfg: TrainConfig, logger):
    """
    Load base model + tokenizer, optionally with QLoRA setup.

    Device priority: CUDA > MPS > CPU.
    QLoRA requires CUDA + bitsandbytes; keep use_qlora=False on Apple Silicon.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if train_cfg.use_qlora:
        logger.info("QLoRA path selected (requires CUDA + bitsandbytes).")
        from peft import LoraConfig, get_peft_model
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )

        tokenizer = AutoTokenizer.from_pretrained(train_cfg.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            train_cfg.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        logger.info("QLoRA model ready.")

    else:
        logger.info("Standard (non-quantised) LoRA-compatible load.")
        tokenizer = AutoTokenizer.from_pretrained(train_cfg.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(train_cfg.model_name)

    return model, tokenizer


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int, logger) -> Dataset:
    logger.info("Tokenising dataset (max_length=%d) ...", max_length)

    def _tok(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(
        _tok,
        batched=True,
        remove_columns=["text", "transcript_id", "utterance_id"],
    )
    logger.info("Tokenisation complete — %d examples.", len(tokenized))
    return tokenized


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
        help="Key in config.yaml under 'models' to fine-tune.",
    ),
    prompts_path: Path = typer.Option(
        Path("prompts/ciu_prompts.yaml"),
        help="YAML with CIU prompt templates.",
    ),
    labeled_path: Path = typer.Option(
        Path("data/labeled/ciu_tokens_normalized.parquet"),
        help="Normalized labeled tokens.",
    ),
    prompt_ids_path: Path = typer.Option(
        Path("data/splits/prompt_ids.txt"),
        help="Transcript IDs used to build fine-tuning set.",
    ),
    mode: str = typer.Option(
        "z_shot_local",
        help="Prompt template key to use for SFT examples.",
    ),
    seed: int = typer.Option(2025, help="Random seed."),
    log_dir: Path = typer.Option(
        Path("log"), help="Directory for log files."
    ),
) -> None:
    """
    Fine-tune a causal LM with (Q)LoRA on CIU labeling examples drawn from
    the prompt-support set.  Saves adapters under models/llm/<model_key>-ciu-lora/.
    """
    logger = setup_logger(
        f"finetune_llm__{model_key}__seed{seed}",
        log_dir=log_dir,
    )
    logger.info(
        "Starting fine-tune — model_key=%s  mode=%s  seed=%d",
        model_key, mode, seed,
    )

    set_global_seed(seed)
    cfg = Config.load(config_path)

    df          = pd.read_parquet(labeled_path)
    prompt_ids  = prompt_ids_path.read_text().splitlines()
    logger.info(
        "Loaded %d labeled rows; %d prompt-support transcripts.",
        len(df), len(prompt_ids),
    )

    examples = make_sft_examples(
        df=df,
        prompt_ids=prompt_ids,
        prompts_path=prompts_path,
        mode=mode,
        logger=logger,
    )
    dataset = Dataset.from_pandas(pd.DataFrame(examples))

    adapter_dir = Path("models/llm") / f"{model_key}-ciu-lora"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = get_train_config(cfg, model_key, adapter_dir, seed)
    logger.info(
        "TrainConfig — model=%s  use_qlora=%s  epochs=%d  lr=%s  batch=%d  "
        "max_seq_len=%d  output=%s",
        train_cfg.model_name,
        train_cfg.use_qlora,
        train_cfg.num_train_epochs,
        train_cfg.learning_rate,
        train_cfg.per_device_train_batch_size,
        train_cfg.max_seq_length,
        adapter_dir,
    )

    model, tokenizer = load_model_and_tokenizer(train_cfg, logger)
    tokenized        = tokenize_dataset(dataset, tokenizer, train_cfg.max_seq_length, logger)

    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(adapter_dir),
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        num_train_epochs=train_cfg.num_train_epochs,
        learning_rate=train_cfg.learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch",
        seed=train_cfg.seed,
        gradient_checkpointing=True,
        fp16=train_cfg.use_qlora,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Training started.")
    trainer.train()
    logger.info("Training complete.")

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Adapters saved to %s", adapter_dir)

    # ------------------------------------------------------------------ #
    # Run metadata sidecar                                                 #
    # ------------------------------------------------------------------ #
    save_run_metadata(
        adapter_dir / "finetune_metadata.json",
        model_key=model_key,
        model_name=train_cfg.model_name,
        mode=mode,
        seed=seed,
        use_qlora=train_cfg.use_qlora,
        num_train_epochs=train_cfg.num_train_epochs,
        learning_rate=train_cfg.learning_rate,
        max_seq_length=train_cfg.max_seq_length,
        n_training_examples=len(examples),
        adapter_dir=str(adapter_dir),
    )

    logger.info("finetune_llm complete.")


if __name__ == "__main__":
    app()

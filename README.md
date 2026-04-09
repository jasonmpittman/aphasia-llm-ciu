# aphasia-llm-ciu

Benchmarking local LLMs and baseline models for Correct Information Unit (CIU) token classification in aphasia speech samples.

The repository centers on a labeled token dataset from a prior CIU study and provides an end-to-end pipeline to:

- normalize the labeled data into a canonical parquet table,
- create transcript-level prompt-support and evaluation splits,
- run zero-shot and few-shot prompting with local Hugging Face models,
- generate equivalent prompt files for manual ChatGPT web UI use,
- fine-tune selected models with LoRA or QLoRA,
- parse model outputs back into token-level predictions, and
- compute evaluation metrics, confidence intervals, severity breakdowns, and McNemar comparisons.

## What The Project Contains

- `src/` contains the Typer CLI scripts for data prep, splitting, inference, parsing, fine-tuning, baselines, and metrics.
- `scripts/run_all.sh` runs the full experiment matrix across models, prompting modes, and seeds, with checkpointing.
- `config.yaml` stores dataset paths, chunking defaults, and model-specific inference and fine-tuning settings.
- `prompts/ciu_prompts.yaml` defines the system prompt plus `z_shot_local`, `few_shot_local`, and `few_shot_global` templates.
- `data/` contains the labeled CSV, normalized parquet, and split metadata.
- `results/` stores raw generations, parsed predictions, and metric summaries.
- `log/` is the active logging directory used by the Python scripts and `run_all.sh`.

## Current Model Setup

`config.yaml` currently defines these local Hugging Face model keys:

- `phi3-mini`
- `llama3-8b`
- `qwen2.5-7b`
- `mistral-7b`

Inference defaults are chunk-based to keep long transcripts within context limits:

- `chunk_size: 16`
- `min_chunk_size: 10`

## Environment Setup

This project is Python-based and the scripts are written as Typer CLIs.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- `torch` should be installed in the variant appropriate for your hardware. The requirements file notes using the recommended PyTorch install command from `pytorch.org` if needed.
- `bitsandbytes` is listed for QLoRA but is CUDA-oriented; the code comments explicitly note that QLoRA should stay off on Apple Silicon.
- If you plan to process raw transcript text with spaCy, also install `en_core_web_sm`.

## Data Inputs

The main expected input is:

- `data/labeled/ciu_tokens.csv`

The normalization step expects these columns:

- `transcript_id`
- `token_index`
- `token_text`
- `word_label`
- `ciu_label`
- `speaker_id`
- `severity`

The repo already includes derived artifacts from prior runs, including:

- `data/labeled/ciu_tokens_normalized.parquet`
- `data/splits/prompt_ids.txt`
- `data/splits/eval_ids.txt`

## Pipeline Overview

The typical workflow is:

1. Normalize the labeled token table.
2. Create a transcript-level split into prompt-support and eval sets.
3. Train the classic baseline.
4. Run local Hugging Face inference for one or more models, prompting modes, and seeds.
5. Parse raw model outputs into merged token-level prediction tables.
6. Compute per-run, cross-seed, per-severity, and McNemar metrics.
7. Optionally fine-tune models with LoRA and re-run inference.
8. Optionally generate prompt files for manual ChatGPT web UI experiments.

## Quickstart

Run the core preprocessing steps:

```bash
python src/data_prep.py
python src/split_dataset.py --prompt-n 5 --seed 2025
python src/train_baselines.py --seed 2025
```

Run one local inference condition:

```bash
python src/run_llm_inference.py \
  --model-key phi3-mini \
  --mode z_shot_local \
  --seed 2025
```

Parse and score that run:

```bash
python src/parse_llm_outputs_hf.py \
  --raw-dir results/raw/hf_local/phi3-mini/z_shot_local/seed2025

python src/compute_metrics.py \
  --merged-path results/parsed/llm_predictions_hf.parquet \
  --out-dir results/metrics/phi3-mini/z_shot_local/seed2025
```

If you want the parser output path to mirror the full experiment script exactly, pass an explicit `--out-path`, for example:

```bash
python src/parse_llm_outputs_hf.py \
  --raw-dir results/raw/hf_local/phi3-mini/z_shot_local/seed2025 \
  --out-path results/parsed/llm_predictions_phi3-mini_z_shot_local_seed2025.parquet
```

## Full Experiment Script

The main orchestrator is:

```bash
bash scripts/run_all.sh
```

Useful flags:

- `--dry-run` prints the full command matrix without executing it.
- `--skip-finetune` skips Step 5, the LoRA fine-tune and LoRA inference stages.
- `--skip-ablations` skips Step 6, the few-shot ablation experiments.

`run_all.sh` does the following:

- Step 1: normalize `data/labeled/ciu_tokens.csv`
- Step 2: build prompt/eval splits
- Step 3: train the `LinearSVC` baseline
- Step 4: run the main model zoo x prompting mode x seed matrix
- Step 5: optionally fine-tune each model and run LoRA inference
- Step 6: optionally run few-shot count and selection-strategy ablations
- Step 7: aggregate parsed predictions and run McNemar tests on the combined file

The script checkpoints completed cells in `log/completed_cells.txt`, which makes restarts safer after interruptions.

## Main CLI Scripts

- `src/data_prep.py`: validates schema and invariants, then writes `data/labeled/ciu_tokens_normalized.parquet`
- `src/split_dataset.py`: creates a transcript-level severity-stratified split into prompt-support and eval IDs
- `src/train_baselines.py`: trains a frozen `LinearSVC` baseline using TF-IDF token text features plus severity
- `src/run_llm_inference.py`: runs chunked local Hugging Face inference, optionally with LoRA adapters
- `src/parse_llm_outputs_hf.py`: parses raw chunk wrapper JSON files and merges predictions with ground truth
- `src/generate_chatgpt_prompts.py`: writes prompt `.txt` files for manual ChatGPT web UI labeling
- `src/parse_llm_outputs_chatgpt.py`: parses manual ChatGPT response files and merges predictions with ground truth
- `src/finetune_llm.py`: builds supervised fine-tuning examples from the prompt-support set and trains LoRA or QLoRA adapters
- `src/compute_metrics.py`: computes point estimates, bootstrapped confidence intervals, severity breakdowns, and McNemar tests

## Prompting Modes

The prompt templates currently defined in `prompts/ciu_prompts.yaml` are:

- `z_shot_local`: zero-shot labeling for a single short utterance
- `few_shot_local`: few-shot labeling for a single short utterance
- `few_shot_global`: few-shot labeling for a full transcript

For few-shot runs, example blocks are sampled from the prompt-support split using either:

- `random`
- `severity_stratified`

## Manual ChatGPT Workflow

To generate prompt files for use in the ChatGPT web UI:

```bash
python src/generate_chatgpt_prompts.py \
  --mode few_shot_local \
  --n-few-shot 3 \
  --few-shot-strategy random
```

This writes prompts under:

- `results/prompts/chatgpt/<mode>/`

After collecting ChatGPT responses as `.txt` files, parse them with:

```bash
python src/parse_llm_outputs_chatgpt.py \
  --raw-dir results/raw/chatgpt/z_shot_local \
  --model-name chatgpt-webui \
  --mode z_shot_local
```

## Fine-Tuning

To fine-tune one configured model on prompt-support examples:

```bash
python src/finetune_llm.py \
  --model-key phi3-mini \
  --mode z_shot_local \
  --seed 2025
```

By default, adapters are saved under:

- `models/llm/<model_key>-ciu-lora/`

You can then run inference with those adapters using `--use-lora`.

## Outputs

Common outputs include:

- `data/labeled/ciu_tokens_normalized.parquet`
- `data/splits/prompt_ids.txt`
- `data/splits/eval_ids.txt`
- `models/baselines/linear_svc_baseline.joblib`
- `results/raw/hf_local/<model>/<mode>/seed<seed>/`
- `results/parsed/*.parquet`
- `results/metrics/<model>/<mode>/seed<seed>/summary_per_run.csv`
- `results/metrics/<model>/<mode>/seed<seed>/summary_aggregated.csv`
- `results/metrics/<model>/<mode>/seed<seed>/summary_per_severity.csv`
- `results/metrics/<model>/<mode>/seed<seed>/mcnemar_tests.csv`
- `log/*.log`

Most scripts also write JSON sidecars with run metadata for reproducibility.

## Parsing And Evaluation Notes

- The HF inference path writes one JSON wrapper per chunk, not per transcript.
- Parsers include robust JSON extraction and parse-quality reporting.
- Token-count mismatches can be handled with `drop` or `truncate`, depending on the parser option you choose.
- Metric computation filters to `word_label == 1` before scoring CIU predictions.

## Repository Layout

```text
.
├── README.md
├── LICENSE
├── config.yaml
├── requirements.txt
├── data/
│   ├── labeled/
│   └── splits/
├── docs/
│   ├── article/
│   ├── notes/
│   └── presentation/
├── log/
├── logs/
├── models/
├── prompts/
├── results/
│   ├── metrics/
│   ├── parsed/
│   └── raw/
├── scratch/
├── scripts/
│   └── run_all.sh
└── src/
```

## Reproducibility

The codebase is set up for reproducible multi-run experiments:

- most scripts accept a `--seed` argument,
- `run_all.sh` runs a fixed seed set of `2025 2026 2027 2028 2029`,
- log files are timestamped,
- run metadata sidecars record configuration details and library versions.

## License

Apache License 2.0. See `LICENSE`.

#!/usr/bin/env bash

# run_all.sh — Full experimental pipeline for CIU LLM benchmarking study.
#
# __author__    = "Jason M. Pittman"
# __copyright__ = "Copyright 2026"
# __credits__   = ["Jason M. Pittman"]
# __license__   = "Apache License 2.0"
# __version__   = "0.2.3"
# __maintainer__ = "Jason M. Pittman"
# __status__    = "Research"
#
# Runs the complete model zoo × mode × seed matrix, followed by ablation
# trials on the best-performing configuration.  All stdout/stderr is tee'd
# to log/run_all_<timestamp>.log so every run is fully auditable.
#
# Usage:
#   bash run_all.sh [--dry-run] [--skip-finetune] [--skip-ablations]
#
# Flags:
#   --dry-run          Echo commands without executing them (useful for
#                      verifying the full matrix before committing GPU time).
#   --skip-finetune    Skip the fine-tune + LoRA inference steps.
#   --skip-ablations   Skip the few-shot count and selection-strategy ablations.
#
# Prerequisites:
#   - Python environment with requirements.txt installed.
#   - data/labeled/ciu_tokens.csv present.
#   - prompts/ciu_prompts.yaml present.
#
# Exit behaviour:
#   set -euo pipefail — the script aborts on any non-zero exit, unset
#   variable reference, or pipeline failure.  Each completed cell is
#   checkpointed to log/completed_cells.txt so a restarted run can skip
#   already-finished work (see the `cell_done` / `mark_done` helpers).

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
DRY_RUN=false
SKIP_FINETUNE=false
SKIP_ABLATIONS=false

for arg in "$@"; do
  case $arg in
    --dry-run)         DRY_RUN=true ;;
    --skip-finetune)   SKIP_FINETUNE=true ;;
    --skip-ablations)  SKIP_ABLATIONS=true ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Logging setup — tee everything to a timestamped file under log/
# ---------------------------------------------------------------------------
mkdir -p log
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
LOG_FILE="log/run_all_${TIMESTAMP}.log"

# Redirect all stdout + stderr through tee so the terminal still shows output
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================"
echo "CIU LLM Benchmarking Pipeline — started at $(date -u)"
echo "Log file: $LOG_FILE"
echo "DRY_RUN=$DRY_RUN  SKIP_FINETUNE=$SKIP_FINETUNE  SKIP_ABLATIONS=$SKIP_ABLATIONS"
echo "================================================================"

# ---------------------------------------------------------------------------
# Checkpoint helpers — skip cells already completed in a prior interrupted run
# ---------------------------------------------------------------------------
CHECKPOINT_FILE="log/completed_cells.txt"
touch "$CHECKPOINT_FILE"

cell_done() {
  # Returns 0 (true) if $1 is already recorded as complete.
  grep -qxF "$1" "$CHECKPOINT_FILE"
}

mark_done() {
  echo "$1" >> "$CHECKPOINT_FILE"
  echo "[checkpoint] Marked done: $1"
}

# ---------------------------------------------------------------------------
# Wrapper that respects --dry-run and logs each command
# ---------------------------------------------------------------------------
run() {
  echo ""
  echo ">>> $*"
  if [ "$DRY_RUN" = false ]; then
    "$@"
  fi
}

# ---------------------------------------------------------------------------
# Paths and experiment parameters
# ---------------------------------------------------------------------------
CONFIG_PATH="config.yaml"
LABELED_CSV="data/labeled/ciu_tokens.csv"
LABELED_PARQUET="data/labeled/ciu_tokens_normalized.parquet"
PROMPT_IDS="data/splits/prompt_ids.txt"
EVAL_IDS="data/splits/eval_ids.txt"

# Full model zoo (must match keys in config.yaml)
MODEL_KEYS=("phi3-mini" "llama3-8b" "qwen2.5-7b" "mistral-7b")

# All prompting modes
MODES=("z_shot_local" "few_shot_local" "few_shot_global")

# Seeds for multi-trial reproducibility — report mean ± std across these
SEEDS=(2025 2026 2027 2028 2029)

# Ablation: few-shot example counts (applied to few_shot_local on best model)
ABLATION_N_FEW_SHOT=(1 3 5 10)

# Ablation: few-shot selection strategies
ABLATION_STRATEGIES=("random" "severity_stratified")

# ---------------------------------------------------------------------------
# Step 1 — Data preparation (runs once, not per seed)
# ---------------------------------------------------------------------------
CELL="data_prep"
if cell_done "$CELL"; then
  echo "[skip] $CELL already complete."
else
  echo ""
  echo "=== Step 1: Normalize labeled tokens ==="
  run python src/data_prep.py \
    --input-path  "$LABELED_CSV" \
    --output-path "$LABELED_PARQUET"
  mark_done "$CELL"
fi

# ---------------------------------------------------------------------------
# Step 2 — Dataset split (runs once; seed fixed for stable train/eval boundary)
# ---------------------------------------------------------------------------
CELL="split_dataset"
if cell_done "$CELL"; then
  echo "[skip] $CELL already complete."
else
  echo ""
  echo "=== Step 2: Create prompt/eval splits ==="
  run python src/split_dataset.py \
    --input-path "$LABELED_PARQUET" \
    --prompt-n 5 \
    --seed 2025
  mark_done "$CELL"
fi

# ---------------------------------------------------------------------------
# Step 3 — Classic ML baseline (frozen before any LLM results are seen)
# ---------------------------------------------------------------------------
CELL="train_baselines"
if cell_done "$CELL"; then
  echo "[skip] $CELL already complete."
else
  echo ""
  echo "=== Step 3: Train and evaluate classic baselines ==="
  run python src/train_baselines.py \
    --input-path    "$LABELED_PARQUET" \
    --eval-ids-path "$EVAL_IDS" \
    --out-dir       models/baselines \
    --seed          2025
  mark_done "$CELL"
fi

# ---------------------------------------------------------------------------
# Step 4 — Main experiment: model zoo × mode × seed
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 4: Main experiment — model zoo × mode × seed ==="
echo "Models : ${MODEL_KEYS[*]}"
echo "Modes  : ${MODES[*]}"
echo "Seeds  : ${SEEDS[*]}"
echo "Total cells (prompt-based): $(( ${#MODEL_KEYS[@]} * ${#MODES[@]} * ${#SEEDS[@]} ))"

for MODEL_KEY in "${MODEL_KEYS[@]}"; do
  for MODE in "${MODES[@]}"; do
    for SEED in "${SEEDS[@]}"; do

      CELL="infer__${MODEL_KEY}__${MODE}__seed${SEED}"
      if cell_done "$CELL"; then
        echo "[skip] $CELL"
        continue
      fi

      echo ""
      echo "--- Inference: model=$MODEL_KEY  mode=$MODE  seed=$SEED ---"

      RAW_DIR="results/raw/hf_local/${MODEL_KEY}/${MODE}/seed${SEED}"
      PARSED="results/parsed/llm_predictions_${MODEL_KEY}_${MODE}_seed${SEED}.parquet"
      METRICS="results/metrics/${MODEL_KEY}/${MODE}/seed${SEED}"

      run python src/run_llm_inference.py \
        --config-path "$CONFIG_PATH" \
        --model-key   "$MODEL_KEY" \
        --mode        "$MODE" \
        --out-root    "$RAW_DIR" \
        --seed        "$SEED"

      run python src/parse_llm_outputs_hf.py \
        --labeled-path "$LABELED_PARQUET" \
        --raw-dir      "$RAW_DIR" \
        --out-path     "$PARSED"

      run python src/compute_metrics.py \
        --merged-path "$PARSED" \
        --out-dir     "$METRICS"

      mark_done "$CELL"
    done
  done
done

# ---------------------------------------------------------------------------
# Step 5 — Fine-tune + LoRA inference (optional)
# ---------------------------------------------------------------------------
if [ "$SKIP_FINETUNE" = true ]; then
  echo ""
  echo "[skip] Fine-tune steps (--skip-finetune set)."
else
  echo ""
  echo "=== Step 5: Fine-tune + LoRA inference ==="

  for MODEL_KEY in "${MODEL_KEYS[@]}"; do
    for SEED in "${SEEDS[@]}"; do

      CELL="finetune__${MODEL_KEY}__seed${SEED}"
      if cell_done "$CELL"; then
        echo "[skip] $CELL"
      else
        echo ""
        echo "--- Fine-tune: model=$MODEL_KEY  seed=$SEED ---"
        run python src/finetune_llm.py \
          --config-path     "$CONFIG_PATH" \
          --model-key       "$MODEL_KEY" \
          --labeled-path    "$LABELED_PARQUET" \
          --prompt-ids-path "$PROMPT_IDS" \
          --seed            "$SEED"
        mark_done "$CELL"
      fi

      # LoRA inference — zero-shot mode only (the model has been trained)
      CELL="infer_lora__${MODEL_KEY}__seed${SEED}"
      if cell_done "$CELL"; then
        echo "[skip] $CELL"
        continue
      fi

      ADAPTER_DIR="models/llm/${MODEL_KEY}-ciu-lora"
      RAW_DIR="results/raw/hf_local/${MODEL_KEY}/lora_z_shot/seed${SEED}"
      PARSED="results/parsed/llm_predictions_${MODEL_KEY}_lora_z_shot_seed${SEED}.parquet"
      METRICS="results/metrics/${MODEL_KEY}/lora_z_shot/seed${SEED}"

      echo ""
      echo "--- LoRA inference: model=$MODEL_KEY  seed=$SEED ---"
      run python src/run_llm_inference.py \
        --config-path "$CONFIG_PATH" \
        --model-key   "$MODEL_KEY" \
        --mode        z_shot_local \
        --out-root    "$RAW_DIR" \
        --use-lora \
        --adapter-dir "$ADAPTER_DIR" \
        --seed        "$SEED"

      run python src/parse_llm_outputs_hf.py \
        --labeled-path "$LABELED_PARQUET" \
        --raw-dir      "$RAW_DIR" \
        --out-path     "$PARSED"

      run python src/compute_metrics.py \
        --merged-path "$PARSED" \
        --out-dir     "$METRICS"

      mark_done "$CELL"
    done
  done
fi

# ---------------------------------------------------------------------------
# Step 6 — Ablations (run on best model identified from Step 4 results)
# ---------------------------------------------------------------------------
if [ "$SKIP_ABLATIONS" = true ]; then
  echo ""
  echo "[skip] Ablation steps (--skip-ablations set)."
else
  echo ""
  echo "=== Step 6: Ablations ==="
  echo "NOTE: Edit ABLATION_MODEL_KEY below once Step 4 results are reviewed."
  echo "      Defaulting to phi3-mini as a placeholder."

  # ---- Update this after inspecting Step 4 results ----
  ABLATION_MODEL_KEY="phi3-mini"
  ABLATION_MODE="few_shot_local"

  # -- Ablation A: few-shot example count --
  echo ""
  echo "--- Ablation A: n_few_shot in {${ABLATION_N_FEW_SHOT[*]}} ---"

  for N in "${ABLATION_N_FEW_SHOT[@]}"; do
    for SEED in "${SEEDS[@]}"; do

      CELL="ablation_n${N}__${ABLATION_MODEL_KEY}__seed${SEED}"
      if cell_done "$CELL"; then
        echo "[skip] $CELL"
        continue
      fi

      RAW_DIR="results/raw/ablations/n_few_shot/${ABLATION_MODEL_KEY}/n${N}/seed${SEED}"
      PARSED="results/parsed/ablation_nfewshot${N}_${ABLATION_MODEL_KEY}_seed${SEED}.parquet"
      METRICS="results/metrics/ablations/n_few_shot/${ABLATION_MODEL_KEY}/n${N}/seed${SEED}"

      echo ""
      echo "--- Ablation A: n=$N  model=$ABLATION_MODEL_KEY  seed=$SEED ---"
      run python src/run_llm_inference.py \
        --config-path "$CONFIG_PATH" \
        --model-key   "$ABLATION_MODEL_KEY" \
        --mode        "$ABLATION_MODE" \
        --out-root    "$RAW_DIR" \
        --n-few-shot  "$N" \
        --seed        "$SEED"

      run python src/parse_llm_outputs_hf.py \
        --labeled-path "$LABELED_PARQUET" \
        --raw-dir      "$RAW_DIR" \
        --out-path     "$PARSED"

      run python src/compute_metrics.py \
        --merged-path "$PARSED" \
        --out-dir     "$METRICS"

      mark_done "$CELL"
    done
  done

  # -- Ablation B: few-shot selection strategy --
  echo ""
  echo "--- Ablation B: selection strategy in {${ABLATION_STRATEGIES[*]}} ---"

  for STRATEGY in "${ABLATION_STRATEGIES[@]}"; do
    for SEED in "${SEEDS[@]}"; do

      CELL="ablation_strat_${STRATEGY}__${ABLATION_MODEL_KEY}__seed${SEED}"
      if cell_done "$CELL"; then
        echo "[skip] $CELL"
        continue
      fi

      RAW_DIR="results/raw/ablations/strategy/${ABLATION_MODEL_KEY}/${STRATEGY}/seed${SEED}"
      PARSED="results/parsed/ablation_${STRATEGY}_${ABLATION_MODEL_KEY}_seed${SEED}.parquet"
      METRICS="results/metrics/ablations/strategy/${ABLATION_MODEL_KEY}/${STRATEGY}/seed${SEED}"

      echo ""
      echo "--- Ablation B: strategy=$STRATEGY  model=$ABLATION_MODEL_KEY  seed=$SEED ---"
      run python src/run_llm_inference.py \
        --config-path        "$CONFIG_PATH" \
        --model-key          "$ABLATION_MODEL_KEY" \
        --mode               "$ABLATION_MODE" \
        --out-root           "$RAW_DIR" \
        --few-shot-strategy  "$STRATEGY" \
        --seed               "$SEED"

      run python src/parse_llm_outputs_hf.py \
        --labeled-path "$LABELED_PARQUET" \
        --raw-dir      "$RAW_DIR" \
        --out-path     "$PARSED"

      run python src/compute_metrics.py \
        --merged-path "$PARSED" \
        --out-dir     "$METRICS"

      mark_done "$CELL"
    done
  done
fi

# ---------------------------------------------------------------------------
# Step 7 — Aggregate all per-cell parquets and run McNemar's tests
#
# McNemar's test is a pairwise comparison across (model, mode) conditions.
# It requires all predictions to be present in one parquet — running it
# per-cell (as Steps 4-6 do) always produces an empty result.  This step
# concatenates every per-cell parquet and runs compute_metrics.py once more
# on the combined file so McNemar's tests are actually populated.
# ---------------------------------------------------------------------------
CELL="aggregate_and_mcnemar"
if cell_done "$CELL"; then
  echo "[skip] $CELL already complete."
else
  echo ""
  echo "=== Step 7: Aggregate predictions and run McNemar's tests ==="

  AGGREGATED="results/parsed/all_predictions.parquet"
  AGGREGATED_METRICS="results/metrics/aggregate"

  # Concatenate all per-cell parquets via Python — handles missing files
  # gracefully so a partially-completed matrix still produces valid output.
  run python3 - << 'PYEOF'
import pandas as pd
from pathlib import Path

parsed_dir = Path("results/parsed")
parquets = sorted(parsed_dir.glob("*.parquet"))

if not parquets:
    print("WARNING: no parquet files found in results/parsed/ — skipping aggregation.")
else:
    dfs = []
    for p in parquets:
        try:
            dfs.append(pd.read_parquet(p))
            print(f"  loaded: {p.name}  ({len(dfs[-1])} rows)")
        except Exception as e:
            print(f"  WARNING: could not load {p.name}: {e}")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        out = Path("results/parsed/all_predictions.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out, index=False)
        print(f"Aggregated {len(dfs)} files → {len(combined)} rows → {out}")
    else:
        print("WARNING: no files could be loaded — aggregation skipped.")
PYEOF

  run python src/compute_metrics.py     --merged-path "$AGGREGATED"     --out-dir     "$AGGREGATED_METRICS"

  mark_done "$CELL"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Pipeline complete — $(date -u)"
echo "Log: $LOG_FILE"
echo "Checkpoint: $CHECKPOINT_FILE"
echo "================================================================"

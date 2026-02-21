#!/usr/bin/env bash

#__author__ = "Jason M. Pittman"
#__copyright__ = "Copyright 2025"
#__credits__ = ["Jason M. Pittman"]
#__license__ = "Apache License 2.0"
#__version__ = "0.1.0"
#__maintainer__ = "Jason M. Pittman"
#__status__ = "Research"

set -euo pipefail

CONFIG_PATH="config.yaml"
LABELED_CSV="data/labeled/ciu_tokens.csv"
LABELED_PARQUET="data/labeled/ciu_tokens_normalized.parquet"

# 1) Data prep & baselines (once)
echo "=== Step 1: Normalize labeled tokens ==="
python src/data_prep.py \
  --input-path "$LABELED_CSV" \
  --output-path "$LABELED_PARQUET"

echo "=== Step 2: Create prompt/eval splits ==="
python src/split_dataset.py \
  --input-path "$LABELED_PARQUET" \
  --prompt-n 30

echo "=== Step 3: Train classic baselines (models/baselines) ==="
python src/train_baselines.py \
  --input-path "$LABELED_PARQUET" \
  --eval-ids-path data/splits/eval_ids.txt \
  --out-dir models/baselines

# 2) Model zoo loop
MODEL_KEYS=("phi3-mini" "llama3-8b" "qwen2.5-7b")

for MODEL_KEY in "${MODEL_KEYS[@]}"; do
  echo "======================================="
  echo "Running pipeline for model_key: $MODEL_KEY"
  echo "======================================="

  echo "--- HF inference (z_shot_local) ---"
  python src/run_llm_inference.py \
    --config-path "$CONFIG_PATH" \
    --model-key "$MODEL_KEY" \
    --mode z_shot_local \
    --out-root results/raw/hf_local

  RAW_DIR_HF="results/raw/hf_local/${MODEL_KEY}/z_shot_local"
  MERGED_PARQUET="results/parsed/llm_predictions_${MODEL_KEY}_z_shot_local.parquet"
  METRICS_CSV="results/metrics/summary_${MODEL_KEY}_z_shot_local.csv"

  echo "--- Parse HF outputs ---"
  python src/parse_llm_outputs_hf.py \
    --labeled-path "$LABELED_PARQUET" \
    --raw-dir "$RAW_DIR_HF" \
    --out-path "$MERGED_PARQUET"

  echo "--- Compute metrics ---"
  python src/compute_metrics.py \
    --merged-path "$MERGED_PARQUET" \
    --out-path "$METRICS_CSV"

  echo "Done for $MODEL_KEY. Metrics: $METRICS_CSV"
done

cat << 'NOTE'

---------------------------------------------------------
Manual ChatGPT pipeline (independent of model zoo)
---------------------------------------------------------
1. Generate utterance-level prompts:
   python src/generate_chatgpt_prompts.py --mode z_shot_local

2. For each .txt in results/prompts/chatgpt/z_shot_local/:
   - Paste SYSTEM/USER into ChatGPT
   - Save JSON reply into results/raw/chatgpt/z_shot_local/<group_id>.txt

3. Parse & evaluate:
   python src/parse_llm_outputs_chatgpt.py \
     --labeled-path data/labeled/ciu_tokens_normalized.parquet \
     --raw-dir results/raw/chatgpt/z_shot_local \
     --out-path results/parsed/llm_predictions_chatgpt_z_shot_local.parquet \
     --model-name "chatgpt-webui" \
     --mode z_shot_local

   python src/compute_metrics.py \
     --merged-path results/parsed/llm_predictions_chatgpt_z_shot_local.parquet \
     --out-path results/metrics/summary_chatgpt_z_shot_local.csv
---------------------------------------------------------
NOTE


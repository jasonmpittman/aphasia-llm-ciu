#!/usr/bin/env bash
# slurm/sbatch_inference.sh — SLURM job script for one model's inference cells.
#
# __author__    = "Jason M. Pittman"
# __copyright__ = "Copyright 2026"
# __credits__   = ["Jason M. Pittman"]
# __license__   = "Apache License 2.0"
# __version__   = "1.0.1"
# __maintainer__ = "Jason M. Pittman"
# __status__    = "Research"
#
# Runs all 15 cells (3 modes x 5 seeds) for a single model key sequentially
# on one GPU node. Submitted once per model by submit_all.sh.
#
# BEFORE SUBMITTING — edit these two lines with your specific values:
#   -A YOUR_SLURM_ACCOUNT   Find in RT Projects portal under
#                           "Submitting Slurm Jobs with your Project's Account"
#   --mail-user             Replace with your IU email address
#
# Required environment variable (set by submit_all.sh via --export):
#   MODEL_KEY   Must match a key in config.yaml (e.g. llama3-8b)
#
# Resource rationale:
#   --gres=gpu:1      One GPU; 7-8B models in float16 fit on a single A100/H100
#   --mem=64G         ~28G model weights + 28G KV cache headroom + OS overhead
#   --cpus-per-task=8 Sufficient for DataLoader workers and tokenizer
#   --time=12:00:00   15 cells x ~45 min each = ~11h worst case

#SBATCH -J ciu_infer
#SBATCH -o log/slurm_infer_%j.out
#SBATCH -e log/slurm_infer_%j.err
#SBATCH -p general
#SBATCH -A YOUR_SLURM_ACCOUNT            # ← REQUIRED: replace with your account name
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@iu.edu   # ← replace with your IU email

set -euo pipefail

if [ -z "${MODEL_KEY:-}" ]; then
  echo "ERROR: MODEL_KEY environment variable is not set." >&2
  echo "Submit via submit_all.sh or pass --export=MODEL_KEY=<key> to sbatch." >&2
  exit 1
fi

echo "========================================================"
echo "CIU inference job — model: $MODEL_KEY"
echo "SLURM job ID  : $SLURM_JOB_ID"
echo "Node          : $SLURMD_NODENAME"
echo "Started       : $(date -u)"
echo "========================================================"

# Uncomment and edit the appropriate line for your environment:
# conda activate ciu_env
# source .venv/bin/activate
# module load python/3.11

cd "$SLURM_SUBMIT_DIR"
mkdir -p log

CHECKPOINT_FILE="log/completed_cells.txt"
touch "$CHECKPOINT_FILE"

cell_done() { grep -qxF "$1" "$CHECKPOINT_FILE"; }
mark_done() {
  echo "$1" >> "$CHECKPOINT_FILE"
  echo "[checkpoint] Marked done: $1"
}

CONFIG_PATH="config.yaml"
LABELED_PARQUET="data/labeled/ciu_tokens_normalized.parquet"
MODES=("z_shot_local" "few_shot_local" "few_shot_global")
SEEDS=(2025 2026 2027 2028 2029)

for MODE in "${MODES[@]}"; do
  for SEED in "${SEEDS[@]}"; do

    CELL="infer__${MODEL_KEY}__${MODE}__seed${SEED}"
    if cell_done "$CELL"; then
      echo "[skip] $CELL"
      continue
    fi

    echo ""
    echo "--- model=$MODEL_KEY  mode=$MODE  seed=$SEED ---"

    RAW_DIR="results/raw/hf_local/${MODEL_KEY}/${MODE}/seed${SEED}"
    PARSED="results/parsed/llm_predictions_${MODEL_KEY}_${MODE}_seed${SEED}.parquet"
    METRICS="results/metrics/${MODEL_KEY}/${MODE}/seed${SEED}"

    srun python src/run_llm_inference.py \
      --config-path "$CONFIG_PATH" \
      --model-key   "$MODEL_KEY" \
      --mode        "$MODE" \
      --out-root    "$RAW_DIR" \
      --seed        "$SEED"

    srun python src/parse_llm_outputs_hf.py \
      --labeled-path "$LABELED_PARQUET" \
      --raw-dir      "$RAW_DIR" \
      --out-path     "$PARSED"

    srun python src/compute_metrics.py \
      --merged-path "$PARSED" \
      --out-dir     "$METRICS"

    mark_done "$CELL"

  done
done

echo ""
echo "========================================================"
echo "Inference job complete — model: $MODEL_KEY"
echo "Finished: $(date -u)"
echo "========================================================"
#!/usr/bin/env bash
# slurm/submit_all.sh — Submit the full CIU benchmarking matrix to SLURM.
#
# __author__    = "Jason M. Pittman"
# __copyright__ = "Copyright 2026"
# __credits__   = ["Jason M. Pittman"]
# __license__   = "Apache License 2.0"
# __version__   = "1.0.1"
# __maintainer__ = "Jason M. Pittman"
# __status__    = "Research"
#
# Submits four parallel GPU inference jobs (one per model) and one aggregate
# job that waits for all four to succeed before running.
#
# Job dependency graph:
#
#   sbatch_inference.sh [phi3-mini]  --+
#   sbatch_inference.sh [llama3-8b]  --+
#   sbatch_inference.sh [qwen2.5-7b] --+--> sbatch_aggregate.sh
#   sbatch_inference.sh [mistral-7b] --+
#
# Usage:
#   bash slurm/submit_all.sh [--dry-run]
#
# Prerequisites:
#   - Steps 1-3 already complete (data_prep, split_dataset, train_baselines).
#     Run these once locally before submitting to the cluster:
#       python src/data_prep.py
#       python src/split_dataset.py --prompt-n 5 --seed 2025
#       python src/train_baselines.py --seed 2025
#   - -A and --mail-user filled in sbatch_inference.sh and sbatch_aggregate.sh
#   - Python environment activation line uncommented in both sbatch scripts

set -euo pipefail

DRY_RUN=false
for arg in "$@"; do
  case $arg in
    --dry-run) DRY_RUN=true ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
submit() {
  local desc="$1"; shift
  if [ "$DRY_RUN" = true ]; then
    echo "  [dry-run] sbatch $*" >&2
    echo "DRY_RUN_ID"
  else
    local job_id
    job_id=$(sbatch --parsable "$@")
    echo "  Submitted $desc -> job $job_id" >&2
    echo "$job_id"
  fi
}

# ---------------------------------------------------------------------------
# Validate scripts exist
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_SCRIPT="$SCRIPT_DIR/sbatch_inference.sh"
AGG_SCRIPT="$SCRIPT_DIR/sbatch_aggregate.sh"

for f in "$INFER_SCRIPT" "$AGG_SCRIPT"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: $f not found." >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Ensure output directories exist before submission
# ---------------------------------------------------------------------------
mkdir -p log results/parsed results/metrics

echo "========================================================"
echo "CIU LLM Benchmarking — SLURM submission"
echo "Dry run: $DRY_RUN"
echo "========================================================"
echo ""

# ---------------------------------------------------------------------------
# Submit one inference job per model — these run in parallel
# --chdir ensures each job's relative paths resolve from the project root
# ---------------------------------------------------------------------------
MODEL_KEYS=("phi3-mini" "llama3-8b" "qwen2.5-7b" "mistral-7b")
INFER_JOB_IDS=()

echo "Submitting inference jobs..."
for MODEL_KEY in "${MODEL_KEYS[@]}"; do
  JOB_ID=$(submit \
    "inference [$MODEL_KEY]" \
    --export="MODEL_KEY=${MODEL_KEY}" \
    --chdir="$(pwd)" \
    "$INFER_SCRIPT"
  )
  INFER_JOB_IDS+=("$JOB_ID")
  echo "  model=$MODEL_KEY  job_id=$JOB_ID"
done

# ---------------------------------------------------------------------------
# Submit aggregate job — held until ALL inference jobs succeed
# afterok means: run only if every listed job exited with code 0
# ---------------------------------------------------------------------------
echo ""
echo "Submitting aggregate job (depends on all inference jobs)..."

DEPENDENCY="afterok"
for JOB_ID in "${INFER_JOB_IDS[@]}"; do
  DEPENDENCY="${DEPENDENCY}:${JOB_ID}"
done

if [ "$DRY_RUN" = true ]; then
  echo "  [dry-run] sbatch --dependency=$DEPENDENCY --chdir=$(pwd) $AGG_SCRIPT"
  AGG_JOB_ID="DRY_RUN_AGG_ID"
else
  AGG_JOB_ID=$(sbatch --parsable \
    --dependency="$DEPENDENCY" \
    --chdir="$(pwd)" \
    "$AGG_SCRIPT")
fi

echo "  aggregate  job_id=$AGG_JOB_ID"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo "Submission complete."
echo ""
echo "Inference jobs (parallel, one GPU each):"
for i in "${!MODEL_KEYS[@]}"; do
  printf "  %-20s job %s\n" "${MODEL_KEYS[$i]}" "${INFER_JOB_IDS[$i]}"
done
echo ""
echo "Aggregate job (held until all inference jobs succeed):"
echo "  job $AGG_JOB_ID"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     tail -f log/slurm_infer_<job_id>.out"
echo ""
echo "If any job fails, fix and resubmit with the same command."
echo "Completed cells in log/completed_cells.txt will be skipped."
echo "========================================================"
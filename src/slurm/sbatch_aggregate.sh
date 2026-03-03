#!/usr/bin/env bash
# slurm/sbatch_aggregate.sh — Aggregate predictions and run McNemar's tests.
#
# __author__    = "Jason M. Pittman"
# __copyright__ = "Copyright 2026"
# __credits__   = ["Jason M. Pittman"]
# __license__   = "Apache License 2.0"
# __version__   = "1.0.1"
# __maintainer__ = "Jason M. Pittman"
# __status__    = "Research"
#
# Concatenates all per-cell prediction parquets and runs compute_metrics.py
# on the combined file so McNemar's pairwise tests are populated across all
# (model, mode) conditions.
#
# Submitted by submit_all.sh with --dependency=afterok on all inference jobs.
# No GPU needed — pure pandas + scipy work.

#SBATCH -J ciu_aggregate
#SBATCH -o log/slurm_aggregate_%j.out
#SBATCH -e log/slurm_aggregate_%j.err
#SBATCH -p general
#SBATCH -A YOUR_SLURM_ACCOUNT            # ← REQUIRED: same account as inference jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@iu.edu   # ← replace with your IU email

set -euo pipefail

echo "========================================================"
echo "CIU aggregation + McNemar job"
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

CELL="aggregate_and_mcnemar"
if cell_done "$CELL"; then
  echo "[skip] $CELL already complete."
else
  echo ""
  echo "=== Aggregating per-cell parquets ==="

  AGGREGATED="results/parsed/all_predictions.parquet"
  AGGREGATED_METRICS="results/metrics/aggregate"

  srun python3 - << 'PYEOF'
import pandas as pd
from pathlib import Path

parsed_dir = Path("results/parsed")
parquets = sorted(p for p in parsed_dir.glob("*.parquet")
                  if p.name != "all_predictions.parquet")

if not parquets:
    raise RuntimeError("No per-cell parquet files found in results/parsed/")

dfs = []
for p in parquets:
    try:
        dfs.append(pd.read_parquet(p))
        print(f"  loaded: {p.name}  ({len(dfs[-1])} rows)")
    except Exception as e:
        print(f"  WARNING: could not load {p.name}: {e}")

if not dfs:
    raise RuntimeError("No parquet files could be loaded.")

combined = pd.concat(dfs, ignore_index=True)
out = Path("results/parsed/all_predictions.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
combined.to_parquet(out, index=False)
print(f"\nAggregated {len(dfs)} files -> {len(combined)} rows -> {out}")
PYEOF

  echo ""
  echo "=== Running compute_metrics on aggregated predictions ==="
  srun python src/compute_metrics.py \
    --merged-path "$AGGREGATED" \
    --out-dir     "$AGGREGATED_METRICS"

  mark_done "$CELL"
fi

echo ""
echo "========================================================"
echo "Aggregation job complete."
echo "McNemar results: results/metrics/aggregate/mcnemar_tests.csv"
echo "Finished: $(date -u)"
echo "========================================================"
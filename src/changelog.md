# Changelog

## [Unreleased]

## [0.2.1] - 2026-23-02

### Added

### Changed
**run_all.sh**  
- Full matrix coverage — the loop now runs all 4 models × 3 modes × 5 seeds = 60 prompt-based inference cells, plus fine-tune + LoRA inference for each model × seed.  
- Checkpointing — every completed cell is written to log/completed_cells.txt. If the script is interrupted (GPU OOM, timeout, etc.) and restarted, already-finished cells are skipped. This is essential at the scale you're running.  
- Tee logging — exec > >(tee -a "$LOG_FILE") 2>&1 at the top routes all stdout and stderr to both the terminal and a timestamped file under log/. Every command, every model output, every error is captured permanently without changing any individual script.  
- --dry-run flag — lets you verify the full command matrix before committing any compute. Run bash run_all.sh --dry-run to see every command that would execute.  
- --skip-finetune and --skip-ablations — lets you run stages independently, which is useful when iterating.  

**utils.py**  
setup_logger() — creates a logger writing DEBUG+ to a timestamped file under log/ and INFO+ to the console. Safe to call multiple times within the same process (checks for existing handlers). Every script should call this once at entry.  
save_run_metadata() — writes a JSON sidecar with UTC timestamp, library versions, and any kwargs you pass (model_key, mode, seed, etc.). Outputs become self-describing.  
build_few_shot_block() — new strategy parameter — adds severity_stratified sampling alongside the existing random default, wiring in Ablation B without changing the call signature for existing code.  
set_global_seed() — now also seeds torch.cuda.manual_seed_all() for multi-GPU environments.  

### Removed




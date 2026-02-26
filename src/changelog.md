# Changelog

## [Unreleased]

## [0.2.7] - 2026-25-02
-- updated ciu_prompts.yaml and config.yaml to extend max_new_tokens and modify prompt format to not truncate.

## [0.2.6] - 2026-24-02
-- apply_chat_template() now wraps the messages in the model-specific special tokens that activate instruction-following mode.  

## [0.2.5] - 2026-23-02
-- modified ciu_prompts.yaml and associated methods to refine cross model zoo prompt structures to avoid truncation issues.

## [0.2.4] - 2026-23-02
-- torch_dtype per device. CUDA gets float16 (safe and memory-efficient), MPS and CPU both get float32. Llama-3.1's model config requests bfloat16 by default, but MPS bfloat16 support is incomplete before PyTorch 2.4 and CPU bfloat16 ops are unreliable — both cause the integral conversion overflow you saw.  
-- device_map=None. Without this, from_pretrained can trigger automatic device mapping that re-introduces bfloat16 on MPS regardless of what you pass as torch_dtype. Explicit to(device) after loading is the safe pattern.  
-- use_fast=True and model_max_length clamp. Llama-3.1's tokenizer ships with a sentinel model_max_length of 1e30. Some transformers versions interpret this literally when allocating attention mask tensors, which can also cause overflow. Clamping to 4096 matches the model's actual context window.  

## [0.2.3] - 2026-23-02
-- adjusted max_new_tokens, min_chunk_size in config.yaml, utils.py, and run_llm_inference.py to fix response truncation.

## [0.2.2] - 2026-23-02
Implement token chunking to prevent prompt and ouput truncation due to model context windows

**utils.py**
Two new functions added:  
-- chunk_transcript() — takes a group DataFrame, chunk_size, and identifiers, and returns a list of chunk dicts. Each chunk carries chunk_id, chunk_index, n_chunks, token_start, token_end, tokens, and critically token_indices — the original token_index values from the dataset. This is what allows the parser to align predictions back to ground truth by index rather than by position, which is robust to any dropped or failed chunks.  
-- build_token_block_from_chunk() — renders the numbered token block string for a chunk using original token_index values as the line numbers. 

**run_llm_inference.py**
-- --chunk-size flag (default 50) added. Every group is now passed through chunk_transcript() unconditionally — single-chunk passthrough uses max(chunk_size, len(g)) so short transcripts are unchanged. Each chunk gets its own JSON wrapper file named <group_id>__chunk000.json, carrying the full chunk provenance. Setting --chunk-size 0 disables chunking for ablation purposes if needed.  

**parse_llm_outputs_hf.py**
-- Previous single-pass file loop is replaced with a two-stage process: first all wrappers are loaded and grouped by group_id, then each group's chunks are sorted by chunk_index, parsed individually, and their predictions assembled into a token_pred_map keyed by original token_index. Mismatch checks happen at the chunk level against len(token_indices). Legacy wrappers without token_indices fall back to using pred["index"] directly, so the parser remains backward compatible. The report.success counter now increments per group rather than per chunk.  


## [0.2.1] - 2026-23-02

### Added

### Changed
**run_all.sh**  
-- Full matrix coverage — the loop now runs all 4 models × 3 modes × 5 seeds = 60 prompt-based inference cells, plus fine-tune + LoRA inference for each model × seed.  
--- Checkpointing — every completed cell is written to log/completed_cells.txt. If the script is interrupted (GPU OOM, timeout, etc.) and restarted, already-finished cells are skipped. This is essential at the scale you're running.  
--- Tee logging — exec > >(tee -a "$LOG_FILE") 2>&1 at the top routes all stdout and stderr to both the terminal and a timestamped file under log/. Every command, every model output, every error is captured permanently without changing any individual script.  
--- --dry-run flag — lets you verify the full command matrix before committing any compute. Run bash run_all.sh --dry-run to see every command that would execute.  
--- --skip-finetune and --skip-ablations — lets you run stages independently, which is useful when iterating.  

**utils.py**  
-- setup_logger() — creates a logger writing DEBUG+ to a timestamped file under log/ and INFO+ to the console. Safe to call multiple times within the same process (checks for existing handlers). Every script should call this once at entry.  
-- save_run_metadata() — writes a JSON sidecar with UTC timestamp, library versions, and any kwargs you pass (model_key, mode, seed, etc.). Outputs become self-describing.  
-- build_few_shot_block() — new strategy parameter — adds severity_stratified sampling alongside the existing random default, wiring in Ablation B without changing the call signature for existing code.  
-- set_global_seed() — now also seeds torch.cuda.manual_seed_all() for multi-GPU environments. 

**compute_metrics.py**
-- compute_per_run_metrics() replaces the original flat loop. It groups by (model_name, mode, seed) and computes bootstrapped 95% CIs for both F1 and Kappa using percentile bootstrap with 1,000 resamples (configurable via --n-bootstrap). The seed for each bootstrap is derived from the run seed so resampling is itself reproducible.  
-- aggregate_across_seeds() then collapses per-run rows into mean ± std for each (model, mode) pair — this is what goes in a results table in the paper.  
-- _bootstrap_ci() is a clean general-purpose helper so it can be reused if you later want CIs on per-severity metrics too.
-- compute_per_severity_metrics() groups by (model_name, mode, seed, severity) and runs the same point estimate logic independently for each stratum. Null severities are bucketed as "unknown" rather than silently dropped. The function warns and returns an empty DataFrame gracefully if the severity column is absent.  
-- run_mcnemar_tests() builds a token-level pivot across all (model, mode) conditions pooled across seeds, then runs pairwise McNemar's tests with Yates' continuity correction. The output CSV includes the full 2×2 table per pair, the test statistic, p-value, and a boolean significant_0_05 flag — making it straightforward to identify which model differences are statistically real rather than just numerically different.  

**parse_llm_ouputs_hf.py**
-- Every failure is now categorised into one of five named types — empty_response, no_json_array, json_decode_error, missing_fields, token_count_mismatch — and recorded in a ParseFailure dataclass rather than silently skipped with a print().  
-- try three strategies in sequence: bracket scan, regex extraction from prose/markdown fences, and truncation repair (appending closing tokens for cut-off responses).  
-- mismatches are explicitly detected, logged with the delta, and handled via a configurable --mismatch-strategy: drop (default, safe for metric computation) or truncate (for exploratory use, with a note to flag in the paper if used). The count of dropped and truncated tokens is tracked and reported.  
-- A warning is emitted if the failure rate exceeds 10%, since at that level the metrics are materially affected and the paper should note it explicitly.  

**run_llm_inference.py**
-- Logger — named run_llm_inference__{model_key}__{mode}__seed{seed} so each combination gets its own timestamped log file. This matters at scale: with 60 cells running you want to be able to open a single log and see exactly what happened for llama3-8b / few_shot_local / seed2027 without grepping through a monolithic file.  
-- seed written into every output JSON — this was the critical missing link for compute_metrics.py to group by seed. Previously the seed was only recoverable from the directory path, which is fragile.  
-- --few-shot-strategy flag — wires in the strategy parameter added to build_few_shot_block() in utils.py. run_all.sh already passes --few-shot-strategy in the Ablation B loop; this makes the CLI accept it.  
-- Inference error handling — the generation call is now wrapped in try/except. Failed groups are counted and logged rather than crashing the whole run, and the final summary reports success / error / total so you know immediately if any groups were skipped.  
-- CUDA device support — priority is now CUDA > MPS > CPU, consistent with running on a GPU server rather than just Apple Silicon.  


**finetune_llm.py**
-- Logger — named finetune_llm__{model_key}__seed{seed}, written to log/ alongside inference logs.  
save_run_metadata() — written to adapter_dir/finetune_metadata.json so every saved checkpoint is self-describing: you can open any adapter directory and immediately know what model, seed, hyperparameters, and number of training examples produced it.  
-- TrainConfig logging — all hyperparameters are logged at INFO level before training starts, giving you a clean record of the exact configuration used for each fine-tuning run.  

**others**
-- added structured logging, run metadata sidecars, and consistent error handling throughout.

### Removed




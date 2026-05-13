# EMBC Project Asset Index

Snapshot date: 2026-05-12

This file records important non-code assets for the EMBC TruthRL/Clinical-R1 work. It is a path index only. Do not commit model weights, checkpoints, Apptainer images, generated parquet files, logs, or cache directories to GitHub.

## GitHub Backup Status

The primary code repositories are backed up as follows.

| Repository | Local path | Remote | Branch / commit | Status |
| --- | --- | --- | --- | --- |
| TruthRL | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/TruthRL` | `https://github.com/ManOnTheMove/TruthRL.git` | `embc_clinical_truthrl_v1` at `40fa386` | Backed up |
| CPRO | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/CPRO` | `https://github.com/ManOnTheMove/CPRO.git` | `main` at `a087583` | Backed up |

Important code now covered by the TruthRL backup includes:

| Area | Important paths |
| --- | --- |
| Cold-start / Stage C SFT scripts | `TruthRL/scripts/medical/run_stagec_c7_train_8b_lora_param.sh`, `TruthRL/scripts/medical/submit_stagec_c7_train_8b_len10240_ep10_*` |
| KBP collection and merge scripts | `TruthRL/scripts/medical/kbp_medqa_collect.py`, `TruthRL/scripts/medical/kbp_merge_shards.py`, `TruthRL/scripts/medical/run_kbp_medqa.sh`, `TruthRL/scripts/medical/run_kbp_merge_and_postprocess.sh`, `TruthRL/scripts/medical/submit_kbp_medqa_*.slurm` |
| GRPO / Stage D scripts | `TruthRL/scripts/medical/run_staged_d1_grpo_truthrl.sh`, `TruthRL/scripts/medical/run_staged_d2_grpo_truthrl_cpro.sh`, `TruthRL/scripts/medical/run_staged_d3_hard_ook_truthrl.sh`, `TruthRL/scripts/medical/submit_staged_d*.slurm` |
| Stage D evaluation | `TruthRL/scripts/medical/stageD_eval_collect.py`, `TruthRL/scripts/medical/stageD_eval_compare.py`, `TruthRL/scripts/medical/stageD_d3_diagnose_ook_uptake.py` |
| Medical reward and trainer changes | `TruthRL/training/verl/verl/utils/reward_score/clinical_medqa_reward.py`, `TruthRL/training/verl/verl/trainer/main_ppo.py`, `TruthRL/training/verl/verl/trainer/config/ppo_trainer.yaml` |

## Important Model and Checkpoint Paths

These directories are not suitable for GitHub. They should be preserved on project storage or copied to a dedicated model artifact store.

### Stage C merged models

| Purpose | Path | Notes |
| --- | --- | --- |
| Primary 8B Stage C merged model | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/models/stagec_c8_8b_len10240_ep15_4gpu_step17175/merged_model` | About 16G. Contains HF model shards and tokenizer files. Important for Stage D baseline/evaluation. |
| Stage C step 17000 reference | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/models/stagec_c8_8b_len10240_ep15_4gpu_step17000/merged_model` | Smaller local directory. Keep as reference if it was used in earlier evaluation reports. |
| Stage C step 1500 reference | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/models/stagec_c8_8b_len10240_ep15_2gpu_step1500/merged_model` | Earlier 8B checkpoint export/reference. |
| Stage C 8B full / fixalpha references | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/models/stagec_c8_8b_full/merged_model`, `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/models/stagec_c8_8b_full_fixalpha/merged_model` | Useful for debugging LoRA alpha and merge behavior. |
| 0.6B smoke models | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/models/stagec_smoke_0p6b*`, `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/models/stagec_c7c8_0p6b_smoke` | Smoke-test artifacts, lower priority than 8B assets. |

Key config files for the primary 8B model:

```text
/home/erichyu/links/projects/def-zshakeri/erichyu/embc/models/stagec_c8_8b_len10240_ep15_4gpu_step17175/merged_model/config.json
/home/erichyu/links/projects/def-zshakeri/erichyu/embc/models/stagec_c8_8b_len10240_ep15_4gpu_step17175/merged_model/generation_config.json
/home/erichyu/links/projects/def-zshakeri/erichyu/embc/models/stagec_c8_8b_len10240_ep15_4gpu_step17175/merged_model/tokenizer_config.json
```

### Stage C FSDP checkpoints

| Run | Path | Latest step | Approx size |
| --- | --- | --- | --- |
| 4 GPU, LoRA rank 8 | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/TruthRL/data/medical/stagec_c7_8b_len10240_ep10_4gpu_lora8/sft_ckpt/global_step_11450` | `11450` | About 16G |
| 8 GPU, LoRA rank 16 | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/TruthRL/data/medical/stagec_c7_8b_len10240_ep10_8gpu_lora16/sft_ckpt/global_step_5720` | `5720` | About 16G |
| 8 GPU, LoRA rank 32 | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/TruthRL/data/medical/stagec_c7_8b_len10240_ep10_8gpu_lora32/sft_ckpt/global_step_5720` | `5720` | About 16G |

Associated summary files:

```text
/home/erichyu/links/projects/def-zshakeri/erichyu/embc/TruthRL/data/medical/stagec_c7_8b_len10240_ep10_4gpu_lora8/reports/stageC_c7_train_8b_len10240_ep10_4gpu_lora8_summary.json
/home/erichyu/links/projects/def-zshakeri/erichyu/embc/TruthRL/data/medical/stagec_c7_8b_len10240_ep10_8gpu_lora16/reports/stageC_c7_train_8b_len10240_ep10_8gpu_lora16_summary.json
/home/erichyu/links/projects/def-zshakeri/erichyu/embc/TruthRL/data/medical/stagec_c7_8b_len10240_ep10_8gpu_lora32/reports/stageC_c7_train_8b_len10240_ep10_8gpu_lora32_summary.json
```

### Stage D D3 hard-OOK GRPO run

The Stage D D3 run is currently under scratch storage. Scratch storage should be treated as less durable than project storage.

| Asset | Path | Notes |
| --- | --- | --- |
| Full Stage D run directory | `/scratch/erichyu/stageD_d3_hard_ook` | About 290G. Highest-priority non-Git artifact to preserve if Stage D results matter. |
| GRPO checkpoints | `/scratch/erichyu/stageD_d3_hard_ook/run/checkpoints/global_step_100`, `/scratch/erichyu/stageD_d3_hard_ook/run/checkpoints/global_step_200`, `/scratch/erichyu/stageD_d3_hard_ook/run/checkpoints/global_step_300` | Latest tracker reports step `300`. |
| Latest tracker | `/scratch/erichyu/stageD_d3_hard_ook/run/checkpoints/latest_checkpointed_iteration.txt` | Contains `300`. |
| HF export of D3 step 300 | `/scratch/erichyu/stageD_d3_hard_ook/export/global_step_300_hf` | Contains HF config, tokenizer, and four `.safetensors` shards. |
| Phase 2 / Phase 3 eval | `/scratch/erichyu/stageD_d3_hard_ook/eval/phase2_phase3/phase23_20260511_standard_d26_stagec_vs_d3` | Contains `phase2_phase3_eval_report.md`, predictions, manifests, and metrics. |
| D3 uptake diagnostics | `/scratch/erichyu/stageD_d3_hard_ook/diagnostics/d3_ook_uptake_20260511/d3_ook_uptake_diagnosis.json` | Diagnostic record for OOK uptake behavior. |

If there is only one Stage D artifact to move off scratch, prioritize:

```text
/scratch/erichyu/stageD_d3_hard_ook/export/global_step_300_hf
/scratch/erichyu/stageD_d3_hard_ook/run/checkpoints/global_step_300
/scratch/erichyu/stageD_d3_hard_ook/eval/phase2_phase3/phase23_20260511_standard_d26_stagec_vs_d3
/scratch/erichyu/stageD_d3_hard_ook/diagnostics/d3_ook_uptake_20260511/d3_ook_uptake_diagnosis.json
```

## Data and KBP Artifacts

| Asset | Path | Notes |
| --- | --- | --- |
| Main medical data root | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/TruthRL/data/medical` | Ignored by Git. Contains raw, processed, parquet, KBP, Stage C, and Stage D smoke outputs. |
| veRL medical parquet data | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/TruthRL/data/medical/verl` | Contains generated training/eval parquet files. Do not commit to GitHub. |
| Processed CSVs | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/TruthRL/data/medical/processed_csv` | Derived data. Useful for rebuild/debug. |
| KBP run root | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/TruthRL/data/medical/kbp` | About 2.1G. Ignored by Git. |
| Full KBP pass | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/TruthRL/data/medical/kbp/runs/kbp_20260319_stagec-c8-8b_medqa_grpo_train_fullpass` | Full KBP pass for Stage C 8B. |
| Earlier KBP pass | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/TruthRL/data/medical/kbp/runs/kbp_20260301_stagec-c8-8b_medqa_grpo_train` | Earlier KBP run used in D2.5/D2.6 analysis. |
| Pilot KBP pass | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/TruthRL/data/medical/kbp/runs/kbp_20260301_stagec-c8-8b_medqa_grpo_train_pilot10` | Pilot run. Lower priority. |

## Environment and Container Assets

| Asset | Path | Backup policy |
| --- | --- | --- |
| Apptainer definition file | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/envs/truthrl_apptainer/truthrl.def` | Back this up with source/supporting materials. Small and reproducibility-critical. |
| TruthRL Apptainer image | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/envs/truthrl_apptainer/truthrl.sif` | About 8.98G. Do not upload to GitHub. Archive only if needed. |
| vLLM OpenAI Apptainer image | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/envs/truthrl_apptainer/vllm-openai_v0.8.5.post1.sif` | About 8.68G. Do not upload to GitHub. Archive only if needed. |

## Progress Documents, Plans, and Reports

These are not all inside Git repositories. They should be included in a lightweight supporting-materials backup, excluding model weights and generated large artifacts.

| Area | Important paths |
| --- | --- |
| Project instructions | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/AGENTS.md` |
| Technical audit | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/technical_audit_report_2026-04-21.md` |
| Papers | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/paper/TruthRL.md`, `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/paper/CRPO.md` |
| Integrated plans | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/chat_with_llm/plan/final_integrated_plan.md`, `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/chat_with_llm/plan/truthrl_clinicalr1_integration_plan.md` |
| Stage C plans and reports | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/chat_with_llm/plan/StageC/stageC_implementation_plan_2026-02-15.md`, `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/chat_with_llm/reports/stagec_cold_start_sft_explainer_2026-02-26.md`, `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/chat_with_llm/stage_reports/stage_c/stageC_to_stageD_handover_report_2026-02-24.md` |
| Stage D plans and reports | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/chat_with_llm/plan/StageD/stageD_execution_guideline_for_agents_truthrl_first_2026-02-26.md`, `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/chat_with_llm/reports/stageD_current_status_and_research_questions_2026-05-10.md`, `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/chat_with_llm/reports/stageD_d3_hard_ook_training_report_2026-05-07.md`, `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/chat_with_llm/reports/stageD_phase2_phase3_eval_report_2026-05-11.md` |
| KBP records | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/chat_with_llm/reports/D2_5_KBP_Execution_Record_2026-03-19.md`, `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/chat_with_llm/reports/D2_5_KBP_Closure_Record_2026-03-25.md`, `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/chat_with_llm/reports/D2_6_OOK_Prompt_Execution_Record_2026-03-25.md` |
| Environment and migration records | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/chat_with_llm/reports/TruthRL_Apptainer_Reproduction_Guide.md`, `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/chat_with_llm/reports/TruthRL_Environment_Setup_Guide.md`, `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/chat_with_llm/reports/cluster_migration_manifest_2026-03-12.md` |

## KBP Analysis Support Directory

The top-level `KBP_results` directory is not part of the `TruthRL` or `CPRO` Git repositories.

| Asset | Path | Notes |
| --- | --- | --- |
| KBP analysis scripts | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/KBP_results/scripts` | Should be backed up as source/supporting scripts. |
| KBP config | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/KBP_results/config/default.yaml` | Small and reproducibility-critical. |
| KBP docs | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/KBP_results/docs/kbp_metric_definition.md` | Metric definition. |
| KBP reports/artifacts | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/KBP_results/artifacts` | Contains figures, reports, CSV metrics. Can be archived separately from source. |

## Original Code Snapshot

| Asset | Path | Notes |
| --- | --- | --- |
| Original TruthRL source snapshot | `/home/erichyu/links/projects/def-zshakeri/erichyu/embc/org_codebase/TruthRL-main` | About 6.8M. No `.git` directory. Useful as a baseline source snapshot. |

## Backup Recommendations

1. Keep `TruthRL` and `CPRO` code in GitHub. The current TruthRL backup commit is `40fa386`.
2. Create a separate lightweight supporting-materials backup for `KBP_results`, `paper`, `chat_with_llm`, `org_codebase/TruthRL-main`, `AGENTS.md`, `technical_audit_report_2026-04-21.md`, and `envs/truthrl_apptainer/truthrl.def`.
3. Do not include these patterns in any GitHub repo: `*.sif`, `*.pt`, `*.pth`, `*.ckpt`, `*.bin`, `*.safetensors`, `*.parquet`, `*.npy`, `*.npz`, `*.out`, `*.err`, `data/`, `checkpoints/`, `outputs/`, `logs/`, `runs/`, `wandb/`.
4. Treat `/scratch/erichyu/stageD_d3_hard_ook` as high risk for data loss. If the Stage D D3 model matters, copy the HF export and final checkpoint to durable project storage or an external artifact store.
5. For long-term model preservation, record both the model path and the exact code commit used to produce/evaluate it. For the current code state, use TruthRL commit `40fa386`.

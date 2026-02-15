# Stage B Medical Data Layer

This directory contains Stage B outputs for the TruthRL + Clinical-R1 integration.

## Template Version

- `medical_prompt_v1`
- Source: `data_utils/medical/templates.py`
- If template changes, rebuild all parquet outputs.

## Directory Layout

- `raw/`: raw dataset files (MedQA / MedMCQA)
- `processed_csv/`: normalized csv and split artifacts
- `verl/`: training/eval parquet files consumed by `verl`
- `reports/`: validation and build reports

## Expected Stage B Outputs

- `verl/medqa_grpo_train.parquet`
- `verl/medqa_grpo_test.parquet`
- `verl/medqa_sft_train.parquet`
- `verl/medqa_sft_val.parquet`
- `verl/medmcqa_eval.parquet`
- `reports/stageB_data_report.json`
- `reports/stageB_data_report.md`

## Build (HPC / Apptainer)

Default mode is Apptainer (`USE_APPTAINER=1`).

```bash
bash scripts/medical/build_medical_data.sh
```

Useful overrides:

```bash
RAW_DATA_DIR=/path/to/raw \
REQUIRE_DISTILL=false \
PLACEHOLDER_MODE=simple \
USE_APPTAINER=0 \
bash scripts/medical/build_medical_data.sh
```

Notes:
- `REQUIRE_DISTILL=true` is strict mode (recommended for real SFT data).
- If distilled completions are missing, build fails and writes `reports/missing_distill_completions.txt`.

## Validate

```bash
bash scripts/medical/validate_medical_data.sh
```

Or call python directly:

```bash
python3 data_utils/medical/validate_medical_data.py validate
```

## OOK Label Update Interface

Update `out_of_knowledge` in RL parquet:

```bash
python3 data_utils/medical/validate_medical_data.py update-ook \
  --input-parquet data/medical/verl/medqa_grpo_train.parquet \
  --ook-json /path/to/ook_labels.json \
  --output-parquet data/medical/verl/medqa_grpo_train_with_ook.parquet
```

Accepted `ook_json` formats:
- JSON object: `{ "question_id_1": true, "question_id_2": false }`
- JSON list: `[{"question_id":"...", "out_of_knowledge": true}, ...]`

## Data Contracts

### RL / GRPO schema

Required top-level fields:
- `data_source`
- `prompt`
- `ability`
- `reward_model`
- `extra_info`

Required `reward_model.ground_truth` fields:
- `schema_version`
- `target` (`List[str]`)
- `problem`
- `choices` (`List[str]`)
- `choice_type` (`single|multi`)
- `out_of_knowledge` (`bool`)

Required `extra_info` fields:
- `split`
- `index`
- `question_id`
- `source_dataset`

### SFT schema

Required fields:
- `prompt` (`str`)
- `response` (`str`)

## Common Errors

- `Missing distilled completions detected`: run with `REQUIRE_DISTILL=false` for placeholder smoke mode, or provide distill files.
- `missing RL columns`: parquet schema mismatch; rebuild Stage B outputs.
- `prompt_boxed_contract_violations > 0`: prompt template drifted; regenerate data using current templates.


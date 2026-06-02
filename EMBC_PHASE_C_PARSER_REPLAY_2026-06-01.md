# EMBC Phase C Parser Replay Compatibility Report

Date: 2026-06-01

This report records the Phase C offline parser replay for the EMBC medical
TruthRL / Clinical-R1 work. The replay did not train models, generate new model
responses, or modify historical KBP or Stage D outputs.

## Purpose

Phase B centralized medical answer parsing in:

```text
TruthRL/training/verl/verl/utils/reward_score/medical_answer_parser.py
```

Phase C checks whether that parser refactor changes historical scientific
metrics when the same saved model responses are parsed again.

The main compatibility question is:

```text
Do the Phase B parser policies change historical KBP OOK labels,
question correct_count, difficulty buckets, or Stage D decoded eval metrics?
```

## Replay Tool

The replay tool is:

```text
TruthRL/scripts/medical/parser_replay_metric_compat.py
```

It is source code and should be tracked by Git. Its generated outputs are under
`TruthRL/data/medical/parser_replay/`, which is ignored by Git and should not be
committed.

The full replay output directory is:

```text
TruthRL/data/medical/parser_replay/phaseC_parser_compat_full_20260601
```

Important output files:

```text
summary.md
metadata.json
parse_status_compare.csv
kbp_question_flips.csv
kbp_response_flip_examples.jsonl
stageD_metric_compat.csv
stageD_parse_status_compare.csv
legacy_first_boxed_probe.csv
```

## Inputs

TruthRL code commit used by the replay:

```text
e0344cb refactor(medical): centralize answer parsing policies
```

Canonical KBP fullpass:

```text
TruthRL/data/medical/kbp/runs/kbp_20260319_stagec-c8-8b_medqa_grpo_train_fullpass
```

Stage D decoded eval comparison:

```text
/scratch/erichyu/stageD_d3_hard_ook/eval/phase2_phase3/phase23_20260511_standard_d26_stagec_vs_d3
```

Stage D models replayed:

```text
stagec_base
d3_global_step_300
```

## Parser Policies Checked

KBP replay uses the Phase B KBP policy:

```text
final_answer_last_boxed
```

Stage D decoded eval replay uses the Phase B Stage D policy:

```text
single_boxed_strict
```

The legacy first-boxed behavior is probed separately with:

```text
first_boxed_legacy
```

## Main Findings

### Stage D Eval

Stage D decoded eval is unchanged under the Phase B parser.

```text
stagec_base: 1371 / 1371 rows unchanged
d3_global_step_300: 1371 / 1371 rows unchanged
```

Accuracy, boxed-valid counts, parse-fail counts, and abstain counts match the
historical persisted metrics for both models.

### KBP OOK Labels

KBP OOK labels are unchanged.

```text
processed_responses: 1302784
question_count: 5089
ook_label_changed: 0
abstain_count_changed: 0
```

This means the hard-OOK question set selected from the canonical KBP fullpass is
not invalidated by the Phase B parser refactor.

### KBP correct_count And Difficulty Buckets

KBP question-level `correct_count` changes for a subset of questions.

```text
correct_count_changed questions: 684 / 5089
difficulty_bucket_changed questions: 15 / 5089
negative correct_count deltas: 0
max positive correct_count delta: 15
```

The changes are caused mainly by historical persisted `multi_boxed` rows. In the
historical KBP parsed files, multi-boxed responses were counted as parse
failures. Under the Phase B KBP policy, `final_answer_last_boxed` selects the
last boxed answer, so some of those responses become valid answers.

Historical vs new parse status counts:

```text
choice_out_of_set: historical 26, new 1
invalid_choice: historical 6, new 34
multi_boxed: historical 2070, new 0
no_boxed: historical 389113, new 389113
ok_choice: historical 911569, new 913636
```

## Interpretation

The Phase B parser refactor does not change the major Stage D eval conclusions
and does not change KBP OOK labels.

It does change some KBP `correct_count` and difficulty-bucket values. Therefore
historical KBP metrics should not be silently overwritten by new parser outputs.
Future KBP reports should explicitly record:

```text
parser_module
parser_policy
code_commit
```

For historical comparisons, keep the persisted historical KBP metrics as the
original record and treat the Phase C replay as a compatibility analysis.

## Non-Git Support Script Note

The non-Git script below had previously been updated to use the shared parser
with the `final_answer_last_boxed` policy:

```text
KBP_results/scripts/extract_kbp_metrics.py
```

Because `KBP_results/` is not a Git repository, this script is still not backed
up by the TruthRL commit history. It should be covered by a separate
supporting-materials backup plan or moved into a versioned support area after
review.

## Verification

The replay tool was checked with:

```text
python3 -m py_compile
python3 -m unittest TruthRL.tests.reward_score.test_medical_answer_parser TruthRL.tests.reward_score.test_clinical_medqa_reward_compat
python3 TruthRL/scripts/medical/preflight_medical_runtime.py --schema-mode skip --no-manifest
apptainer strict preflight with --schema-mode strict
```

The preflight still reports the known non-blocking stale-path warning in:

```text
TruthRL/scripts/medical/run_staged_d0_baseline.sh
```

This warning is unrelated to the Phase C replay.

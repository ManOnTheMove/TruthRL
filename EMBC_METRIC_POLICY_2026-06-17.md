# EMBC Metric Policy

Date: 2026-06-17

This policy locks the parser and metric boundary for the EMBC medical
TruthRL work before any new training. It applies to generated KBP reports,
StageD decoded evaluation reports, parser replay reports, and future
handoff summaries.

## Parser Module

The active parser module is:

```text
TruthRL/training/verl/verl/utils/reward_score/medical_answer_parser.py
```

Reports must record the resolved parser module path and the TruthRL code
commit used to generate the report. The active runtime is TruthRL; report
generation must not import parser or reward code from CPRO.

## Parser Policies

The current policy contract is:

```text
Training reward:
  single_boxed_strict

StageD decoded eval:
  single_boxed_strict

KBP collection and regenerated KBP metrics:
  final_answer_last_boxed

Legacy evaluation/evaluate.py compatibility:
  first_boxed_legacy
```

Policy behavior that must stay explicit:

```text
\boxed{Aspirin} is not option A.
\boxed{A. Aspirin} may normalize to A when choice parsing is allowed.
Multiple boxed answers are interpreted only according to the declared policy.
Balanced nested braces inside \boxed{...} must be parsed correctly.
```

## Phase C Compatibility Boundary

Phase C replay showed:

```text
StageD stagec_base rows unchanged: 1371 / 1371
StageD d3_global_step_300 rows unchanged: 1371 / 1371
KBP OOK label changes: 0
KBP abstain_count changes: 0
KBP correct_count changed: 684 / 5089 questions
KBP difficulty_bucket changed: 15 / 5089 questions
```

Interpretation:

```text
StageD decoded eval conclusions remain usable under the shared parser.
KBP hard-OOK labels remain usable under the shared parser.
KBP correct_count and difficulty buckets are parser-policy sensitive.
```

## Historical Metric Rule

Historical KBP fields are the persisted scientific record for their original
parser behavior. Do not silently overwrite:

```text
correct_count
difficulty_bucket
ook labels derived from historical correct_count
abstain_count
```

If a report needs regenerated values under the shared parser, use a separate
namespace or clearly labeled columns, for example:

```text
historical_correct_count
historical_difficulty_bucket
new_parser_correct_count
new_parser_difficulty_bucket
parser_policy=final_answer_last_boxed
metric_namespace=new_parser_final_answer_last_boxed
```

The same table may contain historical and regenerated values only when the
column names and metadata make the boundary explicit.

## Required Report Metadata

Every new KBP, StageD, or parser replay report must include:

```text
generated_at_utc
code_commit
parser_module
parser_policy or parser_policies
metric_namespace, when regenerated parser metrics are emitted
input_paths
output_dir
```

For KBP reports, `input_paths` should include the KBP run directory and input
parquet. For StageD reports, it should include the eval root and prediction
parquet paths. For replay reports, it should include both KBP and StageD inputs
when both are present.

## Stop Conditions

Stop report generation and review before continuing if:

```text
StageD parsed-answer rows change unexpectedly.
KBP OOK label changes are nonzero.
KBP abstain_count changes are nonzero.
Generated reports overwrite historical KBP correct_count or difficulty_bucket.
Output metadata lacks parser_module, parser_policy, code_commit, input_paths,
or output_dir.
TruthRL imports resolve to CPRO/verl.
```

This policy does not authorize new training. It only defines reproducible
metric/reporting behavior for Phase D metric-lock work.

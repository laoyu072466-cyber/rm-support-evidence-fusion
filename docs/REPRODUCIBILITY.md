# Reproducibility Guide

## Repository portability

All scripts derive the repository root from their own location:

```python
ROOT = Path(__file__).resolve().parents[1]
```

The repository may therefore be cloned to any directory. No script
should require `/root/autodl-tmp/rm_traj_project`.

## Artifact policy

Git contains code, configurations, compact manifests, and hashes.
It intentionally excludes large or label-sensitive artifacts.

| Artifact | Expected local path | In Git |
|---|---|---:|
| Generator checkpoints | `models/generator/` | No |
| Reward checkpoints | `models/reward/` | No |
| Raw/external datasets | `data/external/` | No |
| Processed candidates | `data/processed/` | No |
| Cached reward scores | `data/cache/` | No |
| Generated outputs | `outputs/` | No |
| Compact manifests | `data/manifests/` | Yes |
| Frozen configurations | `configs/` | Yes |

Use the file paths and SHA256 values in each manifest to verify
restored artifacts.

## Environment

Reference environment:

```text
Python 3.12.3
PyTorch 2.8.0+cu128
CUDA 12.8
Transformers 4.51.3
```

Install a CUDA-compatible PyTorch build, then:

```bash
python -m pip install -r requirements.txt
```

Verify every Python script without loading models:

```bash
python -m compileall -q scripts
```

## Mathematics pipeline

Required processed files:

```text
data/processed/prototype_v2/gsm_train.jsonl
data/processed/prototype_v2/gsm_pilot_validation.jsonl
data/processed/prototype_v2/gsm_id_test_mixed.jsonl
data/processed/prototype_v2/math_train.jsonl
data/processed/prototype_v2/math_pilot_validation.jsonl
data/processed/prototype_v2/math_id_test_mixed.jsonl
data/processed/prototype_v2/svamp_ood_mixed.jsonl
```

Required cached score arrays follow this convention:

```text
data/cache/reward_scores_full_v1/<reward-model>/<split>.scores_f32.npy
```

Primary evaluation entry points:

```bash
python scripts/eval_answer_cluster_rm_support_full.py
python scripts/eval_answer_cluster_generator_full.py
python scripts/bootstrap_answer_cluster_final.py

python scripts/eval_multi_reward_reproduction.py \
  --model qwen3_4b

python scripts/eval_multi_reward_reproduction.py \
  --model llama_8b_v2

python scripts/eval_multi_reward_reproduction.py \
  --model armorm_8b

python scripts/eval_internlm2_reward_reproduction.py
```

Controlled degradation:

```bash
python scripts/run_controlled_reward_degradation_v1.py
```

## ARC pipeline

Recommended stage order:

```text
1. prepare_arc_challenge_v1.py
2. generate_arc_challenge_qwen2_7b_full_k16.py
3. audit_arc_challenge_qwen2_7b_full_k16_primary.py
4. recover_arc_challenge_qwen2_7b_full_k16.py
5. score_arc_candidates_reward.py
6. score_arc_candidates_internlm2.py
7. audit_arc_multi_reward_scores_v1.py
8. configure_arc_multi_reward_train_pilot_v2.py
9. eval_arc_multi_reward_frozen_test_v1.py
```

Use `--preflight-only` where supported before loading any model or
revealing any label file.

Protocol invariants:

- candidate generation does not read labels;
- recovery does not read labels;
- reward scoring does not read labels;
- Train and Pilot labels may configure the method;
- sealed Test labels are read only for one-shot final evaluation;
- no retraining or retuning follows Test access.

## Frozen tags

Important checkpoints include:

```text
answer-cluster-final-v1
multi-reward-qwen3-4b-reproduction-v1
multi-reward-llama-8b-reproduction-v1
multi-reward-armorm-reproduction-v1
multi-reward-internlm2-1p8b-reproduction-v1
arc-challenge-qwen2-7b-full-k16-recovered-v1
arc-multi-reward-train-pilot-config-v2
arc-multi-reward-frozen-test-v1
reliability-aware-math-fixed-risk-gate-train-pilot-v3
controlled-reward-degradation-results-v1
```

## Expected high-level results

```text
Five-RM mathematical Delta Top1 range:
  +1.75pp to +11.78pp

Controlled degradation:
  sigma 0 -> +1.75pp
  sigma 4 -> +18.56pp

ARC three-RM macro:
  Raw 86.43% -> Fixed gate 88.74%
```

Small BF16 last-digit changes may occur during fresh model scoring,
but they should not alter the qualitative conclusions.

## Data and license responsibilities

Dataset and model licenses are not transferred by this repository.
Users must obtain each external dataset and checkpoint from its
official source and comply with its terms.

Before making this repository public:

1. select a code license;
2. add formal dataset/model attribution;
3. confirm that no external or processed data is present;
4. rerun secret and large-object history scans.

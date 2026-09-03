# RM-Support: Answer-Cluster Evidence Fusion

Code, frozen configurations, compact result manifests, and
experiment history for improving reward-model selection with
answer-cluster evidence.

> Repository status: sanitized private research release.
> Model weights, generated candidates, raw datasets, cached scores,
> processed datasets, outputs, and sealed labels are intentionally
> excluded.

## Overview

A pointwise reward model scores each candidate independently.
RM-Support additionally groups candidates that reach the same
normalized final answer and uses cluster-level support to repair
unstable candidate rankings.

The final pipeline has three layers:

1. score individual candidates with a base reward model;
2. aggregate candidates into normalized answer clusters;
3. combine reward evidence and cluster support, then use a
   Train/Pilot-selected fixed gate to decide whether to replace the
   raw reward-model choice.

The learned reliability-gate experiments are negative ablations,
not the main method.

## Main findings

### Mathematical reasoning

Results are macro-averaged equally across GSM8K-ID, SVAMP-OOD, and
MATH-ID. Test labels were not used for fitting or configuration
selection.

| Base reward model | Raw Top1 | RM-Support | Delta |
|---|---:|---:|---:|
| Skywork Qwen3-1.7B | 80.81% | 83.04% | +2.23pp |
| Skywork Qwen3-4B | 81.80% | 83.54% | +1.75pp |
| Skywork Llama-3.1-8B | 73.60% | 79.38% | +5.78pp |
| ArmoRM Llama3-8B | 75.74% | 81.15% | +5.41pp |
| InternLM2-1.8B | 65.02% | 76.80% | +11.78pp |

Across the five heterogeneous reward models, raw Top1 and Top1
improvement have Pearson correlation -0.995 and Spearman
correlation -1.000. This is a descriptive cross-model trend.

### Controlled reward degradation

The Skywork Qwen3-4B scores were degraded with deterministic
within-question Gaussian noise while keeping candidate responses
fixed.

| Noise sigma | Raw Top1 | RM-Support | Delta |
|---:|---:|---:|---:|
| 0 | 81.80% | 83.54% | +1.75pp |
| 0.25 | 80.69% | 82.74% | +2.05pp |
| 0.5 | 78.06% | 81.33% | +3.28pp |
| 1 | 72.43% | 79.08% | +6.65pp |
| 2 | 64.20% | 76.68% | +12.48pp |
| 4 | 56.92% | 75.48% | +18.56pp |

Across all 26 controlled variants, Pearson correlation between
raw strength and improvement is -0.993. Raw accuracy decreases
strictly and fusion gain increases strictly across the six noise
levels. This is a post-hoc controlled mechanism study rather than
a new confirmatory benchmark.

### ARC-Challenge

ARC uses the complete official split and Qwen2-7B with K=16
candidates. Generation, format recovery, and reward scoring were
completed before the sealed Test labels were used once for final
evaluation.

| Reward model | Raw | Majority | Fixed gate | Delta |
|---|---:|---:|---:|---:|
| Skywork Qwen3-1.7B | 88.31% | 88.99% | 88.40% | +0.09pp |
| InternLM2-1.8B | 82.00% | 88.82% | 88.05% | +6.06pp |
| ArmoRM Llama3-8B | 88.99% | 89.16% | 89.76% | +0.77pp |
| **Three-RM macro** | **86.43%** | **88.99%** | **88.74%** | **+2.30pp** |

ARC supports weak-reward-model repair and pairwise-ranking
improvement. It does not support a claim that the fixed gate
universally outperforms majority voting.

### Learned reliability gates

Three learned reliability/risk-gate variants were developed only
on GSM8K and MATH Train/Pilot:

- v1 underperformed the fixed gate by 3.26 percentage points;
- v2 matched the ungated proposal and did not beat the fixed gate;
- v3 exactly reproduced the fixed-gate result.

These experiments are retained as a negative ablation showing that
the fixed gate already captures the useful reliability signal
available from the current features.

## Repository contents

```text
configs/          Frozen experiment configurations
data/manifests/   Compact metrics, hashes, and protocol records
docs/             Results, reproduction, and legacy documentation
release_metadata/ Sanitization provenance and commit mapping
scripts/          Data, generation, scoring, evaluation, and audits
```

The following artifact classes are excluded from Git:

```text
models/
outputs/
data/cache/
data/external/
data/processed/
```

## Environment

The frozen reference environment used:

```text
Python       3.12.3
PyTorch      2.8.0+cu128
CUDA         12.8
Transformers 4.51.3
GPU          NVIDIA GeForce RTX 5090
```

Install a PyTorch build appropriate for the local CUDA runtime,
then install the remaining pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

Exact reference versions and key file hashes are recorded in
`configs/environment_frozen.json`.

## Reproduction levels

### Results-only inspection

No models or datasets are required. Inspect the compact manifests:

```text
data/manifests/answer_cluster_gate_mechanism_v1.json
data/manifests/multi_reward_reproduction_qwen3_4b_v1.json
data/manifests/multi_reward_reproduction_llama_8b_v2_v1.json
data/manifests/multi_reward_reproduction_armorm_8b_v1.json
data/manifests/multi_reward_reproduction_internlm2_1p8b_v1.json
data/manifests/arc_multi_reward_frozen_test_v1.json
data/manifests/controlled_reward_degradation_results_v1.json
```

### Evaluation from cached artifacts

Restore the processed candidate JSONL files and cached reward-score
arrays described in `docs/REPRODUCIBILITY.md`, then run the
evaluation entry points.

### End-to-end reproduction

Restore the generator and reward-model checkpoints, prepare the
datasets, generate candidates, recover formatting failures, score
candidates, and finally run frozen evaluation.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Primary entry points

```text
scripts/eval_answer_cluster_rm_support_full.py
scripts/bootstrap_answer_cluster_final.py
scripts/eval_multi_reward_reproduction.py
scripts/eval_internlm2_reward_reproduction.py
scripts/run_controlled_reward_degradation_v1.py

scripts/prepare_arc_challenge_v1.py
scripts/generate_arc_challenge_qwen2_7b_full_k16.py
scripts/recover_arc_challenge_qwen2_7b_full_k16.py
scripts/score_arc_candidates_reward.py
scripts/configure_arc_multi_reward_train_pilot_v2.py
scripts/eval_arc_multi_reward_frozen_test_v1.py
```

## Frozen protocol rules

- Fit on Train.
- Select regularization, fusion strength, and gate threshold on
  Pilot.
- Do not use Test labels for training or configuration selection.
- Freeze candidates and reward scores before label reveal.
- Treat new explorations as new versions; never overwrite frozen
  manifests.

## Documentation

- [Results summary](docs/RESULTS.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Original single-RM README](docs/LEGACY_ANSWER_CLUSTER_README.md)
- [Release sanitization record](release_metadata/SANITIZATION.md)

## Important interpretation boundary

The strongest supported conclusion is that answer-cluster evidence
is most useful when the base reward model is weak or noisy.
The current evidence does not establish universal superiority over
majority voting, universal gains for strong reward models, or an
advantage from a learned reliability gate.

## License

No open-source license has been selected yet. The repository should
remain private until the code license and third-party model/data
obligations are reviewed.

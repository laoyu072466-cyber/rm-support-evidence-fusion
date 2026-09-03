# Frozen Experimental Results

All Top1 changes below are absolute percentage-point changes.

## Mathematical reasoning

Three-dataset equal-weight macro average over:

- GSM8K-ID: 1,142 questions
- SVAMP-OOD: 883 questions
- MATH-ID: 340 questions

Total question records: 2,365.

| Reward model | Raw Top1 | Method Top1 | Delta Top1 | Delta Pair | Damage | Correction |
|---|---:|---:|---:|---:|---:|---:|
| Skywork Qwen3-1.7B | 80.81% | 83.04% | +2.23pp | +7.89pp | 2.17% | not retained in handoff |
| Skywork Qwen3-4B | 81.80% | 83.54% | +1.75pp | +6.59pp | 2.37% | 18.99% |
| Skywork Llama-3.1-8B | 73.60% | 79.38% | +5.78pp | +10.09pp | 3.57% | 31.86% |
| ArmoRM Llama3-8B | 75.74% | 81.15% | +5.41pp | +8.63pp | 3.25% | 28.59% |
| InternLM2-1.8B | 65.02% | 76.80% | +11.78pp | +15.20pp | 2.96% | 38.21% |

Paired-bootstrap 95% confidence intervals for Delta Top1:

| Reward model | 95% CI |
|---|---:|
| Skywork Qwen3-4B | [0.53pp, 2.97pp] |
| Skywork Llama-3.1-8B | [4.22pp, 7.36pp] |
| ArmoRM Llama3-8B | [3.81pp, 7.03pp] |
| InternLM2-1.8B | [9.96pp, 13.62pp] |

The supplied handoff does not contain a bootstrap interval for the
Qwen3-1.7B reference; no interval should be fabricated.

## Reward strength trend

Across five heterogeneous reward models:

| Gain measure | Pearson | Spearman |
|---|---:|---:|
| Delta Top1 | -0.995 | -1.000 |
| Delta Pair | -0.978 | -1.000 |
| Delta Best@4 | -0.990 | -1.000 |
| Delta Best@8 | -0.996 | -1.000 |

This comparison is descriptive because model family, size, and
training data differ.

## Controlled degradation

Base model: Skywork Qwen3-4B.

| Sigma | Replicates | Raw Top1 | Method Top1 | Delta Top1 | Delta Pair |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 81.80% | 83.54% | +1.75pp | +6.59pp |
| 0.25 | 5 | 80.69% | 82.74% | +2.05pp | +7.12pp |
| 0.5 | 5 | 78.06% | 81.33% | +3.28pp | +8.60pp |
| 1 | 5 | 72.43% | 79.08% | +6.65pp | +12.87pp |
| 2 | 5 | 64.20% | 76.68% | +12.48pp | +20.29pp |
| 4 | 5 | 56.92% | 75.48% | +18.56pp | +26.66pp |

Zero noise exactly reproduces the original Qwen3-4B experiment.
Across all 26 variants, raw strength versus Delta Top1 has Pearson
-0.993 and Spearman -0.988.

## ARC-Challenge frozen Test

- Questions: 1,172
- Candidates: 18,752
- Parsed candidates: 18,716
- Candidate parse rate: 99.81%

| Reward model | Raw | Majority | Ungated | Fixed gate | Delta Top1 | Delta Pair |
|---|---:|---:|---:|---:|---:|---:|
| Skywork Qwen3-1.7B | 88.31% | 88.99% | 88.48% | 88.40% | +0.09pp | +10.70pp |
| InternLM2-1.8B | 82.00% | 88.82% | 88.57% | 88.05% | +6.06pp | +24.85pp |
| ArmoRM Llama3-8B | 88.99% | 89.16% | 89.76% | 89.76% | +0.77pp | +5.53pp |

Three-RM macro averages:

```text
Raw:        86.43%
Majority:   88.99%
Ungated:    88.94%
Fixed gate: 88.74%
Gate gain:  +2.30pp
```

Fixed-gate Top1 confidence intervals:

```text
Skywork Qwen3-1.7B: [-0.26pp, 0.43pp]
InternLM2-1.8B:      [4.35pp, 7.76pp]
ArmoRM Llama3-8B:    [0.26pp, 1.37pp]
```

## Reliability-gate ablation

| Version | Learned gate Top1 | Fixed-gate Top1 | Difference |
|---|---:|---:|---:|
| v1 | 90.84% | 94.10% | -3.26pp |
| v2 | 94.10% | 94.10% | 0.00pp |
| v3 | 94.10% | 94.10% | 0.00pp |

v3 exactly reproduces the fixed gate. No Test evaluation was
performed for this branch.

## Canonical manifests

```text
data/manifests/reward_strength_gain_trend_v1.json
data/manifests/controlled_reward_degradation_results_v1.json
data/manifests/arc_multi_reward_frozen_test_v1.json
data/manifests/reliability_aware_math_fixed_risk_gate_train_pilot_v3.json
```

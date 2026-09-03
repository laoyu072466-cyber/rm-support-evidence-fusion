from pathlib import Path
from datetime import datetime, timezone
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT / "data/manifests/"
    "reward_strength_gain_trend_v1.json"
)

MODELS = [
    {
        "name": "Skywork-Reward-V2-Qwen3-1.7B",
        "family": "Skywork-Qwen",
        "parameters_b": 1.7,
        "raw_top1": 0.8080966116188856,
        "method_top1": 0.830445295555394,
        "top1_delta": 0.02234868393650844,
        "pair_delta": 0.078859,
        "best_at_4_delta": 0.057225,
        "best_at_8_delta": 0.046716,
        "source": "answer-cluster-gate-mechanism-v1",
    },
    {
        "name": "Skywork-Reward-V2-Qwen3-4B",
        "family": "Skywork-Qwen",
        "parameters_b": 4.0,
        "raw_top1": 0.8179606272776886,
        "method_top1": 0.8354196861978197,
        "top1_delta": 0.017459058920131075,
        "pair_delta": 0.06587205395880376,
        "best_at_4_delta": 0.04891891375807899,
        "best_at_8_delta": 0.04015417518924323,
        "source": "multi-reward-qwen3-4b-reproduction-v1",
    },
    {
        "name": "Skywork-Reward-V2-Llama-3.1-8B",
        "family": "Skywork-Llama",
        "parameters_b": 8.0,
        "raw_top1": 0.7360033873583189,
        "method_top1": 0.7938109037221702,
        "top1_delta": 0.057807516363851175,
        "pair_delta": 0.10088646827462172,
        "best_at_4_delta": 0.0770366685419377,
        "best_at_8_delta": 0.07593753808032204,
        "source": "multi-reward-llama-8b-reproduction-v1",
    },
    {
        "name": "ArmoRM-Llama3-8B-v0.1",
        "family": "ArmoRM",
        "parameters_b": 8.0,
        "raw_top1": 0.7573707146769154,
        "method_top1": 0.8115191747106802,
        "top1_delta": 0.05414846003376469,
        "pair_delta": 0.08631744840247997,
        "best_at_4_delta": 0.06727019359112334,
        "best_at_8_delta": 0.06720169842778495,
        "source": "multi-reward-armorm-reproduction-v1",
    },
    {
        "name": "InternLM2-1.8B-Reward",
        "family": "InternLM",
        "parameters_b": 1.8,
        "raw_top1": 0.6501680184482732,
        "method_top1": 0.7679811784648448,
        "top1_delta": 0.1178131600165716,
        "pair_delta": 0.15199931217105067,
        "best_at_4_delta": 0.11387623212456659,
        "best_at_8_delta": 0.12488338053594839,
        "source": "multi-reward-internlm2-1p8b-reproduction-v1",
    },
]

METRICS = [
    "top1_delta",
    "pair_delta",
    "best_at_4_delta",
    "best_at_8_delta",
]

raw = np.asarray([
    item["raw_top1"]
    for item in MODELS
], dtype=np.float64)

correlations = {}

for metric in METRICS:
    gain = np.asarray([
        item[metric]
        for item in MODELS
    ], dtype=np.float64)

    pearson = pearsonr(raw, gain)
    spearman = spearmanr(raw, gain)
    slope, intercept = np.polyfit(raw, gain, 1)

    leave_one_out = []

    for excluded in range(len(MODELS)):
        keep = np.arange(len(MODELS)) != excluded
        loo_pearson = pearsonr(
            raw[keep],
            gain[keep],
        )
        loo_spearman = spearmanr(
            raw[keep],
            gain[keep],
        )

        leave_one_out.append({
            "excluded_model":
                MODELS[excluded]["name"],
            "pearson": float(
                loo_pearson.statistic
            ),
            "spearman": float(
                loo_spearman.statistic
            ),
        })

    order = np.argsort(raw)
    ordered_gain = gain[order]

    correlations[metric] = {
        "pearson": float(pearson.statistic),
        "spearman": float(
            spearman.statistic
        ),
        "linear_slope": float(slope),
        "linear_intercept": float(intercept),
        "strict_inverse_monotonic": bool(
            np.all(
                ordered_gain[:-1]
                > ordered_gain[1:]
            )
        ),
        "leave_one_out": leave_one_out,
        "leave_one_out_pearson_range": [
            float(min(
                item["pearson"]
                for item in leave_one_out
            )),
            float(max(
                item["pearson"]
                for item in leave_one_out
            )),
        ],
    }

internlm = MODELS[-1]
reference = MODELS[0]

initial_gap = (
    reference["raw_top1"]
    - internlm["raw_top1"]
)
remaining_gap = (
    reference["method_top1"]
    - internlm["method_top1"]
)

result = {
    "version": "reward_strength_gain_trend_v1",
    "created_at": datetime.now(
        timezone.utc
    ).isoformat(),
    "datasets": [
        "GSM8K_ID",
        "SVAMP_OOD",
        "MATH_ID",
    ],
    "aggregation": (
        "unweighted macro average across "
        "the same three frozen test datasets"
    ),
    "models": MODELS,
    "correlations": correlations,
    "internlm_gap_recovery": {
        "reference_model": reference["name"],
        "initial_raw_top1_gap": initial_gap,
        "remaining_method_top1_gap":
            remaining_gap,
        "gap_recovered": (
            initial_gap - remaining_gap
        ),
        "fraction_gap_recovered": (
            (initial_gap - remaining_gap)
            / initial_gap
        ),
    },
    "interpretation": {
        "supported_claim": (
            "Across five frozen reward models, "
            "weaker raw ranking is associated "
            "with larger fusion gains."
        ),
        "causal_claim_allowed": False,
        "reason": (
            "The five real reward models differ "
            "in architecture, scale, and training "
            "data; controlled score degradation "
            "is still required."
        ),
    },
    "labels_accessed_by_this_analysis": False,
}

temporary = OUTPUT.with_suffix(
    OUTPUT.suffix + f".tmp.{os.getpid()}"
)
temporary.write_text(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
temporary.replace(OUTPUT)

print("===== 奖励模型强弱—融合增益趋势 =====")
print(json.dumps(
    correlations,
    ensure_ascii=False,
    indent=2,
))
print()
print("InternLM 补回基础差距比例：")
print(
    result["internlm_gap_recovery"][
        "fraction_gap_recovered"
    ]
)
print("结果：", OUTPUT)

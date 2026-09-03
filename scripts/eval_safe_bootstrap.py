from pathlib import Path
from collections import defaultdict
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safe_trajectory_inference import (
    safe_ensemble_score,
)


CACHE = (
    ROOT / "data/cache/trajectory_features_v1/"
    "Skywork-Reward-V2-Qwen3-1.7B/layer_28"
)
PREDICTIONS = ROOT / "outputs/final_predictions"

DATASETS = {
    "GSM8K_ID": "gsm_id_test",
    "MATH_ID": "math_id_test",
    "SVAMP_OOD": "svamp_ood",
}

SEEDS = [
    20260829,
    20260830,
    20260831,
]
BETA = 0.1
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20260901


class EvaluationData:
    def __init__(self, prefix):
        self.prefix = prefix
        self.scores = np.load(
            CACHE / f"{prefix}.scores_f32.npy"
        ).astype(np.float64)
        self.labels = np.load(
            CACHE / f"{prefix}.labels_i8.npy"
        ).astype(np.int8)

        metadata_path = (
            CACHE / f"{prefix}.metadata.jsonl"
        )
        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        groups = defaultdict(list)
        for index, item in enumerate(self.metadata):
            groups[item["question_uid"]].append(index)

        self.groups = dict(groups)
        self.question_uids = list(self.groups)


def question_effects(data, safe_scores):
    raw_top1 = []
    safe_top1 = []
    raw_pair = []
    safe_pair = []

    correction_count = 0
    damage_count = 0
    raw_wrong_count = 0
    raw_correct_count = 0

    for uid in data.question_uids:
        indices = np.asarray(
            data.groups[uid],
            dtype=np.int64,
        )
        labels = data.labels[indices]
        raw = data.scores[indices]
        safe = safe_scores[indices]

        raw_ok = float(
            labels[int(np.argmax(raw))] == 1
        )
        safe_ok = float(
            labels[int(np.argmax(safe))] == 1
        )

        raw_top1.append(raw_ok)
        safe_top1.append(safe_ok)

        positive = labels == 1
        negative = labels == 0

        raw_pair.append(float(
            (
                raw[positive, None]
                > raw[None, negative]
            ).mean()
        ))
        safe_pair.append(float(
            (
                safe[positive, None]
                > safe[None, negative]
            ).mean()
        ))

        if raw_ok:
            raw_correct_count += 1
            if not safe_ok:
                damage_count += 1
        else:
            raw_wrong_count += 1
            if safe_ok:
                correction_count += 1

    raw_top1 = np.asarray(raw_top1)
    safe_top1 = np.asarray(safe_top1)
    raw_pair = np.asarray(raw_pair)
    safe_pair = np.asarray(safe_pair)

    return {
        "raw_top1": raw_top1,
        "safe_top1": safe_top1,
        "raw_pair": raw_pair,
        "safe_pair": safe_pair,
        "top1_effect": safe_top1 - raw_top1,
        "pair_effect": safe_pair - raw_pair,
        "correction_rate": (
            correction_count / raw_wrong_count
            if raw_wrong_count
            else 0.0
        ),
        "damage_rate": (
            damage_count / raw_correct_count
            if raw_correct_count
            else 0.0
        ),
    }


def bootstrap_means(values, rng):
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    output = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=np.float64,
    )

    batch_size = 500

    for start in range(
        0,
        BOOTSTRAP_SAMPLES,
        batch_size,
    ):
        end = min(
            start + batch_size,
            BOOTSTRAP_SAMPLES,
        )
        indices = rng.integers(
            0,
            count,
            size=(end - start, count),
        )
        output[start:end] = values[indices].mean(
            axis=1
        )

    return output


def effect_summary(point, samples):
    lower, upper = np.quantile(
        samples,
        [0.025, 0.975],
    )

    probability_nonpositive = (
        np.sum(samples <= 0) + 1
    ) / (len(samples) + 1)
    probability_nonnegative = (
        np.sum(samples >= 0) + 1
    ) / (len(samples) + 1)

    p_value = min(
        1.0,
        2.0 * min(
            probability_nonpositive,
            probability_nonnegative,
        ),
    )

    return {
        "point_estimate": float(point),
        "ci_95": [
            float(lower),
            float(upper),
        ],
        "exploratory_two_sided_p": float(p_value),
        "ci_excludes_zero": bool(
            lower > 0 or upper < 0
        ),
    }


def main():
    normalization_manifest = json.loads(
        (
            ROOT / "data/manifests/"
            "trajectory_normalization_stats.json"
        ).read_text(encoding="utf-8")
    )
    normalization = normalization_manifest["modes"][
        "joint_dataset_balanced"
    ]

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    results = {}
    bootstrap_cache = {}

    for dataset_name, prefix in DATASETS.items():
        data = EvaluationData(prefix)

        normalized_original = (
            data.scores
            - normalization["reward_mean"]
        ) / normalization["reward_std"]

        learned_scores = [
            np.load(
                PREDICTIONS
                / (
                    f"{prefix}.seed_{seed}."
                    "scores_f32.npy"
                )
            ).astype(np.float64)
            for seed in SEEDS
        ]

        safe_scores = safe_ensemble_score(
            normalized_original,
            learned_scores,
            beta=BETA,
            ensemble_method="mean",
        )

        np.save(
            PREDICTIONS
            / (
                f"{prefix}.safe_ensemble_"
                "beta0p1.scores_f32.npy"
            ),
            safe_scores.astype(np.float32),
        )

        effects = question_effects(
            data,
            safe_scores,
        )

        top1_samples = bootstrap_means(
            effects["top1_effect"],
            rng,
        )
        pair_samples = bootstrap_means(
            effects["pair_effect"],
            rng,
        )

        bootstrap_cache[dataset_name] = {
            "top1": top1_samples,
            "pair": pair_samples,
        }

        results[dataset_name] = {
            "questions": len(data.question_uids),
            "candidates": len(data.scores),
            "raw_top1": float(
                effects["raw_top1"].mean()
            ),
            "safe_top1": float(
                effects["safe_top1"].mean()
            ),
            "raw_pair": float(
                effects["raw_pair"].mean()
            ),
            "safe_pair": float(
                effects["safe_pair"].mean()
            ),
            "top1_effect": effect_summary(
                effects["top1_effect"].mean(),
                top1_samples,
            ),
            "pair_effect": effect_summary(
                effects["pair_effect"].mean(),
                pair_samples,
            ),
            "correction_rate": effects[
                "correction_rate"
            ],
            "damage_rate": effects["damage_rate"],
        }

    aggregate_sets = {
        "ID_MACRO": [
            "GSM8K_ID",
            "MATH_ID",
        ],
        "ALL_DATASET_MACRO": [
            "GSM8K_ID",
            "MATH_ID",
            "SVAMP_OOD",
        ],
    }

    aggregate = {}

    for name, members in aggregate_sets.items():
        top1_samples = np.mean(
            np.stack([
                bootstrap_cache[item]["top1"]
                for item in members
            ]),
            axis=0,
        )
        pair_samples = np.mean(
            np.stack([
                bootstrap_cache[item]["pair"]
                for item in members
            ]),
            axis=0,
        )

        top1_point = np.mean([
            results[item]["top1_effect"][
                "point_estimate"
            ]
            for item in members
        ])
        pair_point = np.mean([
            results[item]["pair_effect"][
                "point_estimate"
            ]
            for item in members
        ])

        aggregate[name] = {
            "datasets": members,
            "top1_effect": effect_summary(
                top1_point,
                top1_samples,
            ),
            "pair_effect": effect_summary(
                pair_point,
                pair_samples,
            ),
        }

    diagnosis = json.loads(
        (
            ROOT / "outputs/"
            "trajectory_algorithm_diagnosis_current_tests.json"
        ).read_text(encoding="utf-8")
    )

    mean_curve = diagnosis[
        "ensemble_curves"
    ]["mean"]

    ablation = {
        "raw_beta_0": next(
            item for item in mean_curve
            if item["beta"] == 0.0
        ),
        "safe_beta_0p1": next(
            item for item in mean_curve
            if item["beta"] == 0.1
        ),
        "unconstrained_beta_1": next(
            item for item in mean_curve
            if item["beta"] == 1.0
        ),
        "safe_per_seed": {
            seed: next(
                item
                for item in diagnosis[
                    "per_seed_curves"
                ][str(seed)]
                if item["beta"] == 0.1
            )
            for seed in SEEDS
        },
    }

    output = {
        "version": "safe_trajectory_bootstrap_v1",
        "status": "test_informed_exploratory",
        "warning": (
            "beta was selected after observing these "
            "datasets; confidence intervals are diagnostic, "
            "not confirmatory."
        ),
        "method": {
            "beta": BETA,
            "ensemble": "mean_of_three_seeds",
            "bootstrap_unit": "question",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "datasets": results,
        "aggregate": aggregate,
        "ablation": ablation,
    }

    output_path = (
        ROOT / "data/manifests/"
        "trajectory_safe_bootstrap_and_ablation.json"
    )
    output_path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("===== 安全校正 Bootstrap =====")

    for name, item in results.items():
        top = item["top1_effect"]
        pair = item["pair_effect"]

        print(
            f"{name}: "
            f"ΔTop1={top['point_estimate']:+.6f} "
            f"95%CI=[{top['ci_95'][0]:+.6f}, "
            f"{top['ci_95'][1]:+.6f}], "
            f"ΔPair={pair['point_estimate']:+.6f} "
            f"95%CI=[{pair['ci_95'][0]:+.6f}, "
            f"{pair['ci_95'][1]:+.6f}], "
            f"Damage={item['damage_rate']:.6f}"
        )

    print("\n===== 宏平均 Bootstrap =====")
    for name, item in aggregate.items():
        top = item["top1_effect"]
        pair = item["pair_effect"]

        print(
            f"{name}: "
            f"ΔTop1={top['point_estimate']:+.6f} "
            f"95%CI=[{top['ci_95'][0]:+.6f}, "
            f"{top['ci_95'][1]:+.6f}], "
            f"ΔPair={pair['point_estimate']:+.6f} "
            f"95%CI=[{pair['ci_95'][0]:+.6f}, "
            f"{pair['ci_95'][1]:+.6f}]"
        )

    print("\n===== 核心消融 =====")
    for name, item in [
        ("原始 RM", ablation["raw_beta_0"]),
        ("安全校正", ablation["safe_beta_0p1"]),
        ("无约束校正", ablation["unconstrained_beta_1"]),
    ]:
        print(
            f"{name}: "
            f"Macro Top1={item['macro_top1']:.6f}, "
            f"Macro Pair={item['macro_pair']:.6f}, "
            f"Damage={item['macro_damage']:.6f}"
        )

    print("\n结果：", output_path)


if __name__ == "__main__":
    main()

from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import json
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_answer_cluster_generator_full as evaluation
import bootstrap_answer_cluster_final as bootstrap


MODEL_SPECS = {
    "qwen3_4b": {
        "name": "Skywork-Reward-V2-Qwen3-4B",
        "score_root": (
            ROOT
            / "data/cache/reward_scores_full_v1/"
            "Skywork-Reward-V2-Qwen3-4B"
        ),
        "score_manifest": (
            ROOT
            / "data/manifests/"
            "multi_reward_scores_qwen3_4b_v1.json"
        ),
    },
    "llama_8b_v2": {
        "name": (
            "Skywork-Reward-V2-Llama-3.1-8B"
        ),
        "score_root": (
            ROOT
            / "data/cache/reward_scores_full_v1/"
            "Skywork-Reward-V2-Llama-3.1-8B"
        ),
        "score_manifest": (
            ROOT
            / "data/manifests/"
            "multi_reward_scores_llama_8b_v2_v1.json"
        ),
    },

    "armorm_8b": {
        "name": "ArmoRM-Llama3-8B-v0.1",
        "score_root": (
            ROOT
            / "data/cache/reward_scores_full_v1/"
            "ArmoRM-Llama3-8B-v0.1"
        ),
        "score_manifest": (
            ROOT
            / "data/manifests/"
            "multi_reward_scores_armorm_8b_v1.json"
        ),
    },
}

BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 20260903

REFERENCE_1P7B = {
    "base_reward_model": (
        "Skywork-Reward-V2-Qwen3-1.7B"
    ),
    "raw_top1": 0.8080966116188856,
    "method_top1": 0.830445295555394,
    "top1_delta": 0.02234868393650844,
    "raw_pair": 0.830883,
    "method_pair": 0.909742,
    "pair_delta": 0.078859,
    "damage_rate": 0.021721,
    "best_at_4_delta": 0.057225,
    "best_at_8_delta": 0.046716,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values):
    return float(np.mean(
        np.asarray(values, dtype=np.float64)
    ))


def build_macro(points):
    macro = {
        "datasets": len(points),
        "questions": int(sum(
            item["questions"]
            for item in points.values()
        )),
        "raw_top1": mean([
            item["raw_top1"]
            for item in points.values()
        ]),
        "method_top1": mean([
            item["method_top1"]
            for item in points.values()
        ]),
        "top1_delta": mean([
            item["top1_delta"]
            for item in points.values()
        ]),
        "raw_pair": mean([
            item["raw_pair"]
            for item in points.values()
        ]),
        "method_pair": mean([
            item["method_pair"]
            for item in points.values()
        ]),
        "pair_delta": mean([
            item["pair_delta"]
            for item in points.values()
        ]),
        "damage_rate": mean([
            item["damage_rate"]
            for item in points.values()
        ]),
        "correction_rate": mean([
            item["correction_rate"]
            for item in points.values()
        ]),
        "budget": {},
    }

    for k in bootstrap.K_VALUES:
        macro["budget"][f"k{k}"] = {
            "raw_best_at_k": mean([
                item["budget"][f"k{k}"][
                    "raw_best_at_k"
                ]
                for item in points.values()
            ]),
            "method_best_at_k": mean([
                item["budget"][f"k{k}"][
                    "method_best_at_k"
                ]
                for item in points.values()
            ]),
            "best_at_k_delta": mean([
                item["budget"][f"k{k}"][
                    "best_at_k_delta"
                ]
                for item in points.values()
            ]),
            "pass_at_k": mean([
                item["budget"][f"k{k}"][
                    "pass_at_k"
                ]
                for item in points.values()
            ]),
            "coverage": mean([
                item["budget"][f"k{k}"][
                    "coverage"
                ]
                for item in points.values()
            ]),
        }

    return macro


def sampled_dataset_metrics(arrays, indices):
    raw_correct = np.sum(
        arrays["raw_correct"][indices]
    )
    raw_wrong = np.sum(
        arrays["raw_wrong"][indices]
    )

    result = {
        "top1_delta": float(np.mean(
            arrays["top1_delta"][indices]
        )),
        "pair_delta": float(np.mean(
            arrays["pair_delta"][indices]
        )),
        "damage_rate": float(
            np.sum(arrays["damage"][indices])
            / max(raw_correct, 1)
        ),
        "correction_rate": float(
            np.sum(
                arrays["correction"][indices]
            )
            / max(raw_wrong, 1)
        ),
    }

    for k in [4, 8]:
        values = arrays[
            f"best_delta_{k}"
        ][indices]
        valid = np.isfinite(values)

        result[f"best_at_{k}_delta"] = (
            float(np.mean(values[valid]))
            if np.any(valid)
            else np.nan
        )

    return result


def macro_bootstrap(all_arrays):
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    names = list(all_arrays)
    distributions = defaultdict(list)

    for _ in range(BOOTSTRAP_SAMPLES):
        dataset_values = []

        for name in names:
            arrays = all_arrays[name]
            count = len(arrays["top1_delta"])
            indices = rng.integers(
                0,
                count,
                size=count,
            )
            dataset_values.append(
                sampled_dataset_metrics(
                    arrays,
                    indices,
                )
            )

        for key in dataset_values[0]:
            values = [
                item[key]
                for item in dataset_values
                if np.isfinite(item[key])
            ]
            distributions[key].append(
                float(np.mean(values))
            )

    summary = {}

    for key, values in distributions.items():
        values = np.asarray(
            values,
            dtype=np.float64,
        )
        summary[key] = {
            "ci95": [
                float(np.percentile(
                    values,
                    2.5,
                )),
                float(np.percentile(
                    values,
                    97.5,
                )),
            ],
            "probability_positive": float(
                np.mean(values > 0)
            ),
        }

    return summary


def validate_score_files(score_root):
    expected = []

    for domain_spec in (
        evaluation.DOMAINS.values()
    ):
        expected.extend([
            domain_spec["train"][1],
            domain_spec["pilot"][1],
        ])
        expected.extend(
            split_spec[1]
            for split_spec
            in domain_spec["tests"].values()
        )

    missing = []

    for prefix in sorted(set(expected)):
        path = (
            score_root
            / f"{prefix}.scores_f32.npy"
        )
        if not path.exists():
            missing.append(str(path))

    if missing:
        raise RuntimeError(
            "缺少奖励模型分数：\n"
            + "\n".join(missing)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_SPECS),
    )
    args = parser.parse_args()

    started = time.time()
    model_spec = MODEL_SPECS[args.model]

    validate_score_files(
        model_spec["score_root"]
    )

    if not model_spec[
        "score_manifest"
    ].exists():
        raise RuntimeError(
            "缺少评分清单："
            f"{model_spec['score_manifest']}"
        )

    # load_dataset 内部统一从 RM_ROOT 取分数。
    evaluation.RM_ROOT = (
        model_spec["score_root"]
    )

    print(
        "===== 多奖励模型 RM-Support "
        "完整复现 ====="
    )
    print("基础 RM：", model_spec["name"])
    print(
        "RM 分数目录：",
        model_spec["score_root"],
    )
    print(
        "训练与配置：Train + Pilot；"
        "测试标签不参与选择。"
    )

    results = {}
    all_arrays = {}

    for domain_name, domain_spec in (
        evaluation.DOMAINS.items()
    ):
        result, arrays = bootstrap.run_domain(
            domain_name,
            domain_spec,
        )
        results[domain_name] = result
        all_arrays.update(arrays)

    dataset_points = {
        name: bootstrap.point_metrics(arrays)
        for name, arrays in all_arrays.items()
    }

    macro = build_macro(
        dataset_points
    )
    bootstrap_summary = macro_bootstrap(
        all_arrays
    )

    for key, value in bootstrap_summary.items():
        if key in macro:
            value["point"] = macro[key]
        elif key == "best_at_4_delta":
            value["point"] = (
                macro["budget"]["k4"][
                    "best_at_k_delta"
                ]
            )
        elif key == "best_at_8_delta":
            value["point"] = (
                macro["budget"]["k8"][
                    "best_at_k_delta"
                ]
            )

    comparison = {
        "raw_top1_difference": (
            macro["raw_top1"]
            - REFERENCE_1P7B["raw_top1"]
        ),
        "method_top1_difference": (
            macro["method_top1"]
            - REFERENCE_1P7B[
                "method_top1"
            ]
        ),
        "top1_delta_difference": (
            macro["top1_delta"]
            - REFERENCE_1P7B[
                "top1_delta"
            ]
        ),
        "damage_rate_difference": (
            macro["damage_rate"]
            - REFERENCE_1P7B[
                "damage_rate"
            ]
        ),
    }

    output = {
        "version": (
            "multi_reward_reproduction_v1"
        ),
        "base_reward_model": (
            model_spec["name"]
        ),
        "score_root": str(
            model_spec[
                "score_root"
            ].relative_to(ROOT)
        ),
        "score_manifest": str(
            model_spec[
                "score_manifest"
            ].relative_to(ROOT)
        ),
        "score_manifest_sha256": (
            sha256_file(
                model_spec[
                    "score_manifest"
                ]
            )
        ),
        "feature_indices": (
            bootstrap.FEATURE_INDICES
        ),
        "configuration_selection": (
            "independent Train fit and Pilot "
            "selection for each reward model"
        ),
        "test_labels_used_for_selection": False,
        "bootstrap_samples": (
            BOOTSTRAP_SAMPLES
        ),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "results": results,
        "dataset_points": dataset_points,
        "test_macro": macro,
        "paired_bootstrap_macro": (
            bootstrap_summary
        ),
        "reference_qwen3_1p7b": (
            REFERENCE_1P7B
        ),
        "comparison_to_qwen3_1p7b": (
            comparison
        ),
        "elapsed_seconds": round(
            time.time() - started,
            3,
        ),
    }

    output_path = (
        ROOT
        / "data/manifests/"
        / (
            "multi_reward_reproduction_"
            f"{args.model}_v1.json"
        )
    )
    output_path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print("===== 三测试集宏平均 =====")
    print(json.dumps(
        macro,
        ensure_ascii=False,
        indent=2,
    ))

    print()
    print("===== 配对 Bootstrap =====")
    print(json.dumps(
        bootstrap_summary,
        ensure_ascii=False,
        indent=2,
    ))

    print()
    print("===== 相对 1.7B =====")
    print(json.dumps(
        comparison,
        ensure_ascii=False,
        indent=2,
    ))

    print()
    print("结果：", output_path)
    print(
        "耗时秒：",
        output["elapsed_seconds"],
    )


if __name__ == "__main__":
    main()

from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_answer_cluster_holdout_rewards as holdout


SCORE_ROOT = (
    ROOT
    / "data/cache/trajectory_features_v1/"
    "Skywork-Reward-V2-Qwen3-1.7B/"
    "layer_28"
)

SCORE_FILES = {
    "GSM8K_ID": (
        SCORE_ROOT / "gsm_id_test.scores_f32.npy"
    ),
    "MATH_ID": (
        SCORE_ROOT / "math_id_test.scores_f32.npy"
    ),
    "SVAMP_OOD": (
        SCORE_ROOT / "svamp_ood.scores_f32.npy"
    ),
}

MODEL = (
    ROOT
    / "models/reward/"
    "Skywork-Reward-V2-Qwen3-1.7B"
)

OUTPUT = (
    ROOT
    / "data/manifests/"
    "holdout_reward_scoring_reproduction.json"
)


def main():
    (
        rows_by_dataset,
        changed,
        _,
        _,
    ) = holdout.reconstruct_final_selections()

    examples_by_key = {}

    for item in changed:
        dataset = item["dataset"]
        rows = rows_by_dataset[dataset]

        for row_index in [
            item["raw_index"],
            item["new_index"],
        ]:
            key = f"{dataset}:{row_index}"
            row = rows[row_index]

            examples_by_key[key] = {
                "key": key,
                "problem": str(row["problem"]),
                "solution": str(
                    row["solution_text"]
                ),
            }

    examples = list(examples_by_key.values())

    reproduced, peak_gpu = (
        holdout.score_with_judge(
            "Qwen3_1p7B_reproduction",
            MODEL,
            examples,
        )
    )

    frozen_arrays = {
        name: np.asarray(
            np.load(path),
            dtype=np.float64,
        )
        for name, path in SCORE_FILES.items()
    }

    expected = []
    actual = []

    for item in examples:
        dataset, index_text = item["key"].split(
            ":", 1
        )
        index = int(index_text)

        expected.append(
            frozen_arrays[dataset][index]
        )
        actual.append(
            reproduced[item["key"]]
        )

    expected = np.asarray(expected)
    actual = np.asarray(actual)
    difference = actual - expected

    expected_pair_sign = []
    actual_pair_sign = []

    for item in changed:
        dataset = item["dataset"]
        raw_index = item["raw_index"]
        new_index = item["new_index"]

        expected_margin = (
            frozen_arrays[dataset][new_index]
            - frozen_arrays[dataset][raw_index]
        )
        actual_margin = (
            reproduced[
                f"{dataset}:{new_index}"
            ]
            - reproduced[
                f"{dataset}:{raw_index}"
            ]
        )

        expected_pair_sign.append(
            np.sign(expected_margin)
        )
        actual_pair_sign.append(
            np.sign(actual_margin)
        )

    expected_pair_sign = np.asarray(
        expected_pair_sign
    )
    actual_pair_sign = np.asarray(
        actual_pair_sign
    )

    result = {
        "candidates": len(examples),
        "changed_questions": len(changed),
        "pearson_correlation": float(
            np.corrcoef(expected, actual)[0, 1]
        ),
        "mean_absolute_error": float(
            np.mean(np.abs(difference))
        ),
        "median_absolute_error": float(
            np.median(np.abs(difference))
        ),
        "max_absolute_error": float(
            np.max(np.abs(difference))
        ),
        "mean_signed_error": float(
            np.mean(difference)
        ),
        "pair_preference_agreement": float(
            np.mean(
                expected_pair_sign
                == actual_pair_sign
            )
        ),
        "exact_score_match_rate": float(
            np.mean(difference == 0)
        ),
        "peak_gpu_gb": peak_gpu,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("===== 1.7B 冻结分数复现审计 =====")
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )
    print("结果：", OUTPUT)


if __name__ == "__main__":
    main()

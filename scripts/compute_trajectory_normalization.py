from pathlib import Path
import json
import math

import numpy as np


PROJECT = Path("/root/autodl-tmp/rm_traj_project")
CACHE_ROOT = (
    PROJECT
    / "data/cache/trajectory_features_v1"
    / "Skywork-Reward-V2-Qwen3-1.7B"
    / "layer_28"
)
OUTPUT_PATH = (
    PROJECT
    / "data/manifests/trajectory_normalization_stats.json"
)


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_dataset(stem):
    scores = np.load(
        CACHE_ROOT / f"{stem}.scores_f32.npy"
    ).astype(np.float64)

    metadata = read_jsonl(
        CACHE_ROOT / f"{stem}.metadata.jsonl"
    )

    if len(scores) != len(metadata):
        raise RuntimeError(
            f"{stem} score 与 metadata 数量不一致"
        )

    chunk_count = np.asarray(
        [
            row["chunk_count"]
            for row in metadata
        ],
        dtype=np.float64,
    )
    response_length = np.asarray(
        [
            row["response_token_length"]
            for row in metadata
        ],
        dtype=np.float64,
    )

    return {
        "reward": scores,
        "log_t": np.log1p(chunk_count),
        "log_l": np.log1p(response_length),
        "chunk_count": chunk_count,
        "response_token_length": response_length,
    }


def basic_stats(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "count": int(len(values)),
    }


def single_dataset_stats(data):
    result = {
        "reward_mean": float(
            np.mean(data["reward"])
        ),
        "reward_std": float(
            np.std(data["reward"], ddof=0)
        ),
        "log_t_mean": float(
            np.mean(data["log_t"])
        ),
        "log_t_std": float(
            np.std(data["log_t"], ddof=0)
        ),
        "log_l_mean": float(
            np.mean(data["log_l"])
        ),
        "log_l_std": float(
            np.std(data["log_l"], ddof=0)
        ),
    }

    return result


def equal_dataset_mixture_stats(first, second):
    result = {}

    for field, output_prefix in [
        ("reward", "reward"),
        ("log_t", "log_t"),
        ("log_l", "log_l"),
    ]:
        first_values = first[field]
        second_values = second[field]

        mean = (
            float(np.mean(first_values))
            + float(np.mean(second_values))
        ) / 2

        second_moment = (
            float(np.mean(first_values ** 2))
            + float(np.mean(second_values ** 2))
        ) / 2

        variance = max(
            second_moment - mean ** 2,
            0.0,
        )
        std = math.sqrt(variance)

        result[f"{output_prefix}_mean"] = mean
        result[f"{output_prefix}_std"] = std

    return result


gsm = load_dataset("gsm_train")
math_data = load_dataset("math_train")

report = {
    "version": "trajectory_normalization_v1",
    "scope": "train_only",
    "selected_layer": 28,
    "weighting": {
        "gsm_only": "candidate_weighted within GSM8K",
        "math_only": "candidate_weighted within MATH",
        "joint_dataset_balanced": (
            "equal mixture of GSM8K and MATH "
            "candidate distributions"
        ),
    },
    "modes": {
        "gsm_only": single_dataset_stats(gsm),
        "math_only": single_dataset_stats(math_data),
        "joint_dataset_balanced": (
            equal_dataset_mixture_stats(
                gsm,
                math_data,
            )
        ),
    },
    "raw_statistics": {
        "gsm_train": {
            "reward": basic_stats(gsm["reward"]),
            "chunk_count": basic_stats(
                gsm["chunk_count"]
            ),
            "response_token_length": basic_stats(
                gsm["response_token_length"]
            ),
        },
        "math_train": {
            "reward": basic_stats(
                math_data["reward"]
            ),
            "chunk_count": basic_stats(
                math_data["chunk_count"]
            ),
            "response_token_length": basic_stats(
                math_data[
                    "response_token_length"
                ]
            ),
        },
    },
    "pilot_used": False,
    "test_used": False,
    "ood_used": False,
}

for mode, stats in report["modes"].items():
    for key, value in stats.items():
        if key.endswith("_std") and value <= 0:
            raise RuntimeError(
                f"{mode} 的 {key} 非正"
            )

OUTPUT_PATH.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print(json.dumps(
    report,
    ensure_ascii=False,
    indent=2,
))
print()
print("训练集标准化参数已保存：", OUTPUT_PATH)

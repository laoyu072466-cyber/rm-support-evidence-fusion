from pathlib import Path
import json
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_trajectory_head_joint_default import (
    DatasetStore,
    TrajectoryCorrectionHead,
    calculate_metrics,
    predict_store,
    raw_pair_metric,
    DEVICE,
)


DATASETS = {
    "GSM8K_ID": {
        "prefix": "gsm_id_test",
        "display": "GSM8K ID",
    },
    "MATH_ID": {
        "prefix": "math_id_test",
        "display": "MATH ID",
    },
    "SVAMP_OOD": {
        "prefix": "svamp_ood",
        "display": "SVAMP OOD",
    },
}

CHECKPOINTS = {
    20260829: (
        ROOT / "outputs/checkpoints/"
        "trajectory_head_grid_g0p8_bt0p5_cal0p1_"
        "seed20260829.pt"
    ),
    20260830: (
        ROOT / "outputs/checkpoints/"
        "trajectory_head_final_g0p8_bt0p5_cal0p1_"
        "seed20260830.pt"
    ),
    20260831: (
        ROOT / "outputs/checkpoints/"
        "trajectory_head_final_g0p8_bt0p5_cal0p1_"
        "seed20260831.pt"
    ),
}

OUTPUT_PATH = (
    ROOT / "outputs/"
    "final_evaluation_1p7b_three_seeds.json"
)
MANIFEST_PATH = (
    ROOT / "data/manifests/"
    "final_evaluation_1p7b_three_seeds.json"
)
PREDICTION_DIR = ROOT / "outputs/final_predictions"


def mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main():
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    normalization_path = (
        ROOT / "data/manifests/"
        "trajectory_normalization_stats.json"
    )
    normalization_manifest = json.loads(
        normalization_path.read_text(encoding="utf-8")
    )
    normalization = normalization_manifest["modes"][
        "joint_dataset_balanced"
    ]

    stores = {}

    print("===== 加载最终冻结测试特征 =====")
    for name, config in DATASETS.items():
        stores[name] = DatasetStore(
            config["prefix"],
            config["display"],
        )

    baselines = {}

    print("\n===== 原始奖励模型基线 =====")
    for name, store in stores.items():
        baseline = {
            "questions": len(store.question_uids),
            "candidates": len(store.scores),
            "top1": float(
                calculate_metrics(
                    store,
                    store.scores,
                )["raw_top1"]
            ),
            "pair_macro_strict": float(
                raw_pair_metric(store)
            ),
        }
        baselines[name] = baseline

        print(
            f"{name}: "
            f"Top1={baseline['top1']:.6f}, "
            f"Pair={baseline['pair_macro_strict']:.6f}"
        )

    baselines["ID_MACRO"] = {
        "top1": (
            baselines["GSM8K_ID"]["top1"]
            + baselines["MATH_ID"]["top1"]
        ) / 2,
        "pair_macro_strict": (
            baselines["GSM8K_ID"][
                "pair_macro_strict"
            ]
            + baselines["MATH_ID"][
                "pair_macro_strict"
            ]
        ) / 2,
    }

    per_seed = []

    for seed, checkpoint_path in CHECKPOINTS.items():
        print("\n" + "=" * 72)
        print("评测随机种子：", seed)
        print("checkpoint：", checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        model = TrajectoryCorrectionHead(
            normalization
        ).to(DEVICE)
        model.load_state_dict(
            payload["model_state_dict"],
            strict=True,
        )
        model.eval()

        seed_result = {
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "datasets": {},
        }

        for name, store in stores.items():
            corrected_scores = predict_store(
                model,
                store,
            )

            prediction_path = (
                PREDICTION_DIR
                / f"{DATASETS[name]['prefix']}"
                f".seed_{seed}.scores_f32.npy"
            )
            np.save(
                prediction_path,
                corrected_scores.astype(np.float32),
            )

            metrics = calculate_metrics(
                store,
                corrected_scores,
            )
            raw_pair = baselines[name][
                "pair_macro_strict"
            ]

            record = {
                "questions": metrics["questions"],
                "candidates": metrics["candidates"],
                "raw_top1": metrics["raw_top1"],
                "raw_pair_macro_strict": raw_pair,
                "corrected_top1": metrics[
                    "corrected_top1"
                ],
                "corrected_pair_macro_strict": metrics[
                    "corrected_pair_macro_strict"
                ],
                "top1_delta": (
                    metrics["corrected_top1"]
                    - metrics["raw_top1"]
                ),
                "pair_delta": (
                    metrics[
                        "corrected_pair_macro_strict"
                    ]
                    - raw_pair
                ),
                "correction_rate": metrics[
                    "correction_rate"
                ],
                "damage_rate": metrics["damage_rate"],
                "prediction_file": str(prediction_path),
            }
            seed_result["datasets"][name] = record

            print(
                f"{name}: "
                f"Top1 "
                f"{record['raw_top1']:.6f}"
                f" -> {record['corrected_top1']:.6f} "
                f"({record['top1_delta']:+.6f}), "
                f"Pair "
                f"{record['raw_pair_macro_strict']:.6f}"
                f" -> "
                f"{record['corrected_pair_macro_strict']:.6f} "
                f"({record['pair_delta']:+.6f}), "
                f"Damage={record['damage_rate']:.6f}"
            )

        gsm = seed_result["datasets"]["GSM8K_ID"]
        math_result = seed_result["datasets"]["MATH_ID"]

        seed_result["id_macro"] = {
            "corrected_top1": (
                gsm["corrected_top1"]
                + math_result["corrected_top1"]
            ) / 2,
            "corrected_pair_macro_strict": (
                gsm["corrected_pair_macro_strict"]
                + math_result[
                    "corrected_pair_macro_strict"
                ]
            ) / 2,
            "top1_delta": (
                gsm["top1_delta"]
                + math_result["top1_delta"]
            ) / 2,
            "pair_delta": (
                gsm["pair_delta"]
                + math_result["pair_delta"]
            ) / 2,
            "damage_rate": (
                gsm["damage_rate"]
                + math_result["damage_rate"]
            ) / 2,
        }

        print(
            "ID_MACRO: "
            f"Top1={seed_result['id_macro']['corrected_top1']:.6f} "
            f"({seed_result['id_macro']['top1_delta']:+.6f}), "
            f"Pair="
            f"{seed_result['id_macro']['corrected_pair_macro_strict']:.6f} "
            f"({seed_result['id_macro']['pair_delta']:+.6f})"
        )

        per_seed.append(seed_result)

        del model
        torch.cuda.empty_cache()

    aggregate = {}

    for name in DATASETS:
        aggregate[name] = {}

        for metric in [
            "corrected_top1",
            "corrected_pair_macro_strict",
            "top1_delta",
            "pair_delta",
            "correction_rate",
            "damage_rate",
        ]:
            aggregate[name][metric] = mean_std(
                [
                    item["datasets"][name][metric]
                    for item in per_seed
                ]
            )

    aggregate["ID_MACRO"] = {}

    for metric in [
        "corrected_top1",
        "corrected_pair_macro_strict",
        "top1_delta",
        "pair_delta",
        "damage_rate",
    ]:
        aggregate["ID_MACRO"][metric] = mean_std(
            [
                item["id_macro"][metric]
                for item in per_seed
            ]
        )

    result = {
        "version": "final_evaluation_1p7b_v1",
        "model": (
            "Skywork-Reward-V2-Qwen3-1.7B"
        ),
        "selected_layer": 28,
        "frozen_hyperparameters": {
            "gamma": 0.8,
            "lambda_bt": 0.5,
            "lambda_cal": 0.1,
        },
        "seeds": list(CHECKPOINTS),
        "evaluation_scope": {
            "id": ["GSM8K", "MATH"],
            "ood": ["SVAMP"],
        },
        "hyperparameters_frozen_before_test": True,
        "test_labels_used_for_training": False,
        "baselines": baselines,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "total_elapsed_seconds": round(
            time.time() - start_time,
            3,
        ),
        "peak_gpu_gb": round(
            torch.cuda.max_memory_allocated()
            / 1024 ** 3,
            3,
        ),
    }

    serialized = (
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    OUTPUT_PATH.write_text(
        serialized,
        encoding="utf-8",
    )
    MANIFEST_PATH.write_text(
        serialized,
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("===== 三随机种子最终盲评汇总 =====")

    for name in [
        "GSM8K_ID",
        "MATH_ID",
        "ID_MACRO",
        "SVAMP_OOD",
    ]:
        item = aggregate[name]

        print(
            f"{name}: "
            f"Top1="
            f"{item['corrected_top1']['mean']:.6f}"
            f" ± {item['corrected_top1']['std']:.6f}, "
            f"ΔTop1="
            f"{item['top1_delta']['mean']:+.6f}"
            f" ± {item['top1_delta']['std']:.6f}, "
            f"Pair="
            f"{item['corrected_pair_macro_strict']['mean']:.6f}"
            f" ± "
            f"{item['corrected_pair_macro_strict']['std']:.6f}, "
            f"ΔPair="
            f"{item['pair_delta']['mean']:+.6f}"
            f" ± {item['pair_delta']['std']:.6f}"
        )

    print("\n结果文件：", OUTPUT_PATH)
    print("可提交清单：", MANIFEST_PATH)
    print(
        "总耗时：",
        result["total_elapsed_seconds"],
        "秒",
    )
    print("显存峰值：", result["peak_gpu_gb"], "GB")


if __name__ == "__main__":
    main()

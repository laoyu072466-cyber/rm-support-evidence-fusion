from pathlib import Path
import gc
import json
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_sgldsv_rm_residual as residual


SEEDS = [42, 123, 456]
OUTPUT_DIR = (
    ROOT / "outputs/final_predictions/"
    "sgldsv_rm_residual"
)
RESULT_PATH = (
    ROOT / "outputs/"
    "sgldsv_rm_residual_current_tests.json"
)
MANIFEST_PATH = (
    ROOT / "data/manifests/"
    "sgldsv_rm_residual_current_tests.json"
)

FAMILIES = {
    "GSM8K": {
        "validation": "gsm_pilot",
        "tests": {
            "GSM8K_ID": "gsm_id_test",
            "SVAMP_OOD": "svamp_ood",
        },
    },
    "MATH": {
        "validation": "math_pilot",
        "tests": {
            "MATH_ID": "math_id_test",
        },
    },
}


def checkpoint_path(family, seed):
    return (
        ROOT / "outputs/checkpoints/"
        f"sgldsv_rm_residual_"
        f"{family.lower()}_seed{seed}.pt"
    )


def training_result_path(family, seed):
    return (
        ROOT / "outputs/sgldsv_rm_residual/"
        f"sgldsv_rm_residual_"
        f"{family.lower()}_seed{seed}.json"
    )


def load_model(family, seed):
    path = checkpoint_path(family, seed)
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    model = residual.RMPreservingLDSV().to(
        residual.DEVICE
    )
    model.load_state_dict(
        payload["model_state_dict"],
        strict=True,
    )
    model.eval()

    return (
        model,
        float(payload["reward_mean"]),
        float(payload["reward_std"]),
        payload,
    )


def evaluate(
    model,
    cache,
    reward_mean,
    reward_std,
):
    raw_scores = (
        cache.rm_scores - reward_mean
    ) / reward_std

    raw_metrics = residual.ranking_metrics(
        cache,
        raw_scores,
    )
    corrected_scores, corrections = residual.predict(
        model,
        cache,
        reward_mean,
        reward_std,
    )
    corrected_metrics = residual.ranking_metrics(
        cache,
        corrected_scores,
    )
    transitions = residual.transition_metrics(
        cache,
        raw_scores,
        corrected_scores,
    )

    return {
        "raw_rm": raw_metrics,
        "residual": corrected_metrics,
        "delta_top1": (
            corrected_metrics["top1"]
            - raw_metrics["top1"]
        ),
        "delta_pair_macro_strict": (
            corrected_metrics[
                "pair_macro_strict"
            ]
            - raw_metrics[
                "pair_macro_strict"
            ]
        ),
        "transition": transitions,
        "correction_diagnostics": {
            "mean": float(corrections.mean()),
            "std": float(corrections.std()),
            "abs_mean": float(
                np.abs(corrections).mean()
            ),
            "max_abs": float(
                np.abs(corrections).max()
            ),
            "near_cap_rate": float(
                (
                    np.abs(corrections)
                    >= residual.CORRECTION_CAP * 0.98
                ).mean()
            ),
        },
    }, corrected_scores


def reproduce_pilot():
    print("===== 复现残差版本 Pilot =====")

    for family, config in FAMILIES.items():
        for seed in SEEDS:
            model, mean, std, payload = load_model(
                family,
                seed,
            )
            cache = residual.CachedResponses(
                config["validation"]
            )
            current, _ = evaluate(
                model,
                cache,
                mean,
                std,
            )

            saved = json.loads(
                training_result_path(
                    family,
                    seed,
                ).read_text(encoding="utf-8")
            )
            expected = saved["best_pilot"]

            top1_gap = abs(
                current["residual"]["top1"]
                - expected["top1"]
            )
            pair_gap = abs(
                current["residual"][
                    "pair_macro_strict"
                ]
                - expected[
                    "pair_macro_strict"
                ]
            )

            print(
                f"{family} seed={seed}: "
                f"epoch={payload['best_epoch']}, "
                f"Top1 gap={top1_gap:.10f}, "
                f"Pair gap={pair_gap:.10f}"
            )

            del model
            del cache
            gc.collect()
            torch.cuda.empty_cache()

            if (
                top1_gap > 1e-7
                or pair_gap > 1e-6
            ):
                raise RuntimeError(
                    f"{family} seed={seed} "
                    "Pilot 无法复现，停止测试评估"
                )

    print("Pilot 全部复现通过。\n")


def summarize(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "values": [
            float(value)
            for value in values
        ],
    }


def main():
    start = time.time()
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.set_float32_matmul_precision("high")

    reproduce_pilot()

    per_seed = {
        str(seed): {}
        for seed in SEEDS
    }

    for family, config in FAMILIES.items():
        for seed in SEEDS:
            print("\n" + "=" * 72)
            print(
                f"评估 {family} seed={seed}"
            )

            model, mean, std, payload = load_model(
                family,
                seed,
            )

            for display, prefix in config["tests"].items():
                cache = residual.CachedResponses(
                    prefix
                )
                result, predictions = evaluate(
                    model,
                    cache,
                    mean,
                    std,
                )
                result["best_epoch"] = int(
                    payload["best_epoch"]
                )
                per_seed[str(seed)][display] = result

                np.save(
                    OUTPUT_DIR
                    / (
                        f"{display.lower()}_"
                        f"seed{seed}.npy"
                    ),
                    predictions,
                )

                transition = result["transition"]

                print(
                    f"{display}: "
                    f"Top1 "
                    f"{result['raw_rm']['top1']:.6f}"
                    f" -> "
                    f"{result['residual']['top1']:.6f} "
                    f"({result['delta_top1']:+.6f}), "
                    f"Pair "
                    f"{result['raw_rm']['pair_macro_strict']:.6f}"
                    f" -> "
                    f"{result['residual']['pair_macro_strict']:.6f} "
                    f"({result['delta_pair_macro_strict']:+.6f}), "
                    f"Damage="
                    f"{transition['damage_rate']:.6f}, "
                    f"|delta|="
                    f"{result['correction_diagnostics']['abs_mean']:.6f}"
                )

                del cache
                gc.collect()

            del model
            gc.collect()
            torch.cuda.empty_cache()

    datasets = [
        "GSM8K_ID",
        "MATH_ID",
        "SVAMP_OOD",
    ]
    aggregate = {}

    for dataset in datasets:
        aggregate[dataset] = {
            "top1": summarize([
                per_seed[str(seed)][dataset][
                    "residual"
                ]["top1"]
                for seed in SEEDS
            ]),
            "delta_top1": summarize([
                per_seed[str(seed)][dataset][
                    "delta_top1"
                ]
                for seed in SEEDS
            ]),
            "pair_macro_strict": summarize([
                per_seed[str(seed)][dataset][
                    "residual"
                ]["pair_macro_strict"]
                for seed in SEEDS
            ]),
            "delta_pair_macro_strict": summarize([
                per_seed[str(seed)][dataset][
                    "delta_pair_macro_strict"
                ]
                for seed in SEEDS
            ]),
            "damage_rate": summarize([
                per_seed[str(seed)][dataset][
                    "transition"
                ]["damage_rate"]
                for seed in SEEDS
            ]),
            "correction_rate": summarize([
                per_seed[str(seed)][dataset][
                    "transition"
                ]["correction_rate"]
                for seed in SEEDS
            ]),
            "correction_abs_mean": summarize([
                per_seed[str(seed)][dataset][
                    "correction_diagnostics"
                ]["abs_mean"]
                for seed in SEEDS
            ]),
        }

    id_macro_top1 = []
    id_macro_pair = []
    id_macro_delta_top1 = []
    id_macro_delta_pair = []

    all_macro_top1 = []
    all_macro_pair = []
    all_macro_delta_top1 = []
    all_macro_delta_pair = []

    for seed in SEEDS:
        current = per_seed[str(seed)]

        id_macro_top1.append(np.mean([
            current["GSM8K_ID"]["residual"]["top1"],
            current["MATH_ID"]["residual"]["top1"],
        ]))
        id_macro_pair.append(np.mean([
            current["GSM8K_ID"]["residual"][
                "pair_macro_strict"
            ],
            current["MATH_ID"]["residual"][
                "pair_macro_strict"
            ],
        ]))
        id_macro_delta_top1.append(np.mean([
            current["GSM8K_ID"]["delta_top1"],
            current["MATH_ID"]["delta_top1"],
        ]))
        id_macro_delta_pair.append(np.mean([
            current["GSM8K_ID"][
                "delta_pair_macro_strict"
            ],
            current["MATH_ID"][
                "delta_pair_macro_strict"
            ],
        ]))

        all_macro_top1.append(np.mean([
            current[name]["residual"]["top1"]
            for name in datasets
        ]))
        all_macro_pair.append(np.mean([
            current[name]["residual"][
                "pair_macro_strict"
            ]
            for name in datasets
        ]))
        all_macro_delta_top1.append(np.mean([
            current[name]["delta_top1"]
            for name in datasets
        ]))
        all_macro_delta_pair.append(np.mean([
            current[name][
                "delta_pair_macro_strict"
            ]
            for name in datasets
        ]))

    aggregate["ID_MACRO"] = {
        "top1": summarize(id_macro_top1),
        "pair_macro_strict": summarize(
            id_macro_pair
        ),
        "delta_top1": summarize(
            id_macro_delta_top1
        ),
        "delta_pair_macro_strict": summarize(
            id_macro_delta_pair
        ),
    }
    aggregate["ALL_DATASET_MACRO"] = {
        "top1": summarize(all_macro_top1),
        "pair_macro_strict": summarize(
            all_macro_pair
        ),
        "delta_top1": summarize(
            all_macro_delta_top1
        ),
        "delta_pair_macro_strict": summarize(
            all_macro_delta_pair
        ),
    }

    output = {
        "version": (
            "sgldsv_rm_residual_current_tests_v1"
        ),
        "status": (
            "test_informed_exploratory_analysis"
        ),
        "reason": (
            "designed after observing failure of "
            "exact SG-LDSV transfer"
        ),
        "tests_used_for_training": False,
        "tests_used_for_epoch_selection": False,
        "architecture": {
            "endpoint": "frozen_original_RM_score",
            "correction": (
                "0.5*tanh(SG-LDSV_local_score)"
            ),
            "residual_penalty": (
                residual.RESIDUAL_PENALTY
            ),
        },
        "seeds": SEEDS,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "elapsed_seconds": round(
            time.time() - start,
            3,
        ),
        "peak_gpu_gb": round(
            torch.cuda.max_memory_allocated()
            / 1024 ** 3,
            3,
        ),
    }

    text = json.dumps(
        output,
        ensure_ascii=False,
        indent=2,
    ) + "\n"

    RESULT_PATH.write_text(
        text,
        encoding="utf-8",
    )
    MANIFEST_PATH.write_text(
        text,
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("===== RM 残差适配测试汇总 =====")

    for dataset in datasets:
        item = aggregate[dataset]
        print(
            f"{dataset}: "
            f"Top1="
            f"{item['top1']['mean']:.6f}"
            f" ± {item['top1']['std']:.6f}, "
            f"ΔTop1="
            f"{item['delta_top1']['mean']:+.6f}"
            f" ± {item['delta_top1']['std']:.6f}, "
            f"Pair="
            f"{item['pair_macro_strict']['mean']:.6f}"
            f" ± "
            f"{item['pair_macro_strict']['std']:.6f}, "
            f"ΔPair="
            f"{item['delta_pair_macro_strict']['mean']:+.6f}"
            f" ± "
            f"{item['delta_pair_macro_strict']['std']:.6f}"
        )

    print("结果：", RESULT_PATH)
    print("清单：", MANIFEST_PATH)
    print("耗时秒：", output["elapsed_seconds"])


if __name__ == "__main__":
    main()

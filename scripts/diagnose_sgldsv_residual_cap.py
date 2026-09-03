from pathlib import Path
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_sgldsv_rm_residual as residual


SEEDS = [42, 123, 456]
ORIGINAL_CAP = 0.5
CAPS = [
    0.00,
    0.025,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
]

DATASETS = {
    "GSM8K_ID": {
        "family": "GSM8K",
        "prefix": "gsm_id_test",
    },
    "MATH_ID": {
        "family": "MATH",
        "prefix": "math_id_test",
    },
    "SVAMP_OOD": {
        "family": "GSM8K",
        "prefix": "svamp_ood",
    },
}

PREDICTION_DIR = (
    ROOT / "outputs/final_predictions/"
    "sgldsv_rm_residual"
)
OUTPUT_PATH = (
    ROOT / "data/manifests/"
    "sgldsv_residual_cap_diagnosis.json"
)


def checkpoint_path(family, seed):
    return (
        ROOT / "outputs/checkpoints/"
        f"sgldsv_rm_residual_"
        f"{family.lower()}_seed{seed}.pt"
    )


def prediction_path(dataset, seed):
    return (
        PREDICTION_DIR
        / f"{dataset.lower()}_seed{seed}.npy"
    )


def summary(values):
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
    records = {
        f"{cap:.3f}": {}
        for cap in CAPS
    }

    cache_objects = {
        dataset: residual.CachedResponses(
            config["prefix"]
        )
        for dataset, config in DATASETS.items()
    }

    for dataset, config in DATASETS.items():
        cache = cache_objects[dataset]
        family = config["family"]

        for seed in SEEDS:
            checkpoint = torch.load(
                checkpoint_path(family, seed),
                map_location="cpu",
                weights_only=False,
            )
            reward_mean = float(
                checkpoint["reward_mean"]
            )
            reward_std = float(
                checkpoint["reward_std"]
            )

            raw_scores = (
                cache.rm_scores - reward_mean
            ) / reward_std

            full_scores = np.load(
                prediction_path(dataset, seed)
            ).astype(np.float32)

            full_correction = (
                full_scores - raw_scores
            )

            if (
                np.abs(full_correction).max()
                > ORIGINAL_CAP + 1e-4
            ):
                raise RuntimeError(
                    f"{dataset} seed={seed} "
                    "残差超过原始上限"
                )

            raw_metrics = residual.ranking_metrics(
                cache,
                raw_scores,
            )

            for cap in CAPS:
                cap_key = f"{cap:.3f}"
                scaled_scores = (
                    raw_scores
                    + full_correction
                    * (cap / ORIGINAL_CAP)
                )

                metrics = residual.ranking_metrics(
                    cache,
                    scaled_scores,
                )
                transitions = (
                    residual.transition_metrics(
                        cache,
                        raw_scores,
                        scaled_scores,
                    )
                )

                records[cap_key].setdefault(
                    dataset,
                    {},
                )
                records[cap_key][dataset][
                    str(seed)
                ] = {
                    "raw_top1": raw_metrics["top1"],
                    "top1": metrics["top1"],
                    "delta_top1": (
                        metrics["top1"]
                        - raw_metrics["top1"]
                    ),
                    "raw_pair": raw_metrics[
                        "pair_macro_strict"
                    ],
                    "pair": metrics[
                        "pair_macro_strict"
                    ],
                    "delta_pair": (
                        metrics[
                            "pair_macro_strict"
                        ]
                        - raw_metrics[
                            "pair_macro_strict"
                        ]
                    ),
                    "damage_rate": transitions[
                        "damage_rate"
                    ],
                    "correction_rate": transitions[
                        "correction_rate"
                    ],
                    "correction_abs_mean": float(
                        np.abs(
                            full_correction
                            * (cap / ORIGINAL_CAP)
                        ).mean()
                    ),
                }

    aggregate = {}

    for cap in CAPS:
        cap_key = f"{cap:.3f}"
        aggregate[cap_key] = {
            "datasets": {},
        }

        for dataset in DATASETS:
            items = [
                records[cap_key][dataset][str(seed)]
                for seed in SEEDS
            ]

            aggregate[cap_key]["datasets"][
                dataset
            ] = {
                "top1": summary([
                    item["top1"]
                    for item in items
                ]),
                "delta_top1": summary([
                    item["delta_top1"]
                    for item in items
                ]),
                "pair": summary([
                    item["pair"]
                    for item in items
                ]),
                "delta_pair": summary([
                    item["delta_pair"]
                    for item in items
                ]),
                "damage_rate": summary([
                    item["damage_rate"]
                    for item in items
                ]),
                "correction_rate": summary([
                    item["correction_rate"]
                    for item in items
                ]),
                "correction_abs_mean": summary([
                    item["correction_abs_mean"]
                    for item in items
                ]),
            }

        macro_by_seed = []

        for seed in SEEDS:
            seed_items = [
                records[cap_key][dataset][str(seed)]
                for dataset in DATASETS
            ]

            macro_by_seed.append({
                "top1": float(np.mean([
                    item["top1"]
                    for item in seed_items
                ])),
                "delta_top1": float(np.mean([
                    item["delta_top1"]
                    for item in seed_items
                ])),
                "pair": float(np.mean([
                    item["pair"]
                    for item in seed_items
                ])),
                "delta_pair": float(np.mean([
                    item["delta_pair"]
                    for item in seed_items
                ])),
                "damage_rate": float(np.mean([
                    item["damage_rate"]
                    for item in seed_items
                ])),
            })

        aggregate[cap_key]["macro"] = {
            key: summary([
                item[key]
                for item in macro_by_seed
            ])
            for key in [
                "top1",
                "delta_top1",
                "pair",
                "delta_pair",
                "damage_rate",
            ]
        }

    output = {
        "version": (
            "sgldsv_residual_cap_diagnosis_v1"
        ),
        "status": (
            "test_informed_diagnostic_not_blind"
        ),
        "original_training_cap": ORIGINAL_CAP,
        "evaluated_caps": CAPS,
        "seeds": SEEDS,
        "records": records,
        "aggregate": aggregate,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        "cap | Macro Top1 | ΔTop1 | "
        "Macro Pair | ΔPair | Damage"
    )
    print("-" * 76)

    for cap in CAPS:
        item = aggregate[
            f"{cap:.3f}"
        ]["macro"]

        print(
            f"{cap:>4.3f} | "
            f"{item['top1']['mean']:.6f} | "
            f"{item['delta_top1']['mean']:+.6f} | "
            f"{item['pair']['mean']:.6f} | "
            f"{item['delta_pair']['mean']:+.6f} | "
            f"{item['damage_rate']['mean']:.6f}"
        )

    print("\n===== 各数据集 ΔTop1 / ΔPair =====")

    for cap in CAPS:
        print(f"\ncap={cap:.3f}")

        for dataset in DATASETS:
            item = aggregate[
                f"{cap:.3f}"
            ]["datasets"][dataset]

            print(
                f"  {dataset}: "
                f"ΔTop1="
                f"{item['delta_top1']['mean']:+.6f}, "
                f"ΔPair="
                f"{item['delta_pair']['mean']:+.6f}, "
                f"Damage="
                f"{item['damage_rate']['mean']:.6f}"
            )

    print("\n结果：", OUTPUT_PATH)


if __name__ == "__main__":
    main()

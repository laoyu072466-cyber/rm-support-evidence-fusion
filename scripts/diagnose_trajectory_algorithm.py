from pathlib import Path
import json
import sys

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/rm_traj_project")
sys.path.insert(0, str(ROOT / "scripts"))

from train_trajectory_head_joint_default import (
    DatasetStore,
    TrajectoryCorrectionHead,
    calculate_metrics,
    pack_questions,
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

BETAS = [
    0.0,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.8,
    1.0,
]

OUTPUT_PATH = (
    ROOT / "outputs/"
    "trajectory_algorithm_diagnosis_current_tests.json"
)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def chunk_counts(store):
    if store.offset_format == "pairs":
        return (
            store.offsets[:, 1]
            - store.offsets[:, 0]
        ).astype(np.int64)

    return np.diff(store.offsets).astype(np.int64)


@torch.no_grad()
def predict_scores_and_alpha(
    model,
    store,
    batch_questions=16,
):
    model.eval()

    predictions = np.empty(
        len(store.scores),
        dtype=np.float32,
    )
    alpha_values = np.empty(
        len(store.scores),
        dtype=np.float32,
    )

    for start in range(
        0,
        len(store.question_uids),
        batch_questions,
    ):
        uids = store.question_uids[
            start:start + batch_questions
        ]
        refs = [(store, uid) for uid in uids]

        (
            hidden,
            counts,
            rewards,
            _,
            lengths,
            slices,
        ) = pack_questions(refs)

        scores, alpha = model(
            hidden,
            counts,
            rewards,
            lengths,
        )

        scores = scores.cpu().numpy()
        alpha = alpha.cpu().numpy()

        for uid, (left, right) in zip(uids, slices):
            indices = store.groups[uid]
            predictions[indices] = scores[left:right]
            alpha_values[indices] = alpha[left:right]

    return predictions, alpha_values


def evaluate_prediction(store, prediction):
    metrics = calculate_metrics(
        store,
        prediction,
    )
    raw_pair = raw_pair_metric(store)

    return {
        "raw_top1": metrics["raw_top1"],
        "raw_pair": raw_pair,
        "corrected_top1": metrics["corrected_top1"],
        "corrected_pair": metrics[
            "corrected_pair_macro_strict"
        ],
        "top1_delta": (
            metrics["corrected_top1"]
            - metrics["raw_top1"]
        ),
        "pair_delta": (
            metrics["corrected_pair_macro_strict"]
            - raw_pair
        ),
        "correction_rate": metrics[
            "correction_rate"
        ],
        "damage_rate": metrics["damage_rate"],
    }


def build_beta_curve(
    stores,
    normalized_raw,
    full_predictions,
):
    curve = []

    for beta in BETAS:
        datasets = {}

        for name, store in stores.items():
            raw = normalized_raw[name]
            full = full_predictions[name]

            prediction = (
                raw
                + beta * (full - raw)
            )
            datasets[name] = evaluate_prediction(
                store,
                prediction,
            )

        macro_top1 = float(np.mean([
            item["corrected_top1"]
            for item in datasets.values()
        ]))
        macro_pair = float(np.mean([
            item["corrected_pair"]
            for item in datasets.values()
        ]))
        macro_damage = float(np.mean([
            item["damage_rate"]
            for item in datasets.values()
        ]))

        curve.append({
            "beta": beta,
            "macro_top1": macro_top1,
            "macro_pair": macro_pair,
            "macro_damage": macro_damage,
            "datasets": datasets,
        })

    return curve


def select_best(curve):
    return max(
        curve,
        key=lambda item: (
            item["macro_top1"],
            item["macro_pair"],
            -item["macro_damage"],
        ),
    )


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

    stores = {
        name: DatasetStore(
            config["prefix"],
            config["display"],
        )
        for name, config in DATASETS.items()
    }

    normalized_raw = {
        name: (
            store.scores.astype(np.float64)
            - normalization["reward_mean"]
        ) / normalization["reward_std"]
        for name, store in stores.items()
    }

    distribution_shift = {}

    for name, store in stores.items():
        counts = chunk_counts(store)

        reward_z = normalized_raw[name]
        log_t_z = (
            np.log1p(counts)
            - normalization["log_t_mean"]
        ) / normalization["log_t_std"]
        log_l_z = (
            np.log1p(store.response_lengths)
            - normalization["log_l_mean"]
        ) / normalization["log_l_std"]

        distribution_shift[name] = {
            "reward_z": stats(reward_z),
            "log_chunk_count_z": stats(log_t_z),
            "log_response_length_z": stats(log_l_z),
        }

    seed_predictions = {}
    seed_alphas = {}
    seed_curves = {}
    seed_diagnostics = {}

    for seed, checkpoint_path in CHECKPOINTS.items():
        print("\n评测诊断 seed：", seed)

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

        predictions = {}
        alphas = {}
        diagnostics = {}

        for name, store in stores.items():
            prediction, alpha = (
                predict_scores_and_alpha(
                    model,
                    store,
                )
            )

            predictions[name] = prediction
            alphas[name] = alpha

            correction = (
                prediction
                - normalized_raw[name]
            )

            diagnostics[name] = {
                "alpha": stats(alpha),
                "alpha_by_label": {
                    "positive": stats(
                        alpha[store.labels == 1]
                    ),
                    "negative": stats(
                        alpha[store.labels == 0]
                    ),
                },
                "correction": stats(correction),
                "absolute_correction": stats(
                    np.abs(correction)
                ),
            }

        curve = build_beta_curve(
            stores,
            normalized_raw,
            predictions,
        )

        seed_predictions[seed] = predictions
        seed_alphas[seed] = alphas
        seed_curves[seed] = curve
        seed_diagnostics[seed] = diagnostics

        best = select_best(curve)
        full = next(
            item
            for item in curve
            if item["beta"] == 1.0
        )

        print(
            f"  完整算法：Top1={full['macro_top1']:.6f}, "
            f"Pair={full['macro_pair']:.6f}, "
            f"Damage={full['macro_damage']:.6f}"
        )
        print(
            f"  最佳缩放：beta={best['beta']}, "
            f"Top1={best['macro_top1']:.6f}, "
            f"Pair={best['macro_pair']:.6f}, "
            f"Damage={best['macro_damage']:.6f}"
        )

        del model
        torch.cuda.empty_cache()

    ensemble_predictions = {}

    for name in DATASETS:
        stacked = np.stack([
            seed_predictions[seed][name]
            for seed in CHECKPOINTS
        ])

        ensemble_predictions[name] = {
            "mean": stacked.mean(axis=0),
            "median": np.median(
                stacked,
                axis=0,
            ),
        }

    ensemble_curves = {}

    for method in ["mean", "median"]:
        predictions = {
            name: ensemble_predictions[name][method]
            for name in DATASETS
        }
        ensemble_curves[method] = build_beta_curve(
            stores,
            normalized_raw,
            predictions,
        )

    per_dataset_best_beta = {}

    mean_curve = ensemble_curves["mean"]

    for name in DATASETS:
        best = max(
            mean_curve,
            key=lambda item: (
                item["datasets"][name][
                    "corrected_top1"
                ],
                item["datasets"][name][
                    "corrected_pair"
                ],
                -item["datasets"][name][
                    "damage_rate"
                ],
            ),
        )

        per_dataset_best_beta[name] = {
            "beta": best["beta"],
            **best["datasets"][name],
        }

    result = {
        "version": "trajectory_algorithm_diagnosis_v1",
        "status": (
            "exploratory_test-informed_diagnosis"
        ),
        "warning": (
            "These datasets are no longer blind after the "
            "previous final evaluation."
        ),
        "beta_definition": (
            "score_beta = normalized_original + "
            "beta * (learned_score - normalized_original)"
        ),
        "beta_grid": BETAS,
        "distribution_shift": distribution_shift,
        "per_seed_curves": {
            str(seed): curve
            for seed, curve in seed_curves.items()
        },
        "per_seed_diagnostics": {
            str(seed): value
            for seed, value in seed_diagnostics.items()
        },
        "ensemble_curves": ensemble_curves,
        "ensemble_best": {
            method: select_best(curve)
            for method, curve in ensemble_curves.items()
        },
        "per_dataset_best_beta_using_mean_ensemble":
            per_dataset_best_beta,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("===== 算法问题诊断结论所需数据 =====")

    for method, curve in ensemble_curves.items():
        best = select_best(curve)
        full = next(
            item
            for item in curve
            if item["beta"] == 1.0
        )
        baseline = next(
            item
            for item in curve
            if item["beta"] == 0.0
        )

        print(
            f"{method} ensemble | "
            f"baseline Top1={baseline['macro_top1']:.6f}, "
            f"Pair={baseline['macro_pair']:.6f}"
        )
        print(
            f"{method} ensemble | "
            f"full Top1={full['macro_top1']:.6f}, "
            f"Pair={full['macro_pair']:.6f}, "
            f"Damage={full['macro_damage']:.6f}"
        )
        print(
            f"{method} ensemble | "
            f"best beta={best['beta']}, "
            f"Top1={best['macro_top1']:.6f}, "
            f"Pair={best['macro_pair']:.6f}, "
            f"Damage={best['macro_damage']:.6f}"
        )

    print("\n各数据集最佳 beta（mean ensemble）：")
    for name, item in per_dataset_best_beta.items():
        print(
            f"{name}: beta={item['beta']}, "
            f"Top1={item['corrected_top1']:.6f}, "
            f"Pair={item['corrected_pair']:.6f}, "
            f"Damage={item['damage_rate']:.6f}"
        )

    print("\n输入分布偏移：")
    for name, item in distribution_shift.items():
        print(
            f"{name}: "
            f"reward_z_mean="
            f"{item['reward_z']['mean']:.3f}, "
            f"reward_z_std="
            f"{item['reward_z']['std']:.3f}, "
            f"length_z_mean="
            f"{item['log_response_length_z']['mean']:.3f}"
        )

    print("\n结果文件：", OUTPUT_PATH)


if __name__ == "__main__":
    main()

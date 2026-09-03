from pathlib import Path
import json
import sys
import time

import numpy as np
import torch
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import explore_cast_rm_gate as cast


QUANTILES = [0.90, 0.95, 0.975, 0.99]
AGREEMENTS = [0.0, 2.0 / 3.0, 1.0]


def macro_with_guard(
    tables,
    utilities,
    distances,
    distance_cutoff,
    agreement_min,
    gate_threshold,
):
    guarded = {}
    rejection = {}

    for name, table in tables.items():
        utility = utilities[name].copy()

        is_ood = distances[name] > distance_cutoff
        low_agreement = (
            table["X"][:, 8] + 1e-8
            < agreement_min
        )
        rejected = is_ood | low_agreement

        utility[rejected] = -1e9
        guarded[name] = utility

        switched = table["switch"]
        rejection[name] = {
            "all_questions": float(np.mean(rejected)),
            "proposed_switches": float(
                np.mean(rejected[switched])
                if np.any(switched)
                else 0.0
            ),
            "distance_only": float(np.mean(is_ood)),
            "agreement_only": float(
                np.mean(low_agreement)
            ),
        }

    summary = cast.summarize(
        tables,
        guarded,
        gate_threshold,
    )
    summary["guard_rejection"] = rejection
    return summary


def main():
    started = time.time()

    previous = json.loads(
        (
            ROOT / "outputs/cast_rm/"
            "cast_rm_gate_v0_current_tests.json"
        ).read_text(encoding="utf-8")
    )

    beta = float(previous["selected_beta"])
    gate_threshold = float(
        previous["selected_threshold"]
    )

    print("===== CAST-RM OOD Guard =====")
    print("沿用 beta：", beta)
    print("沿用门控阈值：", gate_threshold)

    stores = {}
    for name, (prefix, dataset, _) in cast.SPECS.items():
        stores[name] = cast.DatasetStore(
            prefix,
            dataset,
        )

    ensemble, normalization = cast.extract_ensemble(
        stores
    )

    pilot_names = ["GSM8K_PILOT", "MATH_PILOT"]
    test_names = [
        "GSM8K_ID",
        "MATH_ID",
        "SVAMP_OOD",
    ]

    pilot_tables = {
        name: cast.make_table(
            name,
            stores[name],
            ensemble[name],
            normalization,
            beta,
        )
        for name in pilot_names
    }

    X_pilot, y_pilot, switch_pilot, _ = (
        cast.concatenate_tables(pilot_tables)
    )

    gates = cast.fit_full_gate(
        X_pilot,
        y_pilot,
        switch_pilot,
    )

    # 去掉恒定特征，例如当前固定 beta。
    usable = np.std(X_pilot, axis=0) > 1e-7

    scaler = RobustScaler(
        quantile_range=(10.0, 90.0)
    )
    scaler.fit(X_pilot[:, usable])

    def distance(X):
        z = scaler.transform(X[:, usable])
        z = np.clip(z, -10.0, 10.0)
        return np.sqrt(np.mean(z * z, axis=1))

    pilot_distance = distance(X_pilot)

    test_tables = {
        name: cast.make_table(
            name,
            stores[name],
            ensemble[name],
            normalization,
            beta,
        )
        for name in test_names
    }

    test_utilities = {
        name: cast.predict_full_gate(
            gates,
            table["X"],
            table["switch"],
        )
        for name, table in test_tables.items()
    }

    test_distances = {
        name: distance(table["X"])
        for name, table in test_tables.items()
    }

    print("\n===== 分布距离 =====")
    print(
        "Pilot:",
        {
            "median": float(np.median(pilot_distance)),
            "p90": float(np.quantile(
                pilot_distance, 0.90
            )),
            "p95": float(np.quantile(
                pilot_distance, 0.95
            )),
            "p99": float(np.quantile(
                pilot_distance, 0.99
            )),
        },
    )

    for name in test_names:
        values = test_distances[name]
        print(
            name,
            {
                "median": round(
                    float(np.median(values)), 4
                ),
                "p90": round(
                    float(np.quantile(values, 0.90)),
                    4,
                ),
                "above_pilot_p95": round(
                    float(np.mean(
                        values > np.quantile(
                            pilot_distance, 0.95
                        )
                    )),
                    4,
                ),
            },
        )

    runs = []

    for quantile in QUANTILES:
        cutoff = float(np.quantile(
            pilot_distance,
            quantile,
        ))

        for agreement in AGREEMENTS:
            summary = macro_with_guard(
                test_tables,
                test_utilities,
                test_distances,
                cutoff,
                agreement,
                gate_threshold,
            )

            runs.append({
                "pilot_distance_quantile": quantile,
                "distance_cutoff": cutoff,
                "minimum_seed_agreement": agreement,
                "result": summary,
            })

    # 预先推荐的默认规则，不按测试集身份判断。
    recommended = next(
        row for row in runs
        if (
            row["pilot_distance_quantile"] == 0.95
            and abs(
                row["minimum_seed_agreement"]
                - 2.0 / 3.0
            ) < 1e-8
        )
    )

    print("\n===== 默认 OOD Guard：P95 + 2/3 种子一致 =====")

    for name in test_names:
        value = recommended["result"][name]
        reject = recommended["result"][
            "guard_rejection"
        ][name]

        print(
            f"{name}: "
            f"Top1={value['raw_top1']:.6f}"
            f" -> {value['top1']:.6f} "
            f"({value['top1_delta']:+.6f}), "
            f"Pair={value['pair_macro_strict']:.6f}, "
            f"Damage={value['damage_rate']:.6f}, "
            f"Coverage={value['authorized_coverage']:.6f}, "
            f"GuardReject={reject['proposed_switches']:.6f}"
        )

    print(
        "Macro:",
        json.dumps(
            recommended["result"]["macro"],
            ensure_ascii=False,
        ),
    )

    print("\n===== 全部 Guard 组合 =====")
    for row in runs:
        macro = row["result"]["macro"]
        print(
            f"q={row['pilot_distance_quantile']:.3f}, "
            f"agree={row['minimum_seed_agreement']:.3f} | "
            f"Top1={macro['top1']:.6f}, "
            f"Pair={macro['pair_macro_strict']:.6f}, "
            f"Damage={macro['damage_rate']:.6f}, "
            f"Coverage={macro['authorized_coverage']:.6f}"
        )

    result = {
        "version": "cast_rm_ood_guard_v1",
        "scope": "exploratory_current_tests",
        "beta": beta,
        "gate_threshold": gate_threshold,
        "recommended_rule": {
            "pilot_distance_quantile": 0.95,
            "minimum_seed_agreement": 2.0 / 3.0,
        },
        "recommended_result": recommended,
        "all_runs": runs,
        "elapsed_seconds": round(
            time.time() - started,
            3,
        ),
        "peak_gpu_gb": round(
            torch.cuda.max_memory_allocated()
            / (1024 ** 3),
            3,
        ),
    }

    output = (
        ROOT / "outputs/cast_rm/"
        "cast_rm_ood_guard_v1.json"
    )
    manifest = (
        ROOT / "data/manifests/"
        "cast_rm_ood_guard_v1.json"
    )

    text = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    ) + "\n"

    output.write_text(text, encoding="utf-8")
    manifest.write_text(text, encoding="utf-8")

    print("\n结果：", output)
    print("清单：", manifest)
    print("耗时秒：", result["elapsed_seconds"])


if __name__ == "__main__":
    main()

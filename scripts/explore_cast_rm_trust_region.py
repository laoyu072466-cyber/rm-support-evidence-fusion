from pathlib import Path
import json
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import explore_cast_rm_gate as cast


RANK_BETA = 0.30
EVIDENCE_BETAS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
KAPPAS = [0.0, 0.5, 1.0, 1.5, 2.0]
TARGET_DAMAGE = 0.02
TOP1_TIE_BAND = 0.005
EPSILON = 1e-6


def build_state(
    name,
    store,
    ensemble,
    normalization,
):
    table = cast.make_table(
        name,
        store,
        ensemble,
        normalization,
        RANK_BETA,
    )

    raw = table["raw"]
    seed_delta = (
        ensemble["predictions"]
        - raw[None, :]
    ).astype(np.float32)

    return {
        "table": table,
        "seed_delta": seed_delta,
    }


def apply_trust_region(
    state,
    evidence_beta,
    kappa,
):
    table = state["table"]
    raw = table["raw"]
    seed_delta = state["seed_delta"]

    # 首先执行非 Top-1 重排，再把原始 Top-1 投影回来。
    scores = np.array(
        table["blend"],
        dtype=np.float32,
        copy=True,
    )

    authorized = 0
    evidence_values = []
    uncertainty_values = []

    for record in table["records"]:
        indices = record["indices"]
        raw_top = record["raw_top"]

        others = indices[indices != raw_top]

        # 默认安全状态：原始 Top-1 保持第一。
        scores[raw_top] = max(
            float(scores[raw_top]),
            float(np.max(scores[others])) + EPSILON,
        )

        raw_margins = (
            raw[raw_top] - raw[indices]
        )

        gap_by_seed = (
            seed_delta[:, indices]
            - seed_delta[:, [raw_top]]
        )
        gap_mean = np.mean(gap_by_seed, axis=0)
        gap_std = np.std(gap_by_seed, axis=0)

        conservative_gap = (
            gap_mean - kappa * gap_std
        )

        evidence = (
            evidence_beta * conservative_gap
            - raw_margins
        )

        # 原始第一名不能作为自己的 challenger。
        raw_top_local = int(
            np.where(indices == raw_top)[0][0]
        )
        evidence[raw_top_local] = -np.inf

        challenger_local = int(np.argmax(evidence))
        best_evidence = float(
            evidence[challenger_local]
        )

        evidence_values.append(best_evidence)
        uncertainty_values.append(float(
            gap_std[challenger_local]
        ))

        if best_evidence <= 0:
            continue

        challenger = int(indices[challenger_local])

        # 最小幅度换榜，不让校正器任意拉大分数。
        scores[challenger] = max(
            float(scores[challenger]),
            float(scores[raw_top]) + EPSILON,
        )
        authorized += 1

    return (
        scores,
        authorized,
        {
            "evidence_mean": float(np.mean(
                evidence_values
            )),
            "evidence_positive_rate": float(
                np.mean(
                    np.asarray(evidence_values) > 0
                )
            ),
            "challenger_uncertainty_mean": float(
                np.mean(uncertainty_values)
            ),
        },
    )


def evaluate_states(
    states,
    evidence_beta,
    kappa,
):
    results = {}
    diagnostics = {}

    for name, state in states.items():
        scores, authorized, diag = apply_trust_region(
            state,
            evidence_beta,
            kappa,
        )

        results[name] = cast.metrics(
            state["table"],
            scores,
            authorized,
        )
        diagnostics[name] = diag

    results["macro"] = {
        "top1": float(np.mean([
            value["top1"]
            for value in results.values()
        ])),
        "pair_macro_strict": float(np.mean([
            value["pair_macro_strict"]
            for value in results.values()
        ])),
        "damage_rate": float(np.mean([
            value["damage_rate"]
            for value in results.values()
        ])),
        "authorized_coverage": float(np.mean([
            value["authorized_coverage"]
            for value in results.values()
        ])),
    }
    results["diagnostics"] = diagnostics

    return results


def main():
    started = time.time()

    print("===== CAST-RM 保守证据信任域 =====")
    print("非 Top-1 重排 beta：", RANK_BETA)

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

    pilot_states = {
        name: build_state(
            name,
            stores[name],
            ensemble[name],
            normalization,
        )
        for name in pilot_names
    }

    search = []

    print("\n===== Pilot 参数搜索 =====")

    for evidence_beta in EVIDENCE_BETAS:
        for kappa in KAPPAS:
            result = evaluate_states(
                pilot_states,
                evidence_beta,
                kappa,
            )

            row = {
                "evidence_beta": evidence_beta,
                "kappa": kappa,
                "pilot": result,
            }
            search.append(row)

            macro = result["macro"]
            print(
                f"beta={evidence_beta:.2f}, "
                f"kappa={kappa:.1f} | "
                f"Top1={macro['top1']:.6f}, "
                f"Pair={macro['pair_macro_strict']:.6f}, "
                f"Damage={macro['damage_rate']:.6f}, "
                f"Coverage={macro['authorized_coverage']:.6f}"
            )

    eligible = [
        row for row in search
        if (
            row["pilot"]["macro"]["damage_rate"]
            <= TARGET_DAMAGE
            and row["pilot"]["macro"][
                "authorized_coverage"
            ] >= 0.01
        )
    ]

    if not eligible:
        eligible = [
            row for row in search
            if row["pilot"]["macro"][
                "authorized_coverage"
            ] >= 0.01
        ]

    best_top1 = max(
        row["pilot"]["macro"]["top1"]
        for row in eligible
    )

    band = [
        row for row in eligible
        if row["pilot"]["macro"]["top1"]
        >= best_top1 - TOP1_TIE_BAND
    ]

    selected = max(
        band,
        key=lambda row: (
            row["pilot"]["macro"][
                "pair_macro_strict"
            ],
            -row["pilot"]["macro"]["damage_rate"],
            -row["pilot"]["macro"][
                "authorized_coverage"
            ],
        ),
    )

    evidence_beta = float(
        selected["evidence_beta"]
    )
    kappa = float(selected["kappa"])

    print("\n===== Pilot 最终选择 =====")
    print("evidence_beta：", evidence_beta)
    print("kappa：", kappa)
    print(json.dumps(
        selected["pilot"]["macro"],
        ensure_ascii=False,
        indent=2,
    ))

    test_states = {
        name: build_state(
            name,
            stores[name],
            ensemble[name],
            normalization,
        )
        for name in test_names
    }

    test_result = evaluate_states(
        test_states,
        evidence_beta,
        kappa,
    )

    print("\n===== 信任域测试结果 =====")

    for name in test_names:
        value = test_result[name]
        diag = test_result["diagnostics"][name]

        print(
            f"{name}: "
            f"Top1={value['raw_top1']:.6f}"
            f" -> {value['top1']:.6f} "
            f"({value['top1_delta']:+.6f}), "
            f"Pair={value['pair_macro_strict']:.6f}, "
            f"Damage={value['damage_rate']:.6f}, "
            f"Correction={value['correction_rate']:.6f}, "
            f"Coverage={value['authorized_coverage']:.6f}, "
            f"EvidencePositive="
            f"{diag['evidence_positive_rate']:.6f}"
        )

    print(
        "Macro:",
        json.dumps(
            test_result["macro"],
            ensure_ascii=False,
            indent=2,
        ),
    )

    result = {
        "version": "cast_rm_trust_region_v2",
        "scope": (
            "pilot_selected_current_tests_exploratory"
        ),
        "rank_beta": RANK_BETA,
        "selected_evidence_beta": evidence_beta,
        "selected_kappa": kappa,
        "pilot": selected["pilot"],
        "test": test_result,
        "search": search,
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
        "cast_rm_trust_region_v2.json"
    )
    manifest = (
        ROOT / "data/manifests/"
        "cast_rm_trust_region_v2.json"
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

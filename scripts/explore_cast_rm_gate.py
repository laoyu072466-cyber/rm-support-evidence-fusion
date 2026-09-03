from pathlib import Path
from collections import Counter
import json
import math
import sys
import time

import numpy as np
import torch

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_trajectory_head_joint_default import (
    DatasetStore,
    TrajectoryCorrectionHead,
    DEVICE,
)
from diagnose_trajectory_algorithm import (
    predict_scores_and_alpha,
)

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

SPECS = {
    "GSM8K_PILOT": ("gsm_pilot", "GSM8K", "pilot"),
    "MATH_PILOT": ("math_pilot", "MATH", "pilot"),
    "GSM8K_ID": ("gsm_id_test", "GSM8K", "test"),
    "MATH_ID": ("math_id_test", "MATH", "test"),
    "SVAMP_OOD": ("svamp_ood", "SVAMP", "test"),
}

BETA_GRID = [0.05, 0.10, 0.20, 0.30]
TARGET_DAMAGE = 0.02
TOP1_TIE_BAND = 0.005
FOLDS = 5
GATE_SEED = 20260901


def chunk_counts(store):
    if store.offsets.ndim == 1:
        return np.diff(store.offsets).astype(np.float32)
    return (
        store.offsets[:, 1] - store.offsets[:, 0]
    ).astype(np.float32)


def load_model(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = TrajectoryCorrectionHead(
        checkpoint["normalization"]
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.gamma = float(
        checkpoint["configuration"]["gamma"]
    )
    model.eval()

    return model, checkpoint


def extract_ensemble(stores):
    outputs = {
        name: {
            "predictions": [],
            "alphas": [],
        }
        for name in stores
    }

    normalization = None

    for seed, checkpoint_path in CHECKPOINTS.items():
        print("\n" + "=" * 72)
        print("加载轨迹头 seed：", seed)

        model, checkpoint = load_model(checkpoint_path)
        current_norm = checkpoint["normalization"]

        if normalization is None:
            normalization = dict(current_norm)
        else:
            for key in normalization:
                if not math.isclose(
                    float(normalization[key]),
                    float(current_norm[key]),
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ):
                    raise RuntimeError(
                        f"不同 seed 的标准化参数不一致：{key}"
                    )

        for name, store in stores.items():
            prediction, alpha = predict_scores_and_alpha(
                model,
                store,
                batch_questions=16,
            )
            outputs[name]["predictions"].append(
                np.asarray(prediction, dtype=np.float32)
            )
            outputs[name]["alphas"].append(
                np.asarray(alpha, dtype=np.float32)
            )
            print(
                f"{name}: 候选={len(prediction)}, "
                f"alpha_mean={float(np.mean(alpha)):.4f}"
            )

        del model
        torch.cuda.empty_cache()

    for name in outputs:
        outputs[name]["predictions"] = np.stack(
            outputs[name]["predictions"],
            axis=0,
        )
        outputs[name]["alphas"] = np.stack(
            outputs[name]["alphas"],
            axis=0,
        )

    return outputs, normalization


def make_table(
    name,
    store,
    ensemble,
    normalization,
    beta,
):
    reward_mean = float(normalization["reward_mean"])
    reward_std = float(normalization["reward_std"])

    raw = (
        store.scores.astype(np.float32) - reward_mean
    ) / reward_std

    seed_scores = ensemble["predictions"]
    seed_alphas = ensemble["alphas"]

    seed_delta = seed_scores - raw[None, :]
    mean_delta = np.mean(seed_delta, axis=0)
    std_delta = np.std(seed_delta, axis=0)

    blend = raw + beta * mean_delta

    counts = chunk_counts(store)
    lengths = store.response_lengths.astype(np.float32)

    features = []
    targets = []
    switches = []
    records = []

    for uid in store.question_uids:
        indices = np.asarray(
            store.groups[uid],
            dtype=np.int64,
        )

        raw_local = raw[indices]
        blend_local = blend[indices]
        labels_local = store.labels[indices]

        raw_order = np.argsort(raw_local)
        raw_top_local = int(raw_order[-1])
        raw_second_local = int(raw_order[-2])

        proposal_local = int(
            np.argmax(blend_local)
        )

        raw_top = int(indices[raw_top_local])
        raw_second = int(indices[raw_second_local])
        proposal = int(indices[proposal_local])

        raw_correct = int(store.labels[raw_top] == 1)
        proposal_correct = int(
            store.labels[proposal] == 1
        )
        target = proposal_correct - raw_correct

        seed_candidate_scores = (
            raw[indices][None, :]
            + beta * seed_delta[:, indices]
        )
        seed_top_local = np.argmax(
            seed_candidate_scores,
            axis=1,
        )

        proposal_agreement = float(
            np.mean(seed_top_local == proposal_local)
        )
        seed_switch_rate = float(
            np.mean(seed_top_local != raw_top_local)
        )

        delta_gap_by_seed = (
            seed_delta[:, proposal]
            - seed_delta[:, raw_top]
        )

        alpha_top = seed_alphas[:, raw_top]
        alpha_proposal = seed_alphas[:, proposal]

        raw_margin = float(
            raw[raw_top] - raw[raw_second]
        )
        proposed_margin = float(
            blend[proposal] - blend[raw_top]
        )

        feature = [
            raw_margin,
            float(raw[raw_top]),
            float(raw[proposal]),
            float(np.std(raw_local)),
            float(np.log1p(len(indices))),
            proposed_margin,
            float(np.mean(delta_gap_by_seed)),
            float(np.std(delta_gap_by_seed)),
            proposal_agreement,
            seed_switch_rate,
            float(np.mean(alpha_top)),
            float(np.std(alpha_top)),
            float(np.mean(alpha_proposal)),
            float(np.std(alpha_proposal)),
            float(np.log1p(counts[raw_top])),
            float(
                np.log1p(counts[proposal])
                - np.log1p(counts[raw_top])
            ),
            float(np.log1p(lengths[raw_top])),
            float(
                np.log1p(lengths[proposal])
                - np.log1p(lengths[raw_top])
            ),
            float(np.mean(std_delta[indices])),
            float(np.std(mean_delta[indices])),
            float(beta),
        ]

        features.append(feature)
        targets.append(target)
        switches.append(proposal != raw_top)

        records.append({
            "uid": uid,
            "indices": indices,
            "raw_top": raw_top,
            "proposal": proposal,
            "raw_correct": raw_correct,
            "proposal_correct": proposal_correct,
        })

    return {
        "name": name,
        "store": store,
        "raw": raw,
        "blend": blend,
        "X": np.asarray(features, dtype=np.float32),
        "y": np.asarray(targets, dtype=np.int8),
        "switch": np.asarray(switches, dtype=bool),
        "records": records,
    }


def fit_binary_gate(X, y, positive_kind):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if positive_kind == "benefit":
        binary = (y == 1).astype(np.int8)
        weights = np.where(binary == 1, 3.0, 1.0)
    elif positive_kind == "damage":
        binary = (y == -1).astype(np.int8)
        weights = np.where(binary == 1, 8.0, 1.0)
    else:
        raise ValueError(positive_kind)

    if len(np.unique(binary)) < 2:
        probability = float(
            (binary.sum() + 0.5) / (len(binary) + 1.0)
        )
        return {
            "kind": "constant",
            "probability": probability,
            "scaler": scaler,
        }

    model = LogisticRegression(
        C=0.25,
        max_iter=3000,
        random_state=GATE_SEED,
        solver="lbfgs",
    )
    model.fit(
        X_scaled,
        binary,
        sample_weight=weights,
    )

    return {
        "kind": "model",
        "model": model,
        "scaler": scaler,
    }


def predict_binary_gate(fitted, X):
    if fitted["kind"] == "constant":
        return np.full(
            len(X),
            fitted["probability"],
            dtype=np.float32,
        )

    X_scaled = fitted["scaler"].transform(X)
    return fitted["model"].predict_proba(
        X_scaled
    )[:, 1].astype(np.float32)


def crossfit_gate(X, y, switch):
    selected = np.flatnonzero(switch)

    if len(selected) < 20:
        raise RuntimeError(
            f"发生换榜的问题太少：{len(selected)}"
        )

    utility = np.full(len(X), -1e9, dtype=np.float32)
    benefit_prob = np.zeros(len(X), dtype=np.float32)
    damage_prob = np.zeros(len(X), dtype=np.float32)

    splitter = KFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=GATE_SEED,
    )

    for train_rel, valid_rel in splitter.split(selected):
        train_index = selected[train_rel]
        valid_index = selected[valid_rel]

        benefit_gate = fit_binary_gate(
            X[train_index],
            y[train_index],
            "benefit",
        )
        damage_gate = fit_binary_gate(
            X[train_index],
            y[train_index],
            "damage",
        )

        pb = predict_binary_gate(
            benefit_gate,
            X[valid_index],
        )
        pd = predict_binary_gate(
            damage_gate,
            X[valid_index],
        )

        benefit_prob[valid_index] = pb
        damage_prob[valid_index] = pd

        utility[valid_index] = pb - 3.0 * pd

    return utility, benefit_prob, damage_prob


def fit_full_gate(X, y, switch):
    selected = np.flatnonzero(switch)

    benefit_gate = fit_binary_gate(
        X[selected],
        y[selected],
        "benefit",
    )
    damage_gate = fit_binary_gate(
        X[selected],
        y[selected],
        "damage",
    )

    return benefit_gate, damage_gate


def predict_full_gate(gates, X, switch):
    benefit_gate, damage_gate = gates

    utility = np.full(len(X), -1e9, dtype=np.float32)
    selected = np.flatnonzero(switch)

    pb = predict_binary_gate(
        benefit_gate,
        X[selected],
    )
    pd = predict_binary_gate(
        damage_gate,
        X[selected],
    )

    utility[selected] = pb - 3.0 * pd
    return utility


def apply_gate(table, utility, threshold):
    scores = np.array(
        table["blend"],
        dtype=np.float32,
        copy=True,
    )

    authorized = 0

    for row_index, record in enumerate(table["records"]):
        allow_flip = (
            table["switch"][row_index]
            and utility[row_index] >= threshold
        )

        if allow_flip:
            authorized += 1
            continue

        indices = record["indices"]
        raw_top = record["raw_top"]

        other = indices[indices != raw_top]
        if len(other):
            scores[raw_top] = max(
                float(scores[raw_top]),
                float(np.max(scores[other])) + 1e-6,
            )

    return scores, authorized


def metrics(table, scores, authorized=0):
    store = table["store"]
    raw = table["raw"]

    top1 = []
    raw_top1 = []
    pair = []

    damaged = 0
    raw_correct_count = 0
    corrected = 0
    raw_wrong_count = 0
    changed = 0

    for record in table["records"]:
        indices = record["indices"]
        labels = store.labels[indices]

        raw_local_top = int(
            np.argmax(raw[indices])
        )
        new_local_top = int(
            np.argmax(scores[indices])
        )

        raw_correct = int(
            labels[raw_local_top] == 1
        )
        new_correct = int(
            labels[new_local_top] == 1
        )

        raw_top1.append(raw_correct)
        top1.append(new_correct)

        if raw_correct:
            raw_correct_count += 1
            if not new_correct:
                damaged += 1
        else:
            raw_wrong_count += 1
            if new_correct:
                corrected += 1

        if raw_local_top != new_local_top:
            changed += 1

        positive = scores[indices][labels == 1]
        negative = scores[indices][labels == 0]

        pair.append(
            float(np.mean(
                positive[:, None] > negative[None, :]
            ))
        )

    questions = len(table["records"])

    return {
        "questions": questions,
        "raw_top1": float(np.mean(raw_top1)),
        "top1": float(np.mean(top1)),
        "top1_delta": float(
            np.mean(top1) - np.mean(raw_top1)
        ),
        "pair_macro_strict": float(np.mean(pair)),
        "damage_rate": float(
            damaged / max(raw_correct_count, 1)
        ),
        "correction_rate": float(
            corrected / max(raw_wrong_count, 1)
        ),
        "switch_rate": float(changed / questions),
        "authorized_coverage": float(
            authorized / questions
        ),
        "net_corrected_questions": int(
            sum(top1) - sum(raw_top1)
        ),
    }


def summarize(tables, utilities, threshold):
    result = {}

    for name, table in tables.items():
        scores, authorized = apply_gate(
            table,
            utilities[name],
            threshold,
        )
        result[name] = metrics(
            table,
            scores,
            authorized,
        )

    result["macro"] = {
        "top1": float(np.mean([
            value["top1"]
            for value in result.values()
        ])),
        "pair_macro_strict": float(np.mean([
            value["pair_macro_strict"]
            for value in result.values()
        ])),
        "damage_rate": float(np.mean([
            value["damage_rate"]
            for value in result.values()
        ])),
        "authorized_coverage": float(np.mean([
            value["authorized_coverage"]
            for value in result.values()
        ])),
    }

    return result


def concatenate_tables(tables):
    names = list(tables)
    X_parts = []
    y_parts = []
    switch_parts = []
    slices = {}

    start = 0
    for name in names:
        table = tables[name]
        end = start + len(table["X"])
        slices[name] = slice(start, end)

        X_parts.append(table["X"])
        y_parts.append(table["y"])
        switch_parts.append(table["switch"])
        start = end

    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(switch_parts, axis=0),
        slices,
    )


def unrestricted_metrics(table):
    return metrics(
        table,
        table["blend"],
        authorized=int(np.sum(table["switch"])),
    )


def main():
    started = time.time()

    print("===== 加载 Pilot 与当前测试特征 =====")
    stores = {}

    for name, (prefix, dataset, _) in SPECS.items():
        stores[name] = DatasetStore(prefix, dataset)

    ensemble, normalization = extract_ensemble(stores)

    pilot_names = ["GSM8K_PILOT", "MATH_PILOT"]
    test_names = ["GSM8K_ID", "MATH_ID", "SVAMP_OOD"]

    search_results = []
    cached_pilot_tables = {}

    print("\n===== Pilot 五折门控搜索 =====")

    for beta in BETA_GRID:
        tables = {
            name: make_table(
                name,
                stores[name],
                ensemble[name],
                normalization,
                beta,
            )
            for name in pilot_names
        }
        cached_pilot_tables[beta] = tables

        X, y, switch, slices = concatenate_tables(tables)

        counts = Counter(y[switch].tolist())
        print(
            f"\nbeta={beta:.2f}，换榜问题={int(switch.sum())}，"
            f"收益标签={dict(counts)}"
        )

        utility, _, _ = crossfit_gate(X, y, switch)

        finite = utility[switch]
        thresholds = list(np.quantile(
            finite,
            np.linspace(0.0, 1.0, 41),
        ))
        thresholds.extend([
            float(np.max(finite) + 1e-6),
            float(np.min(finite) - 1e-6),
        ])
        thresholds = sorted(set(
            float(x) for x in thresholds
        ))

        for threshold in thresholds:
            utility_by_name = {
                name: utility[slices[name]]
                for name in pilot_names
            }

            summary = summarize(
                tables,
                utility_by_name,
                threshold,
            )

            search_results.append({
                "beta": beta,
                "threshold": threshold,
                "summary": summary,
            })

    eligible = [
        row for row in search_results
        if (
            row["summary"]["macro"]["damage_rate"]
            <= TARGET_DAMAGE
            and row["summary"]["macro"][
                "authorized_coverage"
            ] >= 0.01
        )
    ]

    if not eligible:
        print(
            "警告：没有配置满足 2% damage，"
            "改用最低 damage 的非零覆盖配置。"
        )
        eligible = [
            row for row in search_results
            if row["summary"]["macro"][
                "authorized_coverage"
            ] >= 0.01
        ]

    best_top1 = max(
        row["summary"]["macro"]["top1"]
        for row in eligible
    )

    top1_band = [
        row for row in eligible
        if row["summary"]["macro"]["top1"]
        >= best_top1 - TOP1_TIE_BAND
    ]

    selected = max(
        top1_band,
        key=lambda row: (
            row["summary"]["macro"][
                "pair_macro_strict"
            ],
            -row["summary"]["macro"]["damage_rate"],
            row["summary"]["macro"][
                "authorized_coverage"
            ],
        ),
    )

    selected_beta = float(selected["beta"])
    selected_threshold = float(selected["threshold"])

    print("\n===== Pilot 选择结果 =====")
    print(json.dumps(
        {
            "beta": selected_beta,
            "threshold": selected_threshold,
            "pilot_oof": selected["summary"],
        },
        ensure_ascii=False,
        indent=2,
    ))

    pilot_tables = cached_pilot_tables[selected_beta]
    X, y, switch, _ = concatenate_tables(pilot_tables)
    gates = fit_full_gate(X, y, switch)

    print("\n===== 当前三个测试集探索性评估 =====")

    test_tables = {
        name: make_table(
            name,
            stores[name],
            ensemble[name],
            normalization,
            selected_beta,
        )
        for name in test_names
    }

    test_utilities = {}

    for name, table in test_tables.items():
        test_utilities[name] = predict_full_gate(
            gates,
            table["X"],
            table["switch"],
        )

    cast_result = summarize(
        test_tables,
        test_utilities,
        selected_threshold,
    )

    projection_result = summarize(
        test_tables,
        {
            name: np.full(
                len(table["records"]),
                -1e9,
                dtype=np.float32,
            )
            for name, table in test_tables.items()
        },
        threshold=0.0,
    )

    unrestricted_result = {
        name: unrestricted_metrics(table)
        for name, table in test_tables.items()
    }
    unrestricted_result["macro"] = {
        "top1": float(np.mean([
            unrestricted_result[name]["top1"]
            for name in test_names
        ])),
        "pair_macro_strict": float(np.mean([
            unrestricted_result[name][
                "pair_macro_strict"
            ]
            for name in test_names
        ])),
        "damage_rate": float(np.mean([
            unrestricted_result[name]["damage_rate"]
            for name in test_names
        ])),
    }

    print("\n===== CAST-RM v0 测试结果 =====")
    for name in test_names:
        value = cast_result[name]
        print(
            f"{name}: "
            f"Top1={value['raw_top1']:.6f}"
            f" -> {value['top1']:.6f} "
            f"({value['top1_delta']:+.6f}), "
            f"Pair={value['pair_macro_strict']:.6f}, "
            f"Damage={value['damage_rate']:.6f}, "
            f"Coverage={value['authorized_coverage']:.6f}"
        )

    print("\n宏平均：")
    print(json.dumps(
        cast_result["macro"],
        ensure_ascii=False,
        indent=2,
    ))

    result = {
        "version": "cast_rm_gate_v0",
        "scope": (
            "exploratory_pilot_trained_current_tests"
        ),
        "selected_beta": selected_beta,
        "selected_threshold": selected_threshold,
        "pilot_oof": selected["summary"],
        "test": {
            "top1_projection_only": projection_result,
            "unrestricted_selected_beta": (
                unrestricted_result
            ),
            "cast_rm_gate": cast_result,
        },
        "gate": {
            "type": (
                "two_regularized_logistic_heads"
            ),
            "benefit_damage_utility": (
                "p_benefit_minus_3_times_p_damage"
            ),
            "folds": FOLDS,
            "target_damage": TARGET_DAMAGE,
        },
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
        "cast_rm_gate_v0_current_tests.json"
    )
    manifest = (
        ROOT / "data/manifests/"
        "cast_rm_gate_v0_current_tests.json"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

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

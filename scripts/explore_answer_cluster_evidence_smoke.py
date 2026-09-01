from pathlib import Path
from collections import defaultdict
import json
import math
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path("/root/autodl-tmp/rm_traj_project")
sys.path.insert(0, str(ROOT / "scripts"))

from audit_answer_cluster_consensus import (
    extract_answer,
    canonical_solution,
    read_jsonl,
)


GEN_CACHE = ROOT / "data/cache/generator_hidden_smoke_v1"
RM_CACHE = (
    ROOT
    / "data/cache/trajectory_features_v1"
    / "Skywork-Reward-V2-Qwen3-1.7B"
    / "layer_28"
)
DATA_ROOT = ROOT / "data/processed/prototype_v2"

OUTPUT = (
    ROOT
    / "data/manifests/answer_cluster_evidence_smoke_v1.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEEDS = [42, 123, 456]
REGULARIZATION = [0.001, 0.01, 0.1, 1.0, 10.0]
BETA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]
THRESHOLD_GRID = [0.0, 0.25, 0.5, 1.0]
PILOT_DAMAGE_LIMIT = 0.03

FEATURE_NAMES = [
    "log_cluster_count",
    "cluster_fraction",
    "log_unique_solution_count",
    "rm_max_relative",
    "rm_mean_relative",
    "rm_logmeanexp_relative",
    "rm_std_relative",
    "raw_answer_indicator",
    "negative_mean_nll_relative",
    "negative_best_nll_relative",
    "negative_last8_nll_relative",
    "hidden_centroid_norm",
    "hidden_pair_cosine",
    "hidden_separation",
    "prompt_hidden_agreement",
]

ABLATIONS = {
    "rm_support": list(range(0, 8)),
    "rm_support_nll": list(range(0, 11)),
    "rm_support_nll_hidden": list(
        range(len(FEATURE_NAMES))
    ),
}

DOMAINS = {
    "GSM8K": {
        "model": "Qwen2-1.5B",
        "family": "gsm",
        "hidden_layer_index": 16,
        "hidden_layer_name": "block_17",
        "train": {
            "smoke": "gsm_train_smoke",
            "data": "gsm_train.jsonl",
            "rm": "gsm_train",
        },
        "pilot": {
            "smoke": "gsm_pilot_smoke",
            "data": "gsm_pilot_validation.jsonl",
            "rm": "gsm_pilot",
        },
        "tests": {
            "GSM8K_ID": {
                "smoke": "gsm_id_smoke",
                "data": "gsm_id_test_mixed.jsonl",
                "rm": "gsm_id_test",
            },
            "SVAMP_OOD": {
                "smoke": "svamp_ood_smoke",
                "data": "svamp_ood_mixed.jsonl",
                "rm": "svamp_ood",
            },
        },
    },
    "MATH": {
        "model": "Qwen2-7B",
        "family": "math",
        "hidden_layer_index": 18,
        "hidden_layer_name": "block_19",
        "train": {
            "smoke": "math_train_smoke",
            "data": "math_train.jsonl",
            "rm": "math_train",
        },
        "pilot": {
            "smoke": "math_pilot_smoke",
            "data": "math_pilot_validation.jsonl",
            "rm": "math_pilot",
        },
        "tests": {
            "MATH_ID": {
                "smoke": "math_id_smoke",
                "data": "math_id_test_mixed.jsonl",
                "rm": "math_id_test",
            },
        },
    },
}


def unit_rows(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(
        values,
        axis=1,
        keepdims=True,
    )
    return values / np.maximum(norms, 1e-8)


def robust_z(values):
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad

    if scale < 1e-6:
        scale = float(np.std(values))
    if scale < 1e-6:
        scale = 1.0

    return ((values - median) / scale).astype(
        np.float32
    )


def ordinary_z(values):
    values = np.asarray(values, dtype=np.float32)
    std = float(np.std(values))
    if std < 1e-6:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def stable_logmeanexp(values):
    values = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(values))
    return float(
        maximum
        + math.log(
            float(np.exp(values - maximum).mean())
        )
    )


def candidate_key(item):
    return (
        str(item["question_uid"]),
        int(item.get("candidate_index", -1)),
    )


def load_smoke_dataset(
    domain_name,
    domain_spec,
    split_spec,
    display_name,
):
    model_name = domain_spec["model"]
    smoke_name = split_spec["smoke"]
    layer_index = domain_spec["hidden_layer_index"]

    reference_dir = (
        GEN_CACHE / model_name / "problem_solution"
    )
    alternate_dir = (
        GEN_CACHE / model_name / "question_answer"
    )

    metadata = read_jsonl(
        reference_dir / f"{smoke_name}.metadata.jsonl"
    )
    alternate_metadata = read_jsonl(
        alternate_dir / f"{smoke_name}.metadata.jsonl"
    )

    reference_keys = [
        candidate_key(item)
        for item in metadata
    ]
    alternate_keys = [
        candidate_key(item)
        for item in alternate_metadata
    ]

    if reference_keys != alternate_keys:
        raise RuntimeError(
            f"{display_name}: 两种提示的候选顺序不一致"
        )

    labels = np.asarray(
        np.load(
            reference_dir
            / f"{smoke_name}.labels_i8.npy"
        ),
        dtype=np.int8,
    )

    hidden_a_file = np.load(
        reference_dir
        / f"{smoke_name}.terminal_layers_f16.npy",
        mmap_mode="r",
    )
    hidden_b_file = np.load(
        alternate_dir
        / f"{smoke_name}.terminal_layers_f16.npy",
        mmap_mode="r",
    )

    hidden_a = unit_rows(
        np.asarray(
            hidden_a_file[layer_index],
            dtype=np.float32,
        )
    )
    hidden_b = unit_rows(
        np.asarray(
            hidden_b_file[layer_index],
            dtype=np.float32,
        )
    )

    prompt_agreement = np.sum(
        hidden_a * hidden_b,
        axis=1,
    ).astype(np.float32)

    hidden = unit_rows(hidden_a + hidden_b)

    nll_a = np.asarray(
        np.load(
            reference_dir
            / f"{smoke_name}.token_nll_f32.npy"
        ),
        dtype=np.float32,
    )
    nll_b = np.asarray(
        np.load(
            alternate_dir
            / f"{smoke_name}.token_nll_f32.npy"
        ),
        dtype=np.float32,
    )
    nll = (nll_a + nll_b) / 2.0

    full_rows = read_jsonl(
        DATA_ROOT / split_spec["data"]
    )
    full_scores = np.asarray(
        np.load(
            RM_CACHE
            / f"{split_spec['rm']}.scores_f32.npy"
        ),
        dtype=np.float32,
    )

    if len(full_rows) != len(full_scores):
        raise RuntimeError(
            f"{display_name}: 原始行数与 RM 分数不一致"
        )

    lookup = {
        candidate_key(row): (row, float(score))
        for row, score in zip(
            full_rows,
            full_scores,
        )
    }

    rows = []
    rm_scores = []

    for key in reference_keys:
        if key not in lookup:
            raise KeyError(
                f"{display_name}: 找不到候选 {key}"
            )
        row, score = lookup[key]
        rows.append(row)
        rm_scores.append(score)

    rm_scores = np.asarray(
        rm_scores,
        dtype=np.float32,
    )

    if not np.array_equal(
        labels,
        np.asarray(
            [int(row["label"]) for row in rows],
            dtype=np.int8,
        ),
    ):
        raise RuntimeError(
            f"{display_name}: 标签映射不一致"
        )

    groups = defaultdict(list)
    for index, item in enumerate(metadata):
        groups[str(item["question_uid"])].append(index)

    dataset = {
        "name": display_name,
        "domain": domain_name,
        "family": domain_spec["family"],
        "rows": rows,
        "metadata": metadata,
        "labels": labels,
        "rm_scores": rm_scores,
        "hidden": hidden,
        "nll": nll,
        "prompt_agreement": prompt_agreement,
        "groups": dict(groups),
    }

    build_questions(dataset)
    return dataset


def build_questions(dataset):
    questions = []
    mixed_clusters = 0

    for uid, indices_list in dataset["groups"].items():
        indices = np.asarray(
            indices_list,
            dtype=np.int64,
        )
        rm = dataset["rm_scores"][indices]
        labels = dataset["labels"][indices]
        nll = dataset["nll"][indices]
        hidden = dataset["hidden"][indices]
        prompt_agreement = (
            dataset["prompt_agreement"][indices]
        )

        rm_relative = robust_z(rm)
        mean_nll_relative = robust_z(nll[:, 0])
        last8_nll_relative = robust_z(nll[:, 3])

        clusters = defaultdict(list)

        for local_index, global_index in enumerate(indices):
            answer, _ = extract_answer(
                str(
                    dataset["rows"][
                        int(global_index)
                    ]["solution_text"]
                ),
                dataset["family"],
            )

            if answer is None:
                answer = (
                    f"__unparsed_{int(global_index)}"
                )

            clusters[answer].append(local_index)

        raw_local = int(np.argmax(rm))
        raw_answer = next(
            answer
            for answer, members in clusters.items()
            if raw_local in members
        )

        cluster_items = []
        total_candidates = len(indices)

        for answer, member_list in clusters.items():
            members = np.asarray(
                member_list,
                dtype=np.int64,
            )
            member_labels = labels[members]
            unique_labels = set(
                member_labels.tolist()
            )

            if len(unique_labels) > 1:
                mixed_clusters += 1

            cluster_label = int(
                np.mean(member_labels) >= 0.5
            )

            member_hidden = hidden[members]
            centroid = np.mean(
                member_hidden,
                axis=0,
            )
            centroid_norm = float(
                np.linalg.norm(centroid)
            )

            if len(members) > 1:
                summed = np.sum(
                    member_hidden,
                    axis=0,
                )
                pair_cosine = float(
                    (
                        np.dot(summed, summed)
                        - len(members)
                    )
                    / (
                        len(members)
                        * (len(members) - 1)
                    )
                )
            else:
                pair_cosine = 0.0

            other_members = np.asarray(
                [
                    local
                    for local in range(total_candidates)
                    if local not in set(members.tolist())
                ],
                dtype=np.int64,
            )

            if (
                len(other_members) > 0
                and centroid_norm > 1e-8
            ):
                other_centroid = np.mean(
                    hidden[other_members],
                    axis=0,
                )
                other_norm = float(
                    np.linalg.norm(other_centroid)
                )
                if other_norm > 1e-8:
                    cosine = float(
                        np.dot(
                            centroid,
                            other_centroid,
                        )
                        / (
                            centroid_norm
                            * other_norm
                        )
                    )
                    separation = 1.0 - cosine
                else:
                    separation = 0.0
            else:
                separation = 0.0

            unique_solutions = {
                canonical_solution(
                    str(
                        dataset["rows"][
                            int(indices[member])
                        ]["solution_text"]
                    )
                )
                for member in members
            }

            member_rm_relative = rm_relative[members]
            member_mean_nll = (
                mean_nll_relative[members]
            )
            member_last8_nll = (
                last8_nll_relative[members]
            )

            features = np.asarray([
                math.log1p(len(members)),
                len(members) / total_candidates,
                math.log1p(len(unique_solutions)),
                float(np.max(member_rm_relative)),
                float(np.mean(member_rm_relative)),
                stable_logmeanexp(
                    member_rm_relative
                ),
                float(np.std(member_rm_relative)),
                float(answer == raw_answer),
                -float(np.mean(member_mean_nll)),
                -float(np.min(member_mean_nll)),
                -float(np.mean(member_last8_nll)),
                centroid_norm,
                pair_cosine,
                separation,
                float(np.mean(
                    prompt_agreement[members]
                )),
            ], dtype=np.float32)

            cluster_items.append({
                "answer": answer,
                "members": members,
                "label": cluster_label,
                "features": features,
                "rm_max": float(np.max(rm[members])),
                "rm_logmass_tau4": (
                    stable_logmeanexp(
                        rm[members] / 4.0
                    )
                    + math.log(len(members))
                ),
            })

        if not any(
            item["label"] == 1
            for item in cluster_items
        ):
            raise RuntimeError(
                f"{dataset['name']} {uid}: 没有正确簇"
            )
        if not any(
            item["label"] == 0
            for item in cluster_items
        ):
            raise RuntimeError(
                f"{dataset['name']} {uid}: 没有错误簇"
            )

        raw_cluster = next(
            index
            for index, item
            in enumerate(cluster_items)
            if raw_local in item["members"]
        )

        questions.append({
            "uid": uid,
            "indices": indices,
            "labels": labels,
            "rm": rm,
            "clusters": cluster_items,
            "raw_local": raw_local,
            "raw_cluster": raw_cluster,
        })

    dataset["questions"] = questions
    dataset["mixed_clusters"] = mixed_clusters


def collect_feature_matrix(dataset):
    arrays = [
        item["features"]
        for question in dataset["questions"]
        for item in question["clusters"]
    ]
    return np.stack(arrays).astype(np.float32)


def training_statistics(dataset, feature_indices):
    matrix = collect_feature_matrix(dataset)[
        :, feature_indices
    ]
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    std = np.maximum(std, 1e-4)
    return mean.astype(np.float32), std.astype(np.float32)


def build_pair_differences(
    dataset,
    feature_indices,
    mean,
    std,
    seed,
):
    rng = np.random.default_rng(seed)
    questions = dataset["questions"]

    sampled = rng.choice(
        len(questions),
        size=len(questions),
        replace=True,
    )

    differences = []

    for question_index in sampled:
        clusters = questions[int(question_index)][
            "clusters"
        ]

        positives = [
            item for item in clusters
            if item["label"] == 1
        ]
        negatives = [
            item for item in clusters
            if item["label"] == 0
        ]

        for positive in positives:
            for negative in negatives:
                positive_features = (
                    positive["features"][
                        feature_indices
                    ] - mean
                ) / std
                negative_features = (
                    negative["features"][
                        feature_indices
                    ] - mean
                ) / std

                differences.append(
                    positive_features
                    - negative_features
                )

    return np.stack(differences).astype(np.float32)


def fit_cluster_model(
    train,
    feature_indices,
    regularization,
    seed,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    mean, std = training_statistics(
        train,
        feature_indices,
    )
    differences = build_pair_differences(
        train,
        feature_indices,
        mean,
        std,
        seed,
    )

    tensor = torch.from_numpy(
        differences
    ).to(DEVICE)

    weights = torch.nn.Parameter(
        torch.zeros(
            len(feature_indices),
            device=DEVICE,
        )
    )

    optimizer = torch.optim.Adam(
        [weights],
        lr=0.05,
    )

    for _ in range(400):
        margins = tensor @ weights
        ranking_loss = F.softplus(-margins).mean()
        penalty = (
            regularization
            * weights.square().mean()
        )
        loss = ranking_loss + penalty

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    result = {
        "weights": weights.detach().cpu().numpy(),
        "mean": mean,
        "std": std,
        "regularization": regularization,
        "seed": seed,
    }

    del tensor, weights, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def model_cluster_scores(
    question,
    feature_indices,
    models,
):
    matrix = np.stack([
        item["features"][feature_indices]
        for item in question["clusters"]
    ]).astype(np.float32)

    all_scores = []

    for model in models:
        normalized = (
            matrix - model["mean"]
        ) / model["std"]
        all_scores.append(
            normalized @ model["weights"]
        )

    return np.mean(
        np.stack(all_scores),
        axis=0,
    )


def candidate_scores_from_clusters(
    question,
    cluster_scores,
    threshold,
):
    cluster_scores = np.asarray(
        cluster_scores,
        dtype=np.float32,
    ).copy()

    proposal = int(np.argmax(cluster_scores))
    raw_cluster = question["raw_cluster"]

    if proposal != raw_cluster:
        advantage = float(
            cluster_scores[proposal]
            - cluster_scores[raw_cluster]
        )
        if advantage <= threshold:
            cluster_scores[raw_cluster] = (
                float(np.max(cluster_scores))
                + 1e-4
            )

    candidate_scores = np.empty(
        len(question["indices"]),
        dtype=np.float32,
    )

    rm_relative = ordinary_z(question["rm"])

    for cluster_index, cluster in enumerate(
        question["clusters"]
    ):
        for member in cluster["members"]:
            candidate_scores[int(member)] = (
                cluster_scores[cluster_index]
                + 1e-4 * rm_relative[int(member)]
            )

    return candidate_scores


def evaluate_raw(dataset):
    labels = []
    scores = []
    question_ids = []

    for question in dataset["questions"]:
        labels.extend(question["labels"].tolist())
        scores.extend(question["rm"].tolist())
        question_ids.extend(
            [question["uid"]] * len(question["labels"])
        )

    return ranking_metrics(
        np.asarray(labels),
        np.asarray(scores),
        question_ids,
        np.asarray(scores),
    )


def ranking_metrics(
    labels,
    scores,
    question_ids,
    raw_scores,
):
    groups = defaultdict(list)
    for index, uid in enumerate(question_ids):
        groups[str(uid)].append(index)

    top1 = []
    pair = []
    raw_correct_count = 0
    raw_wrong_count = 0
    damage = 0
    correction = 0
    switches = 0

    for indices_list in groups.values():
        indices = np.asarray(
            indices_list,
            dtype=np.int64,
        )
        group_labels = labels[indices]
        group_scores = scores[indices]
        group_raw = raw_scores[indices]

        selected = int(np.argmax(group_scores))
        raw_selected = int(np.argmax(group_raw))

        selected_ok = bool(
            group_labels[selected] == 1
        )
        raw_ok = bool(
            group_labels[raw_selected] == 1
        )

        top1.append(float(selected_ok))
        switches += int(selected != raw_selected)

        if raw_ok:
            raw_correct_count += 1
            damage += int(not selected_ok)
        else:
            raw_wrong_count += 1
            correction += int(selected_ok)

        positive = group_scores[
            group_labels == 1
        ]
        negative = group_scores[
            group_labels == 0
        ]
        pair.append(float(np.mean(
            positive[:, None]
            > negative[None, :]
        )))

    return {
        "questions": len(groups),
        "top1": float(np.mean(top1)),
        "pair_macro_strict": float(np.mean(pair)),
        "damage_rate": (
            damage / max(raw_correct_count, 1)
        ),
        "correction_rate": (
            correction / max(raw_wrong_count, 1)
        ),
        "switch_rate": switches / len(groups),
        "net_corrected_questions": (
            correction - damage
        ),
    }


def evaluate_cluster_model(
    dataset,
    feature_indices,
    models,
    beta,
    threshold,
):
    labels = []
    method_scores = []
    raw_scores = []
    question_ids = []

    for question in dataset["questions"]:
        learned = model_cluster_scores(
            question,
            feature_indices,
            models,
        )
        learned = ordinary_z(learned)

        base = np.asarray([
            cluster["rm_max"]
            for cluster in question["clusters"]
        ], dtype=np.float32)
        base = ordinary_z(base)

        hybrid = base + beta * learned

        candidate_scores = (
            candidate_scores_from_clusters(
                question,
                hybrid,
                threshold,
            )
        )

        labels.extend(question["labels"].tolist())
        method_scores.extend(candidate_scores.tolist())
        raw_scores.extend(question["rm"].tolist())
        question_ids.extend(
            [question["uid"]] * len(question["labels"])
        )

    return ranking_metrics(
        np.asarray(labels, dtype=np.int8),
        np.asarray(method_scores, dtype=np.float32),
        question_ids,
        np.asarray(raw_scores, dtype=np.float32),
    )


def evaluate_weighted_cluster(dataset):
    labels = []
    method_scores = []
    raw_scores = []
    question_ids = []

    for question in dataset["questions"]:
        cluster_scores = np.asarray([
            cluster["rm_logmass_tau4"]
            for cluster in question["clusters"]
        ], dtype=np.float32)

        candidate_scores = (
            candidate_scores_from_clusters(
                question,
                cluster_scores,
                threshold=0.0,
            )
        )

        labels.extend(question["labels"].tolist())
        method_scores.extend(candidate_scores.tolist())
        raw_scores.extend(question["rm"].tolist())
        question_ids.extend(
            [question["uid"]] * len(question["labels"])
        )

    return ranking_metrics(
        np.asarray(labels, dtype=np.int8),
        np.asarray(method_scores, dtype=np.float32),
        question_ids,
        np.asarray(raw_scores, dtype=np.float32),
    )


def choose_configuration(
    train,
    pilot,
    feature_indices,
):
    fitted = {}
    grid = []

    for regularization in REGULARIZATION:
        models = [
            fit_cluster_model(
                train,
                feature_indices,
                regularization,
                seed,
            )
            for seed in SEEDS
        ]
        fitted[regularization] = models

        for beta in BETA_GRID:
            for threshold in THRESHOLD_GRID:
                metrics = evaluate_cluster_model(
                    pilot,
                    feature_indices,
                    models,
                    beta,
                    threshold,
                )
                grid.append({
                    "regularization": regularization,
                    "beta": beta,
                    "threshold": threshold,
                    **metrics,
                })

    eligible = [
        item for item in grid
        if item["damage_rate"]
        <= PILOT_DAMAGE_LIMIT
    ]

    if eligible:
        pool = eligible
        fallback = False
    else:
        minimum_damage = min(
            item["damage_rate"]
            for item in grid
        )
        pool = [
            item for item in grid
            if item["damage_rate"]
            == minimum_damage
        ]
        fallback = True

    selected = max(
        pool,
        key=lambda item: (
            item["top1"],
            item["pair_macro_strict"],
            -item["damage_rate"],
            -item["switch_rate"],
        ),
    ).copy()

    selected["damage_constraint_fallback"] = fallback

    return (
        selected,
        fitted[selected["regularization"]],
        grid,
    )


def metric_with_delta(metrics, raw):
    result = dict(metrics)
    result["raw_top1"] = raw["top1"]
    result["raw_pair_macro_strict"] = raw[
        "pair_macro_strict"
    ]
    result["top1_delta"] = (
        metrics["top1"] - raw["top1"]
    )
    result["pair_delta"] = (
        metrics["pair_macro_strict"]
        - raw["pair_macro_strict"]
    )
    return result


def run_domain(domain_name, spec):
    print()
    print("=" * 76)
    print(
        f"{domain_name}: "
        f"{spec['model']} / "
        f"{spec['hidden_layer_name']}",
        flush=True,
    )

    train = load_smoke_dataset(
        domain_name,
        spec,
        spec["train"],
        f"{domain_name}_TRAIN",
    )
    pilot = load_smoke_dataset(
        domain_name,
        spec,
        spec["pilot"],
        f"{domain_name}_PILOT",
    )
    tests = {
        name: load_smoke_dataset(
            domain_name,
            spec,
            split_spec,
            name,
        )
        for name, split_spec
        in spec["tests"].items()
    }

    print(
        f"训练问题={len(train['questions'])}, "
        f"Pilot问题={len(pilot['questions'])}, "
        f"混合答案簇="
        f"{train['mixed_clusters'] + pilot['mixed_clusters']}"
    )

    result = {
        "model": spec["model"],
        "hidden_layer": spec[
            "hidden_layer_name"
        ],
        "feature_names": FEATURE_NAMES,
        "pilot_damage_limit": PILOT_DAMAGE_LIMIT,
        "baselines": {},
        "ablations": {},
    }

    all_evaluation_sets = {
        "PILOT": pilot,
        **tests,
    }

    for name, dataset in all_evaluation_sets.items():
        raw = evaluate_raw(dataset)
        weighted = evaluate_weighted_cluster(dataset)

        result["baselines"][name] = {
            "raw_rm": raw,
            "rm_weighted_cluster_tau4": (
                metric_with_delta(
                    weighted,
                    raw,
                )
            ),
        }

    for ablation_name, feature_indices in ABLATIONS.items():
        print()
        print("消融：", ablation_name)

        selected, models, grid = (
            choose_configuration(
                train,
                pilot,
                feature_indices,
            )
        )

        print(
            "  Pilot选择："
            f"reg={selected['regularization']}, "
            f"beta={selected['beta']}, "
            f"threshold={selected['threshold']}, "
            f"Top1={selected['top1']:.6f}, "
            f"Pair={selected['pair_macro_strict']:.6f}, "
            f"Damage={selected['damage_rate']:.6f}"
        )

        evaluations = {}

        for name, dataset in all_evaluation_sets.items():
            raw = evaluate_raw(dataset)
            metrics = evaluate_cluster_model(
                dataset,
                feature_indices,
                models,
                selected["beta"],
                selected["threshold"],
            )
            evaluations[name] = metric_with_delta(
                metrics,
                raw,
            )

            item = evaluations[name]
            print(
                f"  {name}: "
                f"Top1={item['raw_top1']:.6f}"
                f"->{item['top1']:.6f} "
                f"({item['top1_delta']:+.6f}), "
                f"Pair={item['raw_pair_macro_strict']:.6f}"
                f"->{item['pair_macro_strict']:.6f} "
                f"({item['pair_delta']:+.6f}), "
                f"Damage={item['damage_rate']:.6f}"
            )

        result["ablations"][ablation_name] = {
            "feature_indices": feature_indices,
            "features": [
                FEATURE_NAMES[index]
                for index in feature_indices
            ],
            "selected": selected,
            "evaluations": evaluations,
            "grid": grid,
        }

    return result


def main():
    started = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    results = {
        domain_name: run_domain(
            domain_name,
            spec,
        )
        for domain_name, spec in DOMAINS.items()
    }

    output = {
        "version": "answer_cluster_evidence_smoke_v1",
        "scope": (
            "small deterministic subsets; "
            "train fits models, pilot selects configs, "
            "tests are evaluation only"
        ),
        "design": {
            "supervision_unit": "normalized answer cluster",
            "candidate_selection_within_cluster": (
                "highest original RM"
            ),
            "prompt_hidden_combination": (
                "unit-normalized mean of two prompts"
            ),
            "seeds": SEEDS,
            "regularization_grid": REGULARIZATION,
            "beta_grid": BETA_GRID,
            "threshold_grid": THRESHOLD_GRID,
            "pilot_damage_limit": PILOT_DAMAGE_LIMIT,
        },
        "results": results,
        "elapsed_seconds": round(
            time.time() - started,
            3,
        ),
        "peak_gpu_gb": (
            round(
                torch.cuda.max_memory_allocated()
                / (1024 ** 3),
                3,
            )
            if torch.cuda.is_available()
            else 0.0
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print("答案簇证据模型冒烟实验完成。")
    print("结果：", OUTPUT)
    print("耗时秒：", output["elapsed_seconds"])
    print("显存峰值GB：", output["peak_gpu_gb"])


if __name__ == "__main__":
    main()

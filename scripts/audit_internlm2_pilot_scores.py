from pathlib import Path
from collections import defaultdict
import hashlib
import json

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]

OUTPUT = (
    ROOT / "data/manifests/"
    "internlm2_pilot_score_audit_v1.json"
)

SPECS = {
    "GSM8K_PILOT": {
        "data": (
            "data/processed/prototype_v2/"
            "gsm_pilot_validation.jsonl"
        ),
        "internlm2": (
            "data/cache/reward_scores_full_v1/"
            "InternLM2-1.8B-Reward/"
            "gsm_pilot.scores_f32.npy"
        ),
        "skywork": (
            "data/cache/trajectory_features_v1/"
            "Skywork-Reward-V2-Qwen3-1.7B/"
            "layer_28/gsm_pilot.scores_f32.npy"
        ),
    },
    "MATH_PILOT": {
        "data": (
            "data/processed/prototype_v2/"
            "math_pilot_validation.jsonl"
        ),
        "internlm2": (
            "data/cache/reward_scores_full_v1/"
            "InternLM2-1.8B-Reward/"
            "math_pilot.scores_f32.npy"
        ),
        "skywork": (
            "data/cache/trajectory_features_v1/"
            "Skywork-Reward-V2-Qwen3-1.7B/"
            "layer_28/math_pilot.scores_f32.npy"
        ),
    },
}


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
        "unique_exact_values": int(
            len(np.unique(values))
        ),
    }


def correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)

    if (
        len(left) < 2
        or np.std(left) == 0
        or np.std(right) == 0
    ):
        return None

    return float(np.corrcoef(left, right)[0, 1])


def signal_metrics(scores, labels, groups):
    top1 = []
    optimistic_top1 = []
    top_ties = 0
    all_equal = 0
    within_stds = []

    pair_wins = 0
    pair_losses = 0
    pair_ties = 0

    strict_macro = []
    half_tie_macro = []

    composition = {
        "all_wrong": 0,
        "mixed": 0,
        "all_correct": 0,
    }

    for uid, indices_list in groups.items():
        indices = np.asarray(
            indices_list,
            dtype=np.int64,
        )
        question_scores = scores[indices]
        question_labels = labels[indices]

        maximum = np.max(question_scores)
        tied = np.flatnonzero(
            question_scores == maximum
        )

        selected = int(tied[0])
        top1.append(
            int(question_labels[selected] == 1)
        )
        optimistic_top1.append(
            int(np.any(question_labels[tied] == 1))
        )

        top_ties += int(len(tied) > 1)
        all_equal += int(
            len(np.unique(question_scores)) == 1
        )
        within_stds.append(
            float(np.std(question_scores))
        )

        positives = question_scores[
            question_labels == 1
        ]
        negatives = question_scores[
            question_labels == 0
        ]

        if len(positives) == 0:
            composition["all_wrong"] += 1
        elif len(negatives) == 0:
            composition["all_correct"] += 1
        else:
            composition["mixed"] += 1

            differences = (
                positives[:, None]
                - negatives[None, :]
            )

            wins = int(np.sum(differences > 0))
            losses = int(np.sum(differences < 0))
            ties = int(np.sum(differences == 0))
            pairs = wins + losses + ties

            pair_wins += wins
            pair_losses += losses
            pair_ties += ties

            strict_macro.append(
                wins / pairs
            )
            half_tie_macro.append(
                (wins + 0.5 * ties) / pairs
            )

    total_pairs = (
        pair_wins + pair_losses + pair_ties
    )
    questions = len(groups)

    correct_scores = scores[labels == 1]
    incorrect_scores = scores[labels == 0]

    auc = None
    if len(np.unique(labels)) == 2:
        auc = float(
            roc_auc_score(labels, scores)
        )

    return {
        "statistics": statistics(scores),
        "candidate_accuracy": float(
            np.mean(labels)
        ),
        "correct_score_mean": float(
            np.mean(correct_scores)
        ),
        "incorrect_score_mean": float(
            np.mean(incorrect_scores)
        ),
        "mean_correct_minus_incorrect": float(
            np.mean(correct_scores)
            - np.mean(incorrect_scores)
        ),
        "pooled_auc_diagnostic": auc,
        "top1": float(np.mean(top1)),
        "top1_correct_questions": int(sum(top1)),
        "optimistic_tie_top1": float(
            np.mean(optimistic_top1)
        ),
        "top_tie_rate": float(
            top_ties / questions
        ),
        "all_equal_question_rate": float(
            all_equal / questions
        ),
        "mean_within_question_std": float(
            np.mean(within_stds)
        ),
        "pair_questions": int(
            len(strict_macro)
        ),
        "pair_count": int(total_pairs),
        "pair_wins": int(pair_wins),
        "pair_losses": int(pair_losses),
        "pair_ties": int(pair_ties),
        "pair_exact_tie_rate": float(
            pair_ties / max(total_pairs, 1)
        ),
        "pair_micro_strict": float(
            pair_wins / max(total_pairs, 1)
        ),
        "pair_micro_half_tie": float(
            (
                pair_wins
                + 0.5 * pair_ties
            )
            / max(total_pairs, 1)
        ),
        "pair_macro_strict": float(
            np.mean(strict_macro)
        ),
        "pair_macro_half_tie": float(
            np.mean(half_tie_macro)
        ),
        "composition": composition,
    }


def analyze(name, spec):
    data_path = ROOT / spec["data"]
    internlm2_path = ROOT / spec["internlm2"]
    skywork_path = ROOT / spec["skywork"]

    rows = read_jsonl(data_path)
    internlm2 = np.asarray(
        np.load(internlm2_path),
        dtype=np.float64,
    )
    skywork = np.asarray(
        np.load(skywork_path),
        dtype=np.float64,
    )

    if not (
        len(rows) == len(internlm2) == len(skywork)
    ):
        raise RuntimeError(
            f"{name}: 行数不一致："
            f"{len(rows)}, "
            f"{len(internlm2)}, "
            f"{len(skywork)}"
        )

    if not (
        np.all(np.isfinite(internlm2))
        and np.all(np.isfinite(skywork))
    ):
        raise RuntimeError(
            f"{name}: 存在非有限分数"
        )

    labels = np.asarray(
        [int(row["label"]) for row in rows],
        dtype=np.int64,
    )
    lengths = np.asarray(
        [
            len(str(row["solution_text"]))
            for row in rows
        ],
        dtype=np.float64,
    )

    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["question_uid"])].append(
            index
        )

    internlm2_metrics = signal_metrics(
        internlm2,
        labels,
        groups,
    )
    skywork_metrics = signal_metrics(
        skywork,
        labels,
        groups,
    )

    within_rank_correlations = []
    top1_agreement = 0

    complementarity = {
        "both_correct": 0,
        "both_wrong": 0,
        "internlm2_only_correct": 0,
        "skywork_only_correct": 0,
    }

    for uid, indices_list in groups.items():
        indices = np.asarray(
            indices_list,
            dtype=np.int64,
        )

        internlm2_values = internlm2[indices]
        skywork_values = skywork[indices]
        question_labels = labels[indices]

        internlm2_choice = int(
            np.argmax(internlm2_values)
        )
        skywork_choice = int(
            np.argmax(skywork_values)
        )

        top1_agreement += int(
            internlm2_choice == skywork_choice
        )

        internlm2_ok = bool(
            question_labels[internlm2_choice] == 1
        )
        skywork_ok = bool(
            question_labels[skywork_choice] == 1
        )

        if internlm2_ok and skywork_ok:
            complementarity["both_correct"] += 1
        elif internlm2_ok:
            complementarity[
                "internlm2_only_correct"
            ] += 1
        elif skywork_ok:
            complementarity[
                "skywork_only_correct"
            ] += 1
        else:
            complementarity["both_wrong"] += 1

        if (
            np.std(internlm2_values) > 0
            and np.std(skywork_values) > 0
        ):
            value = correlation(
                rankdata(internlm2_values),
                rankdata(skywork_values),
            )
            if value is not None:
                within_rank_correlations.append(
                    value
                )

    result = {
        "questions": len(groups),
        "candidates": len(rows),
        "internlm2": internlm2_metrics,
        "skywork_qwen3_1p7b": skywork_metrics,
        "cross_reward_model": {
            "pooled_pearson": correlation(
                internlm2,
                skywork,
            ),
            "pooled_spearman": correlation(
                rankdata(internlm2),
                rankdata(skywork),
            ),
            "within_question_spearman_macro": (
                float(np.mean(
                    within_rank_correlations
                ))
                if within_rank_correlations
                else None
            ),
            "within_question_spearman_questions": (
                len(within_rank_correlations)
            ),
            "top1_choice_agreement_rate": float(
                top1_agreement / len(groups)
            ),
            "complementarity": complementarity,
        },
        "bias_diagnostics": {
            "internlm2_character_length_pearson": (
                correlation(internlm2, lengths)
            ),
            "skywork_character_length_pearson": (
                correlation(skywork, lengths)
            ),
        },
        "files": {
            "data": spec["data"],
            "data_sha256": sha256_file(data_path),
            "internlm2_scores": spec["internlm2"],
            "internlm2_scores_sha256": sha256_file(
                internlm2_path
            ),
            "skywork_scores": spec["skywork"],
            "skywork_scores_sha256": sha256_file(
                skywork_path
            ),
        },
    }

    print()
    print("=" * 76)
    print(name)
    print(json.dumps(
        {
            "questions": result["questions"],
            "candidates": result["candidates"],
            "internlm2": result["internlm2"],
            "skywork_qwen3_1p7b": {
                "top1": (
                    result[
                        "skywork_qwen3_1p7b"
                    ]["top1"]
                ),
                "pair_macro_strict": (
                    result[
                        "skywork_qwen3_1p7b"
                    ]["pair_macro_strict"]
                ),
                "pair_macro_half_tie": (
                    result[
                        "skywork_qwen3_1p7b"
                    ]["pair_macro_half_tie"]
                ),
            },
            "cross_reward_model": (
                result["cross_reward_model"]
            ),
            "bias_diagnostics": (
                result["bias_diagnostics"]
            ),
        },
        ensure_ascii=False,
        indent=2,
    ))

    return result


def main():
    print("===== InternLM2 Pilot 分数审计 =====")
    print(
        "标签仅用于评分完成后的 Pilot 诊断；"
        "不参与奖励模型评分。"
    )

    results = {
        name: analyze(name, spec)
        for name, spec in SPECS.items()
    }

    macro = {
        "internlm2_top1": float(np.mean([
            result["internlm2"]["top1"]
            for result in results.values()
        ])),
        "internlm2_pair_macro_strict": float(
            np.mean([
                result["internlm2"][
                    "pair_macro_strict"
                ]
                for result in results.values()
            ])
        ),
        "internlm2_pair_macro_half_tie": float(
            np.mean([
                result["internlm2"][
                    "pair_macro_half_tie"
                ]
                for result in results.values()
            ])
        ),
        "skywork_top1": float(np.mean([
            result["skywork_qwen3_1p7b"][
                "top1"
            ]
            for result in results.values()
        ])),
        "top1_choice_agreement": float(
            np.mean([
                result["cross_reward_model"][
                    "top1_choice_agreement_rate"
                ]
                for result in results.values()
            ])
        ),
    }

    output = {
        "version": (
            "internlm2_pilot_score_audit_v1"
        ),
        "reward_model": (
            "RLHFlow/"
            "InternLM2-1.8B-Reward"
        ),
        "score_definition": (
            "official output.score / output.logits"
        ),
        "score_dtype": (
            "model inference bfloat16, "
            "cached float32"
        ),
        "labels_used_for_scoring": False,
        "labels_used_only_for_posthoc_pilot_audit": (
            True
        ),
        "results": results,
        "macro": macro,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("===== Pilot 宏平均 =====")
    print(json.dumps(
        macro,
        ensure_ascii=False,
        indent=2,
    ))
    print()
    print("结果：", OUTPUT)


if __name__ == "__main__":
    main()

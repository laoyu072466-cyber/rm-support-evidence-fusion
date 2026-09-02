from pathlib import Path
from collections import defaultdict
import json

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


ROOT = Path("/root/autodl-tmp/rm_traj_project")
SPECS = [
    (
        "GSM8K / Qwen2-1.5B",
        ROOT / "data/processed/prototype_v2/gsm_pilot_validation.jsonl",
        ROOT / "data/cache/cisc_ptrue_smoke_v1/GSM8K_PILOT_Qwen2_1p5B.jsonl",
    ),
    (
        "MATH / Qwen2-7B",
        ROOT / "data/processed/prototype_v2/math_pilot_validation.jsonl",
        ROOT / "data/cache/cisc_ptrue_smoke_v1/MATH_PILOT_Qwen2_7B.jsonl",
    ),
]


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def audit(name, dataset_path, score_path):
    labels = {
        (row["question_uid"], int(row["candidate_index"])): int(row["label"])
        for row in read_jsonl(dataset_path)
    }
    groups = defaultdict(list)
    for row in read_jsonl(score_path):
        key = (row["question_uid"], int(row["candidate_index"]))
        if key not in labels:
            raise RuntimeError(f"无法匹配候选：{key}")
        groups[key[0]].append({
            "candidate_index": key[1],
            "label": labels[key],
            "p_true": float(row["p_true_binary_normalized"]),
        })

    joined = [row for rows in groups.values() for row in rows]
    correct = [row["p_true"] for row in joined if row["label"] == 1]
    incorrect = [row["p_true"] for row in joined if row["label"] == 0]
    wins = losses = ties = top_correct = mixed = 0
    macro = []
    within_stds = []
    question_means = []
    details = []

    for uid, rows in groups.items():
        rows.sort(key=lambda row: row["candidate_index"])
        positives = [row for row in rows if row["label"] == 1]
        negatives = [row for row in rows if row["label"] == 0]
        values = np.asarray([row["p_true"] for row in rows], dtype=np.float64)
        within_stds.append(float(values.std()))
        question_means.append(float(values.mean()))
        top = max(rows, key=lambda row: (row["p_true"], -row["candidate_index"]))
        top_correct += top["label"]

        local_wins = local_losses = local_ties = 0
        for positive in positives:
            for negative in negatives:
                delta = positive["p_true"] - negative["p_true"]
                local_wins += int(delta > 0)
                local_losses += int(delta < 0)
                local_ties += int(delta == 0)
        local_pairs = local_wins + local_losses + local_ties
        if positives and negatives:
            mixed += 1
            wins += local_wins
            losses += local_losses
            ties += local_ties
            macro.append((local_wins + 0.5 * local_ties) / local_pairs)

        details.append({
            "uid": uid,
            "correct": len(positives),
            "incorrect": len(negatives),
            "p_min": round(float(values.min()), 6),
            "p_max": round(float(values.max()), 6),
            "p_std": round(float(values.std()), 6),
            "top_correct": bool(top["label"]),
            "wqd": (
                round((local_wins + 0.5 * local_ties) / local_pairs, 6)
                if local_pairs else None
            ),
        })

    total_pairs = wins + losses + ties
    auc = None
    if roc_auc_score is not None and correct and incorrect:
        auc = float(roc_auc_score(
            [row["label"] for row in joined],
            [row["p_true"] for row in joined],
        ))

    summary = {
        "questions": len(groups),
        "candidates": len(joined),
        "correct_candidates": len(correct),
        "incorrect_candidates": len(incorrect),
        "candidate_accuracy": float(np.mean([row["label"] for row in joined])),
        "correct_p_true_mean": float(np.mean(correct)) if correct else None,
        "incorrect_p_true_mean": float(np.mean(incorrect)) if incorrect else None,
        "pooled_auc_diagnostic": auc,
        "mixed_questions": mixed,
        "within_question_pairs": total_pairs,
        "pair_wins": wins,
        "pair_losses": losses,
        "pair_ties": ties,
        "wqd_micro_half_tie": (
            (wins + 0.5 * ties) / total_pairs if total_pairs else None
        ),
        "wqd_macro_half_tie": float(np.mean(macro)) if macro else None,
        "confidence_top1": top_correct / len(groups),
        "mean_within_question_std": float(np.mean(within_stds)),
        "median_within_question_std": float(np.median(within_stds)),
        "between_question_mean_std": float(np.std(question_means)),
    }

    print("\n" + "=" * 76)
    print(name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("逐题诊断：")
    for detail in details:
        print(detail)


def main():
    print("===== CISC P(True) smoke WQD audit =====")
    print("标签仅用于 Pilot 诊断，不参与 P(True) 评分。")
    for spec in SPECS:
        audit(*spec)


if __name__ == "__main__":
    main()

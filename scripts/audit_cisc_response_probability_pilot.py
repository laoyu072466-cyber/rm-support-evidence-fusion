from pathlib import Path
from collections import defaultdict
import json
import math

import numpy as np


ROOT = Path("/root/autodl-tmp/rm_traj_project")
CACHE = (
    ROOT / "data/cache/generator_cluster_features_v1"
)
OUTPUT = (
    ROOT / "data/manifests/"
    "cisc_response_probability_pilot_audit_v1.json"
)

SPECS = {
    "GSM8K_PILOT_Qwen2_1p5B": (
        CACHE / "Qwen2-1.5B/gsm_pilot",
    ),
    "MATH_PILOT_Qwen2_7B": (
        CACHE / "Qwen2-7B/math_pilot",
    ),
}


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def audit(name, base):
    nll = np.load(
        str(base) + ".token_nll_f32.npy",
        mmap_mode="r",
    )
    labels = np.load(
        str(base) + ".labels_i8.npy",
        mmap_mode="r",
    ).astype(np.int64)
    metadata = read_jsonl(
        Path(str(base) + ".metadata.jsonl")
    )

    if nll.ndim != 2 or nll.shape[1] != 5:
        raise RuntimeError(
            f"{name}: unexpected NLL shape {nll.shape}"
        )

    if not (
        len(nll) == len(labels) == len(metadata)
    ):
        raise RuntimeError(
            f"{name}: cache lengths do not match"
        )

    metadata_labels = np.asarray(
        [int(row["label"]) for row in metadata],
        dtype=np.int64,
    )
    if not np.array_equal(labels, metadata_labels):
        raise RuntimeError(
            f"{name}: metadata/label alignment failed"
        )

    mean_nll = np.asarray(
        nll[:, 0],
        dtype=np.float64,
    )
    confidence = np.exp(-mean_nll)

    groups = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[row["question_uid"]].append(index)

    pair_wins = 0
    pair_losses = 0
    pair_ties = 0
    per_question_wqd = []
    confidence_top1_correct = 0
    question_stds = []

    for uid, indices in groups.items():
        indices = np.asarray(
            indices,
            dtype=np.int64,
        )
        local_labels = labels[indices]
        local_conf = confidence[indices]

        top_local = int(
            np.argmax(local_conf)
        )
        confidence_top1_correct += int(
            local_labels[top_local] == 1
        )
        question_stds.append(
            float(np.std(local_conf))
        )

        positive = local_conf[
            local_labels == 1
        ]
        negative = local_conf[
            local_labels == 0
        ]

        if not len(positive) or not len(negative):
            continue

        local_wins = 0
        local_losses = 0
        local_ties = 0

        for pos in positive:
            for neg in negative:
                if math.isclose(
                    float(pos),
                    float(neg),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    local_ties += 1
                elif pos > neg:
                    local_wins += 1
                else:
                    local_losses += 1

        pair_wins += local_wins
        pair_losses += local_losses
        pair_ties += local_ties

        local_pairs = (
            local_wins
            + local_losses
            + local_ties
        )
        per_question_wqd.append(
            (
                local_wins
                + 0.5 * local_ties
            )
            / local_pairs
        )

    total_pairs = (
        pair_wins
        + pair_losses
        + pair_ties
    )

    result = {
        "name": name,
        "questions": len(groups),
        "candidates": len(labels),
        "correct_candidates": int(
            np.sum(labels == 1)
        ),
        "incorrect_candidates": int(
            np.sum(labels == 0)
        ),
        "mean_nll_positive": float(
            np.mean(mean_nll[labels == 1])
        ),
        "mean_nll_negative": float(
            np.mean(mean_nll[labels == 0])
        ),
        "mean_response_probability_positive": float(
            np.mean(confidence[labels == 1])
        ),
        "mean_response_probability_negative": float(
            np.mean(confidence[labels == 0])
        ),
        "mixed_questions": len(
            per_question_wqd
        ),
        "within_question_pairs": total_pairs,
        "pair_wins": pair_wins,
        "pair_losses": pair_losses,
        "pair_ties": pair_ties,
        "wqd_micro_half_tie": (
            pair_wins + 0.5 * pair_ties
        ) / total_pairs,
        "wqd_macro_half_tie": float(
            np.mean(per_question_wqd)
        ),
        "confidence_top1": (
            confidence_top1_correct
            / len(groups)
        ),
        "mean_within_question_confidence_std": float(
            np.mean(question_stds)
        ),
        "response_probability_definition": (
            "exp(-token_nll_f32[:, 0])"
        ),
        "labels_used_only_for_pilot_diagnostic": True,
    }

    print()
    print("=" * 76)
    print(name)
    print(json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    ))

    return result


def main():
    print(
        "===== CISC Response Probability "
        "完整 Pilot WQD ====="
    )
    print(
        "标签仅用于 Pilot 诊断；"
        "未读取 ID/OOD 测试标签。"
    )

    results = {
        name: audit(name, spec[0])
        for name, spec in SPECS.items()
    }

    output = {
        "version": (
            "cisc_response_probability_"
            "pilot_audit_v1"
        ),
        "official_confidence": (
            "geometric mean response-token "
            "probability"
        ),
        "results": results,
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
    print("结果：", OUTPUT)


if __name__ == "__main__":
    main()

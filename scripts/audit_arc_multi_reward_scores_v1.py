from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import hashlib
import json
import os

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_PATH = (
    ROOT / "outputs/arc_challenge_v1/"
    "qwen2_7b_full_k16_recovered_candidates.jsonl"
)
OUTPUT_PATH = (
    ROOT / "data/manifests/"
    "arc_multi_reward_score_audit_v1.json"
)

EXPECTED_CANDIDATE_SHA256 = (
    "eb72914f82678c4738fab24ea055a62b"
    "00f8c207372d3f3ba85d785591267191"
)

EXPECTED_SPLIT_QUESTIONS = {
    "train": 1119,
    "pilot": 299,
    "test": 1172,
}

MODELS = {
    "qwen3_1p7b": {
        "model_name": (
            "Skywork-Reward-V2-Qwen3-1.7B"
        ),
        "manifest": (
            "data/manifests/arc_reward_scores_v1/"
            "multi_reward_scores_qwen3_1p7b_v1.json"
        ),
        "score_file": (
            "data/cache/arc_reward_scores_v1/"
            "Skywork-Reward-V2-Qwen3-1.7B/"
            "arc_full_recovered.scores_f32.npy"
        ),
        "score_sha256": (
            "e2b92df8becba8fa4d9dd4606d5ea4a"
            "5b89a6ae7eb78aa5529af600b4a5d94b0"
        ),
    },
    "internlm2_1p8b": {
        "model_name": "InternLM2-1.8B-Reward",
        "manifest": (
            "data/manifests/arc_reward_scores_v1/"
            "multi_reward_scores_internlm2_1p8b_v1.json"
        ),
        "score_file": (
            "data/cache/arc_reward_scores_v1/"
            "InternLM2-1.8B-Reward/"
            "arc_full_recovered.scores_f32.npy"
        ),
        "score_sha256": (
            "16785a1da4b7da38884137fedb43c39b"
            "1874763fdc54659d9b7ff1449482a31c"
        ),
    },
    "armorm_8b": {
        "model_name": "ArmoRM-Llama3-8B-v0.1",
        "manifest": (
            "data/manifests/arc_reward_scores_v1/"
            "multi_reward_scores_armorm_8b_v1.json"
        ),
        "score_file": (
            "data/cache/arc_reward_scores_v1/"
            "ArmoRM-Llama3-8B-v0.1/"
            "arc_full_recovered.scores_f32.npy"
        ),
        "score_sha256": (
            "09472611d6e1a4129f18772850c9f16e"
            "d2b38e44d1d8dc0d1cb0d9da1e7388cc"
        ),
    },
}

FORBIDDEN_EXACT_FIELDS = {
    "answerKey",
    "answer_index",
    "answer_label",
    "correct",
    "correct_index",
    "correct_label",
    "gold",
    "gold_answer",
    "ground_truth",
    "is_correct",
    "label",
    "labels",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def score_statistics(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def audit_candidates():
    require(
        CANDIDATE_PATH.exists(),
        f"候选文件不存在：{CANDIDATE_PATH}",
    )

    candidate_sha256 = sha256_file(
        CANDIDATE_PATH
    )
    require(
        candidate_sha256
        == EXPECTED_CANDIDATE_SHA256,
        "恢复候选 SHA256 不一致",
    )

    candidate_count = 0
    question_indices = defaultdict(set)
    question_split = {}
    split_candidates = Counter()
    forbidden_hits = Counter()

    with CANDIDATE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            row = json.loads(line)
            candidate_count += 1

            hits = (
                set(row)
                & FORBIDDEN_EXACT_FIELDS
            )
            for field in hits:
                forbidden_hits[field] += 1

            uid = str(row["question_uid"])
            index = int(row["candidate_index"])
            split = str(row["logical_split"])

            require(
                split in EXPECTED_SPLIT_QUESTIONS,
                f"未知逻辑划分：{split}",
            )
            require(
                0 <= index < 16,
                f"候选编号越界：{uid}/{index}",
            )
            require(
                index
                not in question_indices[uid],
                f"候选身份重复：{uid}/{index}",
            )

            question_indices[uid].add(index)
            split_candidates[split] += 1

            previous = question_split.get(uid)
            require(
                previous is None
                or previous == split,
                f"题目跨逻辑划分：{uid}",
            )
            question_split[uid] = split

    require(
        candidate_count == 41440,
        f"候选数错误：{candidate_count}",
    )
    require(
        len(question_indices) == 2590,
        f"题目数错误：{len(question_indices)}",
    )
    require(
        not forbidden_hits,
        f"候选含标签字段：{dict(forbidden_hits)}",
    )

    expected_indices = set(range(16))
    invalid_questions = [
        uid
        for uid, indices
        in question_indices.items()
        if indices != expected_indices
    ]
    require(
        not invalid_questions,
        "存在候选编号不完整的题目",
    )

    split_questions = Counter(
        question_split.values()
    )
    require(
        dict(split_questions)
        == EXPECTED_SPLIT_QUESTIONS,
        "问题划分数量不一致："
        f"{dict(split_questions)}",
    )

    expected_split_candidates = {
        name: count * 16
        for name, count
        in EXPECTED_SPLIT_QUESTIONS.items()
    }
    require(
        dict(split_candidates)
        == expected_split_candidates,
        "候选划分数量不一致："
        f"{dict(split_candidates)}",
    )

    return {
        "file": str(
            CANDIDATE_PATH.relative_to(ROOT)
        ),
        "sha256": candidate_sha256,
        "questions": len(question_indices),
        "candidates": candidate_count,
        "candidates_per_question": 16,
        "question_splits": dict(
            split_questions
        ),
        "candidate_splits": dict(
            split_candidates
        ),
        "unique_candidate_identities": (
            candidate_count
        ),
        "forbidden_label_fields": {},
    }


def audit_model(model_key, spec):
    manifest_path = ROOT / spec["manifest"]
    score_path = ROOT / spec["score_file"]

    require(
        manifest_path.exists(),
        f"缺少评分清单：{manifest_path}",
    )
    require(
        score_path.exists(),
        f"缺少评分数组：{score_path}",
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    require(
        manifest.get("model_key")
        == model_key,
        f"{model_key}: model_key 不一致",
    )
    require(
        manifest.get("model_name")
        == spec["model_name"],
        f"{model_key}: model_name 不一致",
    )
    require(
        manifest.get(
            "labels_used_for_scoring"
        ) is False,
        f"{model_key}: 标签使用标记错误",
    )
    require(
        manifest.get("completed_splits")
        == ["arc_full_recovered"],
        f"{model_key}: completed_splits 错误",
    )

    scoring = manifest.get("scoring", {})
    require(
        scoring.get("labels_used") is False,
        f"{model_key}: scoring.labels_used 错误",
    )
    require(
        scoring.get(
            "sealed_test_labels_loaded"
        ) is False,
        f"{model_key}: 读取了密封 Test 标签",
    )
    require(
        scoring.get("candidate_sha256")
        == EXPECTED_CANDIDATE_SHA256,
        f"{model_key}: 候选哈希错误",
    )

    split = manifest.get(
        "splits",
        {},
    ).get("arc_full_recovered")
    require(
        isinstance(split, dict),
        f"{model_key}: 缺少 split 记录",
    )
    require(
        split.get("candidates") == 41440,
        f"{model_key}: 候选数量错误",
    )
    require(
        split.get("source_sha256")
        == EXPECTED_CANDIDATE_SHA256,
        f"{model_key}: source_sha256 错误",
    )
    require(
        split.get("score_file")
        == spec["score_file"],
        f"{model_key}: score_file 错误",
    )
    require(
        split.get("score_sha256")
        == spec["score_sha256"],
        f"{model_key}: 清单分数哈希错误",
    )

    actual_score_sha256 = sha256_file(
        score_path
    )
    require(
        actual_score_sha256
        == spec["score_sha256"],
        f"{model_key}: 实际分数哈希错误",
    )

    scores = np.load(
        score_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    require(
        scores.shape == (41440,),
        f"{model_key}: 分数形状错误 "
        f"{scores.shape}",
    )
    require(
        scores.dtype == np.float32,
        f"{model_key}: dtype 错误 "
        f"{scores.dtype}",
    )
    require(
        bool(np.all(np.isfinite(scores))),
        f"{model_key}: 存在非有限分数",
    )

    return {
        "model_key": model_key,
        "model_name": spec["model_name"],
        "model_path": manifest.get(
            "model_path"
        ),
        "manifest_file": spec["manifest"],
        "manifest_sha256": sha256_file(
            manifest_path
        ),
        "score_file": spec["score_file"],
        "score_sha256": actual_score_sha256,
        "statistics": score_statistics(
            scores
        ),
        "labels_used_for_scoring": False,
        "sealed_test_labels_loaded": False,
    }


def main():
    print(
        "===== ARC 多奖励模型分数完整性审计 ====="
    )
    print("标签读取：False")
    print("密封 Test 标签读取：False")

    candidates = audit_candidates()

    models = {
        key: audit_model(key, spec)
        for key, spec in MODELS.items()
    }

    result = {
        "version": (
            "arc_multi_reward_score_audit_v1"
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "labels_loaded": False,
        "sealed_test_labels_loaded": False,
        "candidate_source_tag": (
            "arc-challenge-qwen2-7b-"
            "full-k16-recovered-v1"
        ),
        "candidates": candidates,
        "models": models,
        "checks": {
            "candidate_identity_preserved": True,
            "candidate_sha256_verified": True,
            "three_reward_models_present": True,
            "all_score_arrays_float32": True,
            "all_score_arrays_finite": True,
            "all_score_sha256_verified": True,
            "all_manifests_label_free": True,
            "test_labels_still_sealed": True,
        },
        "decision": (
            "freeze_scores_before_"
            "train_pilot_label_configuration"
        ),
    }

    atomic_json(OUTPUT_PATH, result)

    print(json.dumps(
        {
            "candidates": candidates,
            "models": {
                key: {
                    "model_name": item[
                        "model_name"
                    ],
                    "score_sha256": item[
                        "score_sha256"
                    ],
                    "statistics": item[
                        "statistics"
                    ],
                }
                for key, item in models.items()
            },
            "decision": result["decision"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    print(
        "ARC_MULTI_REWARD_SCORES_FREEZE_READY"
    )
    print("结果：", OUTPUT_PATH)


if __name__ == "__main__":
    main()

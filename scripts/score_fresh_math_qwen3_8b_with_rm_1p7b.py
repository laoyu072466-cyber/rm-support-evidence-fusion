from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np


ROOT = Path("/root/autodl-tmp/rm_traj_project")
sys.path.insert(0, str(ROOT / "scripts"))

import eval_answer_cluster_holdout_rewards as holdout


CANDIDATE_PATH = (
    ROOT
    / "outputs/fresh_math_2026/"
    "qwen3_8b_adaptive_k16_candidates.jsonl"
)
CANDIDATE_MANIFEST_PATH = (
    ROOT
    / "data/manifests/"
    "fresh_math_2026_qwen3_8b_adaptive_k16.json"
)
MODEL_PATH = (
    ROOT
    / "models/reward/"
    "Skywork-Reward-V2-Qwen3-1.7B"
)
CACHE_ROOT = (
    ROOT
    / "data/cache/fresh_math_2026/"
    "qwen3_8b_adaptive_k16_rm_1p7b"
)
SCORES_PATH = (
    CACHE_ROOT
    / "scores_f32.npy"
)
OUTPUT_MANIFEST_PATH = (
    ROOT
    / "data/manifests/"
    "fresh_math_2026_qwen3_8b_rm_1p7b_scores.json"
)

EXPECTED_CANDIDATE_SHA256 = (
    "9af1176c0b092608f413d90889315c03"
    "ef4ad5d3d994ad8746dadd5e131f4257"
)
EXPECTED_GENERATION_CONFIG_SHA256 = (
    "7a10db9ea0106798716971245f19b3641"
    "2d3da56da777c708bfe99028ac2b54a"
)
EXPECTED_CANDIDATES = 1008
EXPECTED_QUESTIONS = 63
EXPECTED_K = 16
JUDGE_NAME = (
    "fresh_qwen3_8b_adaptive_k16_"
    "qwen3_1p7b_9af1176c"
)


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return None


def atomic_save_npy(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )
    with temporary.open("wb") as file:
        np.save(file, array)
    temporary.replace(path)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
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


def validate_rows(rows):
    if len(rows) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"候选数错误：{len(rows)} "
            f"!= {EXPECTED_CANDIDATES}"
        )

    groups = defaultdict(list)
    identities = []
    config_hashes = set()
    dataset_questions = defaultdict(set)

    for row_index, row in enumerate(rows):
        uid = str(row["question_uid"])
        candidate_index = int(
            row["candidate_index"]
        )

        groups[uid].append(candidate_index)
        dataset_questions[
            str(row["source_dataset"])
        ].add(uid)

        config_hashes.add(
            row["generation_config_sha256"]
        )
        identities.append({
            "row_index": row_index,
            "question_uid": uid,
            "candidate_index": candidate_index,
        })

        if not str(row["problem"]).strip():
            raise RuntimeError(
                f"第 {row_index} 行 problem 为空"
            )
        if not str(row["solution_text"]).strip():
            raise RuntimeError(
                f"第 {row_index} 行 solution 为空"
            )

    if len(groups) != EXPECTED_QUESTIONS:
        raise RuntimeError(
            f"问题数错误：{len(groups)} "
            f"!= {EXPECTED_QUESTIONS}"
        )

    expected_indices = list(range(EXPECTED_K))
    for uid, indices in groups.items():
        if sorted(indices) != expected_indices:
            raise RuntimeError(
                f"{uid} 候选索引错误："
                f"{sorted(indices)}"
            )

    if config_hashes != {
        EXPECTED_GENERATION_CONFIG_SHA256
    }:
        raise RuntimeError(
            "生成配置哈希不一致："
            f"{config_hashes}"
        )

    counts = {
        name: len(uids)
        for name, uids in dataset_questions.items()
    }
    if counts != {
        "AIME_2026": 30,
        "HMMT_FEB_2026": 33,
    }:
        raise RuntimeError(
            f"数据集题数错误：{counts}"
        )

    return identities, counts


def main():
    started = time.time()

    candidate_sha256 = sha256_file(
        CANDIDATE_PATH
    )
    if (
        candidate_sha256
        != EXPECTED_CANDIDATE_SHA256
    ):
        raise RuntimeError(
            "候选文件 SHA256 不一致："
            f"{candidate_sha256}"
        )

    candidate_manifest = json.loads(
        CANDIDATE_MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )
    manifest_candidate_sha = (
        candidate_manifest.get("output_sha256")
    )
    if (
        manifest_candidate_sha is not None
        and manifest_candidate_sha
        != candidate_sha256
    ):
        raise RuntimeError(
            "候选清单中的 SHA256 不一致："
            f"{manifest_candidate_sha}"
        )

    rows = read_jsonl(CANDIDATE_PATH)
    identities, dataset_questions = (
        validate_rows(rows)
    )

    examples = []
    expected_keys = []

    for row_index, row in enumerate(rows):
        key = f"row:{row_index}"
        expected_keys.append(key)
        examples.append({
            "key": key,
            "problem": str(row["problem"]),
            "solution": str(
                row["solution_text"]
            ),
        })

    print("===== Fresh Math Qwen3 K16 RM 评分 =====")
    print("候选：", len(rows))
    print("问题：", len({
        row["question_uid"] for row in rows
    }))
    print("模型：", MODEL_PATH)
    print("候选 SHA256：", candidate_sha256)
    print("标签读取：False")
    print("缓存：", CACHE_ROOT)

    # 复用已经过冻结分数复现审计的评分实现。
    holdout.CACHE = CACHE_ROOT
    holdout.BATCH_SIZE = 8

    scores, peak_gpu_gb = (
        holdout.score_with_judge(
            JUDGE_NAME,
            MODEL_PATH,
            examples,
        )
    )

    if set(scores) != set(expected_keys):
        missing = sorted(
            set(expected_keys) - set(scores)
        )
        unexpected = sorted(
            set(scores) - set(expected_keys)
        )
        raise RuntimeError(
            "评分键不完整："
            f"missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )

    score_array = np.asarray(
        [scores[key] for key in expected_keys],
        dtype=np.float32,
    )

    if score_array.shape != (
        EXPECTED_CANDIDATES,
    ):
        raise RuntimeError(
            f"分数 shape 错误："
            f"{score_array.shape}"
        )
    if not np.all(np.isfinite(score_array)):
        raise RuntimeError(
            "奖励分数包含 NaN 或 Inf"
        )

    atomic_save_npy(
        SCORES_PATH,
        score_array,
    )
    score_sha256 = sha256_file(SCORES_PATH)

    statistics = {
        "min": float(np.min(score_array)),
        "p05": float(np.percentile(
            score_array, 5
        )),
        "median": float(np.median(
            score_array
        )),
        "mean": float(np.mean(
            score_array
        )),
        "p95": float(np.percentile(
            score_array, 95
        )),
        "max": float(np.max(score_array)),
        "std": float(np.std(score_array)),
    }

    result = {
        "version": (
            "fresh_math_2026_qwen3_8b_"
            "rm_1p7b_scores_v1"
        ),
        "evaluation_status": (
            "post_track_a_reveal_exploratory"
        ),
        "labels_loaded": False,
        "candidate_file": str(
            CANDIDATE_PATH.relative_to(ROOT)
        ),
        "candidate_sha256": (
            candidate_sha256
        ),
        "candidate_order_sha256": (
            stable_sha256(identities)
        ),
        "questions": EXPECTED_QUESTIONS,
        "candidates": EXPECTED_CANDIDATES,
        "candidates_per_question": EXPECTED_K,
        "dataset_questions": dataset_questions,
        "generator": "Qwen3-8B",
        "generation_config_sha256": (
            EXPECTED_GENERATION_CONFIG_SHA256
        ),
        "reward_model": {
            "name": (
                "Skywork-Reward-V2-"
                "Qwen3-1.7B"
            ),
            "path": str(
                MODEL_PATH.relative_to(ROOT)
            ),
            "config_sha256": sha256_file(
                MODEL_PATH / "config.json"
            ),
        },
        "scoring_protocol": {
            "conversation": [
                "user: problem",
                "assistant: solution_text",
            ],
            "chat_template": (
                "model tokenizer/template fallback"
            ),
            "add_generation_prompt": False,
            "padding": True,
            "truncation": False,
            "output": (
                "single sequence-"
                "classification logit"
            ),
            "implementation": (
                "eval_answer_cluster_"
                "holdout_rewards."
                "score_with_judge"
            ),
            "batch_size_initial": 8,
            "oom_batch_backoff": True,
        },
        "scores": {
            "path": str(
                SCORES_PATH.relative_to(ROOT)
            ),
            "dtype": "float32",
            "shape": list(
                score_array.shape
            ),
            "sha256": score_sha256,
            "statistics": statistics,
        },
        "cache_file": str(
            (
                CACHE_ROOT
                / f"{JUDGE_NAME}.json"
            ).relative_to(ROOT)
        ),
        "peak_gpu_gb_current_run": (
            float(peak_gpu_gb)
        ),
        "elapsed_seconds": round(
            time.time() - started,
            3,
        ),
        "git_head": git_head(),
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    atomic_json(
        OUTPUT_MANIFEST_PATH,
        result,
    )

    # 落盘后再次读取，防止写入或顺序异常。
    verified = np.asarray(
        np.load(SCORES_PATH),
        dtype=np.float32,
    )
    if not np.array_equal(
        verified,
        score_array,
    ):
        raise RuntimeError(
            "落盘后的分数数组不一致"
        )

    print()
    print("===== RM 评分完成 =====")
    print(json.dumps(
        {
            "questions": EXPECTED_QUESTIONS,
            "candidates": EXPECTED_CANDIDATES,
            "score_sha256": score_sha256,
            "statistics": statistics,
            "peak_gpu_gb": peak_gpu_gb,
            "elapsed_seconds": result[
                "elapsed_seconds"
            ],
        },
        ensure_ascii=False,
        indent=2,
    ))
    print("分数：", SCORES_PATH)
    print("清单：", OUTPUT_MANIFEST_PATH)


if __name__ == "__main__":
    main()

from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_answer_cluster_generator_full as evaluation
import explore_answer_cluster_evidence_smoke as cluster

from audit_answer_cluster_consensus import (
    normalize_answer,
)


CANDIDATE_PATH = (
    ROOT
    / "outputs/fresh_math_2026/"
    "qwen3_8b_adaptive_k16_candidates.jsonl"
)
RM_SCORE_PATH = (
    ROOT
    / "data/cache/fresh_math_2026/"
    "qwen3_8b_adaptive_k16_rm_1p7b/"
    "scores_f32.npy"
)
FINAL_CONFIG_PATH = (
    ROOT / "configs/answer_cluster_final.json"
)
METHOD_SCORE_PATH = (
    ROOT
    / "data/cache/fresh_math_2026/"
    "qwen3_8b_adaptive_k16_rm_support_math/"
    "scores_f32.npy"
)
OUTPUT_PATH = (
    ROOT
    / "data/manifests/"
    "fresh_math_2026_qwen3_8b_"
    "rm_support_predictions.json"
)

EXPECTED_CANDIDATE_SHA256 = (
    "9af1176c0b092608f413d90889315c03"
    "ef4ad5d3d994ad8746dadd5e131f4257"
)
EXPECTED_RM_SCORE_SHA256 = (
    "c0f6776946aab3283c667d2d503d3027"
    "470419309734c833f11353ab59e555e4"
)

REGULARIZATION = 0.001
BETA = 2.0
THRESHOLD = 0.25
EXPECTED_FEATURE_INDICES = list(range(8))
EXPECTED_SEEDS = [42, 123, 456]


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


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return None


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


def atomic_npy(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )
    with temporary.open("wb") as file:
        np.save(file, value)
    temporary.replace(path)


def build_fresh_questions(rows, rm_scores):
    groups = defaultdict(list)

    for global_index, row in enumerate(rows):
        groups[
            str(row["question_uid"])
        ].append(global_index)

    questions = []

    for uid, indices_list in groups.items():
        indices = np.asarray(
            indices_list,
            dtype=np.int64,
        )
        rm = rm_scores[indices]
        rm_relative = cluster.robust_z(rm)

        answer_groups = defaultdict(list)

        for local_index, global_index in enumerate(
            indices
        ):
            row = rows[int(global_index)]
            raw_answer = row.get(
                "parsed_answer_generation_audit"
            )

            answer = (
                normalize_answer(
                    raw_answer,
                    "math",
                )
                if raw_answer is not None
                else None
            )

            if answer is None:
                answer = (
                    f"__unparsed_"
                    f"{int(global_index)}"
                )

            answer_groups[answer].append(
                local_index
            )

        raw_local = int(np.argmax(rm))
        raw_answer = next(
            answer
            for answer, members
            in answer_groups.items()
            if raw_local in members
        )

        cluster_items = []
        total_candidates = len(indices)

        for answer, member_list in (
            answer_groups.items()
        ):
            members = np.asarray(
                member_list,
                dtype=np.int64,
            )

            unique_solutions = {
                cluster.canonical_solution(
                    str(
                        rows[
                            int(indices[member])
                        ]["solution_text"]
                    )
                )
                for member in members
            }

            member_rm_relative = (
                rm_relative[members]
            )

            # 最终 rm_support 的八个低维特征。
            features = np.asarray([
                math.log1p(len(members)),
                len(members) / total_candidates,
                math.log1p(
                    len(unique_solutions)
                ),
                float(np.max(
                    member_rm_relative
                )),
                float(np.mean(
                    member_rm_relative
                )),
                cluster.stable_logmeanexp(
                    member_rm_relative
                ),
                float(np.std(
                    member_rm_relative
                )),
                float(answer == raw_answer),
            ], dtype=np.float32)

            cluster_items.append({
                "answer": answer,
                "members": members,
                "features": features,
                "rm_max": float(
                    np.max(rm[members])
                ),
            })

        raw_cluster = next(
            index
            for index, item
            in enumerate(cluster_items)
            if raw_local in item["members"]
        )

        questions.append({
            "uid": uid,
            "indices": indices,
            "rm": rm,
            "clusters": cluster_items,
            "raw_local": raw_local,
            "raw_cluster": raw_cluster,
        })

    return questions


def serialize_models(models):
    return [
        {
            "seed": int(model["seed"]),
            "regularization": float(
                model["regularization"]
            ),
            "weights": np.asarray(
                model["weights"],
                dtype=np.float64,
            ).tolist(),
            "mean": np.asarray(
                model["mean"],
                dtype=np.float64,
            ).tolist(),
            "std": np.asarray(
                model["std"],
                dtype=np.float64,
            ).tolist(),
        }
        for model in models
    ]


def main():
    started = time.time()

    candidate_sha256 = sha256_file(
        CANDIDATE_PATH
    )
    rm_score_sha256 = sha256_file(
        RM_SCORE_PATH
    )

    if (
        candidate_sha256
        != EXPECTED_CANDIDATE_SHA256
    ):
        raise RuntimeError(
            "候选 SHA256 不匹配"
        )
    if (
        rm_score_sha256
        != EXPECTED_RM_SCORE_SHA256
    ):
        raise RuntimeError(
            "RM 分数 SHA256 不匹配"
        )

    feature_indices = list(
        evaluation.ABLATIONS["rm_support"]
    )
    seeds = list(cluster.SEEDS)

    if (
        feature_indices
        != EXPECTED_FEATURE_INDICES
    ):
        raise RuntimeError(
            "rm_support 特征发生变化："
            f"{feature_indices}"
        )
    if seeds != EXPECTED_SEEDS:
        raise RuntimeError(
            f"训练种子发生变化：{seeds}"
        )

    rows = read_jsonl(CANDIDATE_PATH)
    rm_scores = np.asarray(
        np.load(RM_SCORE_PATH),
        dtype=np.float32,
    )

    if len(rows) != 1008:
        raise RuntimeError(
            f"候选数错误：{len(rows)}"
        )
    if rm_scores.shape != (1008,):
        raise RuntimeError(
            f"RM 分数 shape 错误："
            f"{rm_scores.shape}"
        )

    print(
        "===== 冻结 MATH RM-support "
        "跨生成器推理 ====="
    )
    print("Fresh 标签读取：False")
    print("候选：", len(rows))
    print(
        "配置：",
        {
            "regularization": (
                REGULARIZATION
            ),
            "beta": BETA,
            "threshold": THRESHOLD,
            "features": feature_indices,
            "seeds": seeds,
        },
    )

    math_spec = evaluation.DOMAINS["MATH"]

    # 这里只读取原始 MATH Train 标签，
    # 不读取任何 Fresh Math 标签。
    train = evaluation.load_dataset(
        "MATH_TRAIN_FOR_FRESH_TRANSFER",
        math_spec,
        *math_spec["train"],
    )

    print("重建三个冻结随机种子模型……")
    models = [
        cluster.fit_cluster_model(
            train,
            feature_indices,
            REGULARIZATION,
            seed,
        )
        for seed in seeds
    ]

    questions = build_fresh_questions(
        rows,
        rm_scores,
    )

    if len(questions) != 63:
        raise RuntimeError(
            f"Fresh 问题数错误："
            f"{len(questions)}"
        )

    method_scores = np.empty(
        len(rows),
        dtype=np.float32,
    )
    selections = []
    switch_count = 0
    authorized_switch_count = 0

    for question in questions:
        learned = cluster.model_cluster_scores(
            question,
            feature_indices,
            models,
        )
        learned = cluster.ordinary_z(learned)

        base = np.asarray([
            item["rm_max"]
            for item in question["clusters"]
        ], dtype=np.float32)
        base = cluster.ordinary_z(base)

        hybrid = base + BETA * learned

        proposal_cluster = int(
            np.argmax(hybrid)
        )
        raw_cluster = int(
            question["raw_cluster"]
        )
        advantage = float(
            hybrid[proposal_cluster]
            - hybrid[raw_cluster]
        )

        authorized = bool(
            proposal_cluster == raw_cluster
            or advantage > THRESHOLD
        )

        local_scores = (
            cluster.candidate_scores_from_clusters(
                question,
                hybrid,
                THRESHOLD,
            )
        )
        method_scores[
            question["indices"]
        ] = local_scores

        selected_local = int(
            np.argmax(local_scores)
        )
        raw_local = int(
            question["raw_local"]
        )
        selected_global = int(
            question["indices"][
                selected_local
            ]
        )
        raw_global = int(
            question["indices"][raw_local]
        )

        selected_cluster = next(
            index
            for index, item
            in enumerate(
                question["clusters"]
            )
            if selected_local
            in item["members"]
        )

        switched = (
            selected_local != raw_local
        )
        switch_count += int(switched)
        authorized_switch_count += int(
            switched and authorized
        )

        selected_row = rows[selected_global]
        raw_row = rows[raw_global]

        selections.append({
            "question_uid": question["uid"],
            "source_dataset": (
                selected_row["source_dataset"]
            ),
            "problem_id": (
                selected_row["problem_id"]
            ),
            "raw_candidate_index": int(
                raw_row["candidate_index"]
            ),
            "selected_candidate_index": int(
                selected_row[
                    "candidate_index"
                ]
            ),
            "raw_answer": (
                question["clusters"][
                    raw_cluster
                ]["answer"]
            ),
            "proposal_answer": (
                question["clusters"][
                    proposal_cluster
                ]["answer"]
            ),
            "selected_answer": (
                question["clusters"][
                    selected_cluster
                ]["answer"]
            ),
            "raw_cluster": raw_cluster,
            "proposal_cluster": (
                proposal_cluster
            ),
            "selected_cluster": (
                selected_cluster
            ),
            "proposal_advantage": (
                advantage
            ),
            "proposal_authorized": (
                authorized
            ),
            "switched": switched,
            "cluster_count": len(
                question["clusters"]
            ),
        })

    if not np.all(
        np.isfinite(method_scores)
    ):
        raise RuntimeError(
            "方法分数包含 NaN 或 Inf"
        )

    atomic_npy(
        METHOD_SCORE_PATH,
        method_scores,
    )
    method_score_sha256 = sha256_file(
        METHOD_SCORE_PATH
    )

    manifest = {
        "version": (
            "fresh_math_2026_qwen3_8b_"
            "rm_support_predictions_v1"
        ),
        "protocol": (
            "zero-shot cross-generator "
            "transfer from frozen MATH "
            "rm_support"
        ),
        "evaluation_status": (
            "post_track_a_reveal_exploratory"
        ),
        "fresh_labels_loaded": False,
        "fresh_labels_used_for_training": False,
        "fresh_labels_used_for_selection": False,
        "training_source": (
            "original MATH Train only"
        ),
        "selection_source": (
            "previously frozen MATH Pilot "
            "configuration"
        ),
        "candidate_file": str(
            CANDIDATE_PATH.relative_to(ROOT)
        ),
        "candidate_sha256": (
            candidate_sha256
        ),
        "raw_rm_score_file": str(
            RM_SCORE_PATH.relative_to(ROOT)
        ),
        "raw_rm_score_sha256": (
            rm_score_sha256
        ),
        "final_config_file": str(
            FINAL_CONFIG_PATH.relative_to(ROOT)
        ),
        "final_config_sha256": sha256_file(
            FINAL_CONFIG_PATH
        ),
        "feature_indices": feature_indices,
        "feature_names": [
            cluster.FEATURE_NAMES[index]
            for index in feature_indices
        ],
        "configuration": {
            "regularization": (
                REGULARIZATION
            ),
            "beta": BETA,
            "threshold": THRESHOLD,
            "seeds": seeds,
        },
        "fitted_models": (
            serialize_models(models)
        ),
        "training_questions": len(
            train["questions"]
        ),
        "training_candidates": len(
            train["rows"]
        ),
        "fresh_questions": len(questions),
        "fresh_candidates": len(rows),
        "switch_count": switch_count,
        "switch_rate": (
            switch_count / len(questions)
        ),
        "authorized_switch_count": (
            authorized_switch_count
        ),
        "method_scores": {
            "path": str(
                METHOD_SCORE_PATH.relative_to(
                    ROOT
                )
            ),
            "sha256": (
                method_score_sha256
            ),
            "dtype": "float32",
            "shape": list(
                method_scores.shape
            ),
            # 将全部分数写入 Git 清单，
            # 保证标签评估前的预测可被精确冻结。
            "values": [
                float(value)
                for value in method_scores
            ],
        },
               "selections": selections,
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
        OUTPUT_PATH,
        manifest,
    )

    verified = json.loads(
        OUTPUT_PATH.read_text(
            encoding="utf-8"
        )
    )
    if verified["fresh_labels_loaded"]:
        raise RuntimeError(
            "无标签协议字段异常"
        )
    if (
        len(verified["method_scores"][
            "values"
        ])
        != 1008
    ):
        raise RuntimeError(
            "清单中的预测分数不完整"
        )
    if len(
        verified["selections"]
    ) != 63:
        raise RuntimeError(
            "清单中的题目选择不完整"
        )

    print()
    print("===== 无标签迁移预测完成 =====")
    print(json.dumps({
        "questions": len(questions),
        "candidates": len(rows),
        "switch_count": switch_count,
        "switch_rate": (
            switch_count / len(questions)
        ),
        "authorized_switch_count": (
            authorized_switch_count
        ),
        "method_score_sha256": (
            method_score_sha256
        ),
        "elapsed_seconds": (
            manifest["elapsed_seconds"]
        ),
    }, ensure_ascii=False, indent=2))
    print("分数：", METHOD_SCORE_PATH)
    print("清单：", OUTPUT_PATH)


if __name__ == "__main__":
    main()

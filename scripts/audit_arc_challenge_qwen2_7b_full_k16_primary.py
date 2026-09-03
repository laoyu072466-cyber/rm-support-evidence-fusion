from pathlib import Path
from collections import Counter, defaultdict
from statistics import median
import hashlib
import json
import os


ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_PATH = (
    ROOT / "data/processed/arc_challenge_v1/"
    "qwen2_7b_full_k16_questions.jsonl"
)
OUTPUT_PATH = (
    ROOT / "outputs/arc_challenge_v1/"
    "qwen2_7b_full_k16_candidates.jsonl"
)
PARTS_PATH = (
    ROOT / "outputs/arc_challenge_v1/"
    "qwen2_7b_full_k16_parts"
)
MANIFEST_PATH = (
    ROOT / "data/manifests/"
    "arc_challenge_qwen2_7b_full_k16_v1.json"
)
EXIT_PATH = (
    ROOT / "outputs/logs/"
    "generate_arc_challenge_qwen2_7b_full_k16.exit"
)

EXPECTED_OUTPUT_SHA256 = (
    "14f83e6a1b2077600081d9060e8d7ef6"
    "f3d46d878a04265c41883a9c71e3148c"
)
EXPECTED_CONFIG_SHA256 = (
    "c6340e9cc3a7376fe6998fbb50d7515f"
    "258ccbb38594bece69a4d65d7e0418d8"
)
EXPECTED_SPLITS = {
    "train": 1119,
    "pilot": 299,
    "test": 1172,
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


def atomic_json(path, value):
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


def main():
    required = [
        QUESTIONS_PATH,
        OUTPUT_PATH,
        MANIFEST_PATH,
        EXIT_PATH,
    ]
    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(
            f"缺少文件：{missing}"
        )

    exit_status = EXIT_PATH.read_text(
        encoding="utf-8"
    ).strip()
    if exit_status != "0":
        raise RuntimeError(
            f"生成退出状态不是 0：{exit_status!r}"
        )

    output_sha256 = sha256_file(
        OUTPUT_PATH
    )
    if (
        output_sha256
        != EXPECTED_OUTPUT_SHA256
    ):
        raise RuntimeError(
            "输出 SHA256 不匹配"
        )

    questions = read_jsonl(
        QUESTIONS_PATH
    )
    candidates = read_jsonl(
        OUTPUT_PATH
    )
    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    question_by_uid = {
        str(row["question_uid"]): row
        for row in questions
    }

    if (
        len(questions) != 2590
        or len(question_by_uid) != 2590
    ):
        raise RuntimeError(
            "问题数量或 UID 唯一性错误"
        )

    expected_split_counts = Counter(
        str(row["logical_split"])
        for row in questions
    )
    if expected_split_counts != Counter(
        EXPECTED_SPLITS
    ):
        raise RuntimeError(
            "问题划分数量错误"
        )

    groups = defaultdict(list)
    identities = set()
    methods = Counter()
    candidate_splits = Counter()
    config_hashes = set()
    parsed_count = 0

    forbidden_fields = {
        "answer",
        "answerKey",
        "answer_index",
        "answer_label",
        "gold",
        "label",
    }

    for row in candidates:
        uid = str(row["question_uid"])
        index = int(row["candidate_index"])
        key = (uid, index)

        if uid not in question_by_uid:
            raise RuntimeError(
                f"未知问题 UID：{uid}"
            )
        if key in identities:
            raise RuntimeError(
                f"候选身份重复：{key}"
            )
        if forbidden_fields & set(row):
            raise RuntimeError(
                f"候选含答案标签字段：{key}"
            )

        identities.add(key)
        groups[uid].append(row)

        split = str(row["logical_split"])
        expected_split = str(
            question_by_uid[uid][
                "logical_split"
            ]
        )
        if split != expected_split:
            raise RuntimeError(
                f"候选划分错误：{key}"
            )

        candidate_splits[split] += 1

        method = str(
            row.get(
                "parse_method_generation_audit",
                "unparsed",
            )
        )
        methods[method] += 1

        if row.get(
            "parsed_choice_index_generation_audit"
        ) is not None:
            parsed_count += 1

        config_hashes.add(str(
            row["generation_config_sha256"]
        ))

    if len(candidates) != 41440:
        raise RuntimeError(
            "候选总数不是 41440"
        )
    if len(identities) != 41440:
        raise RuntimeError(
            "候选身份不唯一"
        )
    if set(groups) != set(question_by_uid):
        raise RuntimeError(
            "候选问题覆盖不完整"
        )

    for uid, rows in groups.items():
        indices = sorted(
            int(row["candidate_index"])
            for row in rows
        )
        if indices != list(range(16)):
            raise RuntimeError(
                f"{uid}: 不满足完整 K16"
            )

    expected_candidate_splits = {
        split: count * 16
        for split, count
        in EXPECTED_SPLITS.items()
    }
    if candidate_splits != Counter(
        expected_candidate_splits
    ):
        raise RuntimeError(
            "候选划分数量错误"
        )

    if config_hashes != {
        EXPECTED_CONFIG_SHA256
    }:
        raise RuntimeError(
            f"候选配置哈希异常："
            f"{config_hashes}"
        )

    if (
        manifest.get("labels_loaded")
        is not False
        or manifest.get(
            "sealed_test_labels_loaded"
        ) is not False
    ):
        raise RuntimeError(
            "清单标签状态异常"
        )

    if (
        manifest.get("output_sha256")
        != EXPECTED_OUTPUT_SHA256
    ):
        raise RuntimeError(
            "清单输出 SHA256 不匹配"
        )

    part_files = (
        list(PARTS_PATH.glob("*.json"))
        if PARTS_PATH.exists()
        else []
    )
    if len(part_files) != 2590:
        raise RuntimeError(
            "断点分片数量不是 2590"
        )

    clusters = []
    no_parsed_questions = 0

    for rows in groups.values():
        values = {
            int(value)
            for value in [
                row.get(
                    "parsed_choice_index_generation_audit"
                )
                for row in rows
            ]
            if value is not None
        }
        clusters.append(len(values))
        if not values:
            no_parsed_questions += 1

    audit = {
        "labels_loaded": False,
        "sealed_test_labels_loaded": False,
        "generation_exit_status": 0,
        "questions": len(questions),
        "candidates": len(candidates),
        "candidates_per_question": 16,
        "question_splits": dict(
            expected_split_counts
        ),
        "candidate_splits": dict(
            candidate_splits
        ),
        "unique_candidate_identities": (
            len(identities)
        ),
        "part_files": len(part_files),
        "parsed_candidates": parsed_count,
        "unparsed_candidates": (
            len(candidates) - parsed_count
        ),
        "parse_rate": (
            parsed_count / len(candidates)
        ),
        "parse_methods": dict(methods),
        "answer_clusters_per_question": {
            "min": min(clusters),
            "median": median(clusters),
            "mean": (
                sum(clusters) / len(clusters)
            ),
            "max": max(clusters),
            "questions_with_no_parsed_choice": (
                no_parsed_questions
            ),
            "multi_cluster_questions": sum(
                value > 1
                for value in clusters
            ),
        },
        "generation_config_sha256": (
            EXPECTED_CONFIG_SHA256
        ),
        "output_sha256": output_sha256,
        "decision": (
            "freeze_primary_before_recovery"
        ),
    }

    manifest[
        "primary_integrity_audit"
    ] = audit
    atomic_json(
        MANIFEST_PATH,
        manifest,
    )

    print(
        "===== ARC 全量 Primary "
        "无标签完整性审计通过 ====="
    )
    print(json.dumps(
        audit,
        ensure_ascii=False,
        indent=2,
    ))
    print("清单：", MANIFEST_PATH)


if __name__ == "__main__":
    main()

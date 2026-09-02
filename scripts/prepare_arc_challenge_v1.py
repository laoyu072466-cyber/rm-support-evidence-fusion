from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import hashlib
import json
import os

import datasets
from datasets import load_dataset


ROOT = Path("/root/autodl-tmp/rm_traj_project")
OUTPUT_DIR = ROOT / "data/external/arc_challenge_v1"
MANIFEST_PATH = (
    ROOT / "data/manifests/"
    "arc_challenge_source_manifest_v1.json"
)

DATASET_NAME = "allenai/ai2_arc"
DATASET_CONFIG = "ARC-Challenge"

SPLITS = {
    "train": {
        "source_split": "train",
        "role": "train",
        "questions": "train_questions.jsonl",
        "labels": "train_labels.jsonl",
        "sealed": False,
    },
    "pilot": {
        "source_split": "validation",
        "role": "pilot",
        "questions": "pilot_questions.jsonl",
        "labels": "pilot_labels.jsonl",
        "sealed": False,
    },
    "test": {
        "source_split": "test",
        "role": "frozen_test",
        "questions": "test_questions.jsonl",
        "labels": "sealed_test_labels.jsonl",
        "sealed": True,
    },
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


def normalize_text(value):
    return " ".join(str(value).lower().split())


def stable_hash(value, length=None):
    digest = hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    return digest[:length] if length else digest


def atomic_jsonl(path, rows):
    temporary = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                ) + "\n"
            )

    temporary.replace(path)


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


def duplicate_records(groups):
    records = []

    for signature, values in groups.items():
        if len(values) <= 1:
            continue

        records.append({
            "signature_sha256": stable_hash(
                signature
            ),
            "count": len(values),
            "source_ids": [
                value["source_id"]
                for value in values
            ],
            "question_uids": [
                value["question_uid"]
                for value in values
            ],
            "option_variants": len({
                value["full_signature"]
                for value in values
            }),
        })

    return records


def main():
    print("===== 冻结 ARC-Challenge v1 =====")
    print("保留完整官方划分。")
    print("标准答案不会打印。")

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_audits = {}
    manifest_splits = {}

    for output_name, spec in SPLITS.items():
        source_split = spec["source_split"]
        source_rows = dataset[source_split]

        questions = []
        labels = []
        stem_groups = defaultdict(list)
        full_groups = defaultdict(list)
        choice_counts = Counter()
        label_formats = Counter()

        for source_index, row in enumerate(
            source_rows
        ):
            source_id = str(row["id"])
            question = str(row["question"]).strip()

            choice_labels = [
                str(value).strip()
                for value in row["choices"]["label"]
            ]
            choice_texts = [
                str(value).strip()
                for value in row["choices"]["text"]
            ]

            if len(choice_labels) != len(choice_texts):
                raise RuntimeError(
                    f"{source_split}/{source_index}: "
                    "选项数量不一致"
                )

            if len(set(choice_labels)) != len(
                choice_labels
            ):
                raise RuntimeError(
                    f"{source_split}/{source_index}: "
                    "选项标签重复"
                )

            answer_key = str(
                row["answerKey"]
            ).strip()

            if answer_key not in choice_labels:
                raise RuntimeError(
                    f"{source_split}/{source_index}: "
                    "答案键不属于选项"
                )

            answer_index = choice_labels.index(
                answer_key
            )

            stem_signature = normalize_text(
                question
            )
            full_signature = json.dumps(
                {
                    "question": stem_signature,
                    "choices": [
                        normalize_text(value)
                        for value in choice_texts
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )

            question_uid = stable_hash(
                {
                    "dataset": DATASET_NAME,
                    "config": DATASET_CONFIG,
                    "split": source_split,
                    "source_id": source_id,
                    "question": question,
                    "choice_labels": choice_labels,
                    "choice_texts": choice_texts,
                },
                length=20,
            )

            questions.append({
                "question_uid": question_uid,
                "source_dataset": "ARC_CHALLENGE",
                "source_name": DATASET_NAME,
                "source_config": DATASET_CONFIG,
                "source_split": source_split,
                "data_role": spec["role"],
                "source_row_index": source_index,
                "source_id": source_id,
                "question": question,
                "choice_labels": choice_labels,
                "choice_texts": choice_texts,
            })

            labels.append({
                "question_uid": question_uid,
                "source_split": source_split,
                "source_row_index": source_index,
                "source_id": source_id,
                "answer_label": answer_key,
                "answer_index": answer_index,
            })

            audit_record = {
                "source_id": source_id,
                "question_uid": question_uid,
                "stem_signature": stem_signature,
                "full_signature": full_signature,
            }

            stem_groups[stem_signature].append(
                audit_record
            )
            full_groups[full_signature].append(
                audit_record
            )

            choice_counts[len(choice_labels)] += 1
            label_formats[
                tuple(choice_labels)
            ] += 1

        question_path = (
            OUTPUT_DIR / spec["questions"]
        )
        label_path = OUTPUT_DIR / spec["labels"]

        atomic_jsonl(question_path, questions)
        atomic_jsonl(label_path, labels)

        stem_duplicates = duplicate_records(
            stem_groups
        )
        full_duplicates = duplicate_records(
            full_groups
        )

        split_audits[output_name] = {
            "stem_groups": stem_groups,
            "full_groups": full_groups,
        }

        manifest_splits[output_name] = {
            "source_split": source_split,
            "role": spec["role"],
            "source_rows": len(source_rows),
            "retained_rows": len(questions),
            "official_rows_retained": True,
            "question_file": str(
                question_path.relative_to(ROOT)
            ),
            "question_sha256": sha256_file(
                question_path
            ),
            "label_file": str(
                label_path.relative_to(ROOT)
            ),
            "label_sha256": sha256_file(
                label_path
            ),
            "labels_sealed": spec["sealed"],
            "choice_counts": {
                str(key): value
                for key, value in sorted(
                    choice_counts.items()
                )
            },
            "choice_label_formats": {
                "|".join(key): value
                for key, value in sorted(
                    label_formats.items()
                )
            },
            "within_split_duplicate_stems": (
                len(stem_duplicates)
            ),
            "within_split_exact_duplicates": (
                len(full_duplicates)
            ),
            "duplicate_stem_records": (
                stem_duplicates
            ),
            "exact_duplicate_records": (
                full_duplicates
            ),
            "dataset_fingerprint": getattr(
                source_rows,
                "_fingerprint",
                None,
            ),
        }

    cross_split = {}
    split_names = list(SPLITS)

    for left_position, left in enumerate(
        split_names
    ):
        for right in split_names[
            left_position + 1:
        ]:
            left_stems = split_audits[left][
                "stem_groups"
            ]
            right_stems = split_audits[right][
                "stem_groups"
            ]
            left_full = split_audits[left][
                "full_groups"
            ]
            right_full = split_audits[right][
                "full_groups"
            ]

            shared_stems = sorted(
                set(left_stems) & set(right_stems)
            )
            shared_full = sorted(
                set(left_full) & set(right_full)
            )

            key = f"{left}__{right}"

            cross_split[key] = {
                "same_question_stems": len(
                    shared_stems
                ),
                "same_question_and_choices": len(
                    shared_full
                ),
                "shared_stem_records": [
                    {
                        "stem_sha256": stable_hash(
                            stem
                        ),
                        "left_source_ids": [
                            value["source_id"]
                            for value in left_stems[
                                stem
                            ]
                        ],
                        "right_source_ids": [
                            value["source_id"]
                            for value in right_stems[
                                stem
                            ]
                        ],
                        "exact_content_match": any(
                            left_value[
                                "full_signature"
                            ]
                            == right_value[
                                "full_signature"
                            ]
                            for left_value in (
                                left_stems[stem]
                            )
                            for right_value in (
                                right_stems[stem]
                            )
                        ),
                    }
                    for stem in shared_stems
                ],
                "exact_content_records": [
                    {
                        "content_sha256": stable_hash(
                            signature
                        ),
                        "left_source_ids": [
                            value["source_id"]
                            for value in left_full[
                                signature
                            ]
                        ],
                        "right_source_ids": [
                            value["source_id"]
                            for value in right_full[
                                signature
                            ]
                        ],
                    }
                    for signature in shared_full
                ],
            }

    cross_exact_total = sum(
        value["same_question_and_choices"]
        for value in cross_split.values()
    )

    manifest = {
        "version": "arc_challenge_source_v1",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": {
            "dataset": DATASET_NAME,
            "config": DATASET_CONFIG,
            "datasets_version": datasets.__version__,
        },
        "protocol": {
            "official_splits_retained": True,
            "train_source": "train",
            "pilot_source": "validation",
            "test_source": "test",
            "generation_reads_label_files": False,
            "test_labels_sealed_before_generation": True,
            "label_values_printed": False,
            "candidate_budget": 16,
            "shared_stem_policy": (
                "retain when option sets differ; "
                "report sensitivity analysis"
            ),
        },
        "splits": manifest_splits,
        "cross_split_audit": cross_split,
        "decision": (
            "pass_with_documented_variants"
            if cross_exact_total == 0
            else "manual_review_required"
        ),
    }

    atomic_json(MANIFEST_PATH, manifest)

    print()
    print("===== 冻结完成 =====")

    for name, value in manifest_splits.items():
        print(
            f"{name}: "
            f"{value['retained_rows']} 题, "
            f"重复题干组="
            f"{value['within_split_duplicate_stems']}, "
            f"完全重复组="
            f"{value['within_split_exact_duplicates']}, "
            f"sealed={value['labels_sealed']}"
        )

    print()
    print("跨 split：")
    for name, value in cross_split.items():
        print(
            f"  {name}: "
            f"共享题干={value['same_question_stems']}, "
            f"完全相同="
            f"{value['same_question_and_choices']}"
        )

    print()
    print("结论：", manifest["decision"])
    print("清单：", MANIFEST_PATH)


if __name__ == "__main__":
    main()

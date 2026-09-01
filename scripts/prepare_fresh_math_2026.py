from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess

from datasets import load_dataset


ROOT = Path("/root/autodl-tmp/rm_traj_project")
OUTPUT_DIR = ROOT / "data/external/fresh_math_2026"
MANIFEST_PATH = (
    ROOT / "data/manifests/fresh_math_2026_source_manifest.json"
)

SOURCES = {
    "AIME_2026": {
        "hf_dataset": "MathArena/aime_2026",
        "split": "train",
        "expected_rows": 30,
        "license": "CC BY-NC-SA 4.0",
    },
    "HMMT_FEB_2026": {
        "hf_dataset": "MathArena/hmmt_feb_2026",
        "split": "train",
        "expected_rows": 33,
        "license": "CC BY-NC-SA 4.0",
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }
    if hasattr(value, "item"):
        return json_safe(value.item())
    return str(value)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def main():
    questions_path = OUTPUT_DIR / "questions.jsonl"
    labels_path = OUTPUT_DIR / "sealed_labels.jsonl"

    for path in [
        questions_path,
        labels_path,
        MANIFEST_PATH,
    ]:
        if path.exists():
            raise FileExistsError(
                f"为避免覆盖冻结数据，已停止：{path}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    questions = []
    labels = []
    source_results = {}

    for source_name, spec in SOURCES.items():
        print("=" * 72)
        print("下载：", spec["hf_dataset"])

        dataset = load_dataset(
            spec["hf_dataset"],
            split=spec["split"],
        )

        if len(dataset) != spec["expected_rows"]:
            raise RuntimeError(
                f"{source_name} 行数异常："
                f"{len(dataset)} != {spec['expected_rows']}"
            )

        source_uids = []

        for row_index, raw_row in enumerate(dataset):
            row = {
                str(key): json_safe(value)
                for key, value in raw_row.items()
            }

            problem = str(row["problem"]).strip()
            if not problem:
                raise RuntimeError(
                    f"{source_name}:{row_index} 题目为空"
                )

            original_id = row.get(
                "problem_idx",
                row_index + 1,
            )
            answer = row.get("answer")
            if answer is None:
                raise RuntimeError(
                    f"{source_name}:{row_index} 缺少答案"
                )

            uid_source = (
                f"{source_name}\n"
                f"{original_id}\n"
                f"{problem}"
            )
            question_uid = hashlib.sha256(
                uid_source.encode("utf-8")
            ).hexdigest()[:20]

            question_row = {
                "question_uid": question_uid,
                "source_dataset": source_name,
                "source_row_index": row_index,
                "problem_id": json_safe(original_id),
                "problem": problem,
            }

            if "problem_type" in row:
                question_row["problem_type"] = row[
                    "problem_type"
                ]

            label_row = {
                "question_uid": question_uid,
                "source_dataset": source_name,
                "problem_id": json_safe(original_id),
                "answer": json_safe(answer),
            }

            questions.append(question_row)
            labels.append(label_row)
            source_uids.append(question_uid)

        source_results[source_name] = {
            **spec,
            "rows": len(dataset),
            "dataset_fingerprint": getattr(
                dataset,
                "_fingerprint",
                None,
            ),
            "question_uid_sha256": hashlib.sha256(
                "\n".join(source_uids).encode("utf-8")
            ).hexdigest(),
        }

        print(
            f"{source_name}: {len(dataset)} 题，"
            f"字段={dataset.column_names}"
        )

    all_uids = [
        row["question_uid"]
        for row in questions
    ]
    if len(all_uids) != len(set(all_uids)):
        raise RuntimeError("question_uid 存在重复")

    write_jsonl(questions_path, questions)
    write_jsonl(labels_path, labels)

    manifest = {
        "version": "fresh_math_2026_source_v1",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "project_git_head_before_import": git_head(),
        "protocol": {
            "role": "sealed_blind_test",
            "method_retraining_allowed": False,
            "hyperparameter_tuning_allowed": False,
            "labels_used_for_generation": False,
            "labels_used_for_ranking": False,
            "evaluation_after_predictions_frozen": True,
        },
        "sources": source_results,
        "total_questions": len(questions),
        "files": {
            "questions": str(
                questions_path.relative_to(ROOT)
            ),
            "sealed_labels": str(
                labels_path.relative_to(ROOT)
            ),
        },
        "sha256": {
            "questions": sha256_file(questions_path),
            "sealed_labels": sha256_file(labels_path),
        },
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("===== 冻结完成 =====")
    print("总题数：", len(questions))
    print("题目文件：", questions_path)
    print("密封标签：", labels_path)
    print("清单：", MANIFEST_PATH)
    print()
    print("第一条题目元数据：")
    preview = dict(questions[0])
    preview["problem"] = preview["problem"][:240]
    print(
        json.dumps(
            preview,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

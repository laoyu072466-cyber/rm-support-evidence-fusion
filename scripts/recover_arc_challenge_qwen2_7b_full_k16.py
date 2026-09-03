from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
)


ROOT = Path("/root/autodl-tmp/rm_traj_project")
sys.path.insert(0, str(ROOT / "scripts"))

import generate_arc_challenge_qwen2_7b_full_k16 as arc


CONFIG_PATH = (
    ROOT / "configs/"
    "arc_challenge_qwen2_7b_full_k16_recovery_v1.json"
)
OUTPUT_PATH = (
    ROOT / "outputs/arc_challenge_v1/"
    "qwen2_7b_full_k16_recovered_candidates.jsonl"
)
MANIFEST_PATH = (
    ROOT / "data/manifests/"
    "arc_challenge_qwen2_7b_full_k16_recovery_v1.json"
)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
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


def stable_hash(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def atomic_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
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


def preflight():
    config = json.loads(
        CONFIG_PATH.read_text(encoding="utf-8")
    )

    source = config["source"]
    primary_path = ROOT / source["primary_candidates"]
    primary_manifest_path = (
        ROOT / source["primary_manifest"]
    )
    questions_path = ROOT / source["questions"]

    for path in [
        primary_path,
        primary_manifest_path,
        questions_path,
    ]:
        if not path.exists():
            raise RuntimeError(
                f"缺少输入：{path}"
            )

    if (
        sha256_file(primary_path)
        != source["primary_candidates_sha256"]
    ):
        raise RuntimeError(
            "Primary 候选 SHA256 不一致"
        )

    if (
        sha256_file(primary_manifest_path)
        != source["primary_manifest_sha256"]
    ):
        raise RuntimeError(
            "Primary 清单 SHA256 不一致"
        )

    if (
        sha256_file(questions_path)
        != source["questions_sha256"]
    ):
        raise RuntimeError(
            "问题 SHA256 不一致"
        )

    primary_manifest = json.loads(
        primary_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if primary_manifest.get(
        "labels_loaded"
    ) is not False:
        raise RuntimeError(
            "Primary 清单标签状态异常"
        )

    if primary_manifest.get(
        "sealed_test_labels_loaded"
    ) is not False:
        raise RuntimeError(
            "Primary 清单 Test 标签状态异常"
        )

    rows = read_jsonl(primary_path)
    questions = read_jsonl(questions_path)

    if len(rows) != source["expected_candidates"]:
        raise RuntimeError(
            f"Primary 候选数错误：{len(rows)}"
        )

    if len(questions) != source["expected_questions"]:
        raise RuntimeError(
            f"问题数错误：{len(questions)}"
        )

    unparsed = [
        row
        for row in rows
        if row["parsed_answer_generation_audit"]
        is None
    ]

    if (
        len(unparsed)
        != config["recovery"][
            "expected_attempted_candidates"
        ]
    ):
        raise RuntimeError(
            f"待恢复数错误：{len(unparsed)}"
        )

    question_by_uid = {
        row["question_uid"]: row
        for row in questions
    }

    if len(question_by_uid) != len(questions):
        raise RuntimeError("问题 UID 重复")

    if not all(
        row["question_uid"] in question_by_uid
        for row in rows
    ):
        raise RuntimeError(
            "候选中存在未知问题 UID"
        )

    groups = defaultdict(list)
    for row in rows:
        groups[row["question_uid"]].append(
            int(row["candidate_index"])
        )

    if len(groups) != len(questions):
        raise RuntimeError("候选问题数错误")

    for uid, indices in groups.items():
        if sorted(indices) != list(range(16)):
            raise RuntimeError(
                f"{uid}: candidate_index 不完整"
            )

    split_counts = Counter(
        row["logical_split"]
        for row in questions
    )
    expected_splits = Counter(
        source["expected_split_questions"]
    )

    if split_counts != expected_splits:
        raise RuntimeError(
            "恢复问题划分数量错误："
            f"{dict(split_counts)}"
        )

    forbidden_paths = [
        source["primary_candidates"],
        source["primary_manifest"],
        source["questions"],
    ]

    if any(
        "label" in value.casefold()
        for value in forbidden_paths
    ):
        raise RuntimeError(
            "恢复配置引用了标签路径"
        )

    model_config_path = (
        ROOT / config["model"]["path"]
        / "config.json"
    )

    if (
        sha256_file(model_config_path)
        != config["model"]["config_sha256"]
    ):
        raise RuntimeError(
            "模型配置 SHA256 不一致"
        )

    template = config["prompt"]["template"]
    if template.count("{problem}") != 1:
        raise RuntimeError(
            "Prompt 必须包含一个 {problem}"
        )
    if template.count("{valid_labels}") != 1:
        raise RuntimeError(
            "Prompt 必须包含一个 {valid_labels}"
        )

    print("===== ARC 恢复静态预检通过 =====")
    print("问题：", len(questions))
    print("Primary 候选：", len(rows))
    print("待恢复：", len(unparsed))
    print(
        "受影响问题：",
        len({
            row["question_uid"]
            for row in unparsed
        }),
    )
    print("标签读取：False")
    print("Test 问题使用：True")
    print("Test 标签读取：False")
    print(
        "恢复配置稳定哈希：",
        stable_hash(config),
    )

    return (
        config,
        rows,
        question_by_uid,
    )


def render_recovery_prompt(config, question):
    return (
        config["prompt"]["template"]
        .replace(
            "{problem}",
            question["problem"],
        )
        .replace(
            "{valid_labels}",
            ", ".join(
                str(value)
                for value
                in question["choice_labels"]
            ),
        )
    )


def promote_recovery(
    primary,
    response,
    parsed,
    parse_method,
    token_count,
    prompt_tokens,
    finished_by_eos,
    stopped_at_box,
    seed,
    recovery_config_sha256,
):
    updated = dict(primary)

    preserved_fields = [
        "solution_text",
        "seed",
        "prompt_tokens",
        "max_new_tokens",
        "generated_tokens",
        "finished_by_eos",
        "stopped_at_box",
        "generation_config_sha256",
        "generator_track",
    ]

    for key in preserved_fields:
        updated[f"primary_{key}"] = (
            primary.get(key)
        )

    choice_index = int(
        parsed.split(":", 1)[1]
    )

    updated.update({
        "solution_text": response,
        "parsed_answer_generation_audit": parsed,
        "parse_method_generation_audit": (
            f"recovery_{parse_method}"
        ),
        "parsed_choice_index_generation_audit": (
            choice_index
        ),
        "parsed_choice_label_generation_audit": (
            primary["choice_labels"][choice_index]
        ),
        "seed": seed,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": 192,
        "generated_tokens": token_count,
        "finished_by_eos": finished_by_eos,
        "stopped_at_box": stopped_at_box,
        "generation_config_sha256": (
            recovery_config_sha256
        ),
        "generator_track": (
            "ARC_qwen2_7b_full_k16_"
            "recovery_v1"
        ),
        "generation_stage": (
            "restart_finalize_recovery"
        ),
        "recovery_attempted": True,
        "recovery_success": True,
        "recovery_seed": seed,
        "recovery_prompt_tokens": prompt_tokens,
        "recovery_generated_tokens": token_count,
        "recovery_parse_method": parse_method,
        "recovery_config_sha256": (
            recovery_config_sha256
        ),
    })

    return updated


def retain_after_failed_recovery(
    primary,
    response,
    parse_method,
    token_count,
    prompt_tokens,
    finished_by_eos,
    seed,
    recovery_config_sha256,
):
    updated = dict(primary)

    updated.update({
        "generation_stage": (
            "primary_after_failed_recovery"
        ),
        "recovery_attempted": True,
        "recovery_success": False,
        "recovery_seed": seed,
        "recovery_prompt_tokens": prompt_tokens,
        "recovery_generated_tokens": token_count,
        "recovery_solution_text": response,
        "recovery_parse_method": parse_method,
        "recovery_finished_by_eos": (
            finished_by_eos
        ),
        "recovery_config_sha256": (
            recovery_config_sha256
        ),
    })

    return updated


def main():
    config, primary_rows, question_by_uid = (
        preflight()
    )

    if "--preflight" in sys.argv:
        return

    if OUTPUT_PATH.exists():
        raise RuntimeError(
            f"恢复输出已存在，拒绝覆盖：{OUTPUT_PATH}"
        )

    recovery_config_sha256 = stable_hash(config)
    recovery = config["recovery"]

    target_positions = defaultdict(list)
    output_rows = []

    for position, row in enumerate(primary_rows):
        updated = dict(row)

        if (
            row["parsed_answer_generation_audit"]
            is None
        ):
            target_positions[
                row["question_uid"]
            ].append(position)
        else:
            updated.update({
                "generation_stage": "primary",
                "recovery_attempted": False,
                "recovery_success": False,
                "recovery_generated_tokens": 0,
                "recovery_config_sha256": (
                    recovery_config_sha256
                ),
            })

        output_rows.append(updated)

    model_path = ROOT / config["model"]["path"]

    print("加载 tokenizer……", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    chat_hash = hashlib.sha256(
        (tokenizer.chat_template or "").encode(
            "utf-8"
        )
    ).hexdigest()

    if (
        chat_hash
        != config["model"]["chat_template_sha256"]
    ):
        raise RuntimeError(
            "chat template SHA256 不一致"
        )

    print("加载 Qwen2-7B……", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.eval()

    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    attempted = 0
    success = 0
    generated_total = 0

    target_items = sorted(
        target_positions.items()
    )

    for question_number, (
        uid,
        positions,
    ) in enumerate(target_items, start=1):
        question = question_by_uid[uid]
        prompt = render_recovery_prompt(
            config,
            question,
        )

        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
        ).to("cuda")

        prompt_tokens = int(
            inputs["input_ids"].shape[1]
        )
        seed = (
            recovery["base_seed"]
            + question_number * 100
        )

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        arc.CURRENT_QUESTION = question

        try:
            stopper = arc.ArcFinalChoiceStop(
                tokenizer,
                prompt_tokens,
            )

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    num_return_sequences=len(
                        positions
                    ),
                    max_new_tokens=recovery[
                        "max_new_tokens"
                    ],
                    do_sample=True,
                    temperature=recovery[
                        "temperature"
                    ],
                    top_p=recovery["top_p"],
                    top_k=recovery["top_k"],
                    pad_token_id=(
                        tokenizer.pad_token_id
                    ),
                    eos_token_id=(
                        model.generation_config
                        .eos_token_id
                    ),
                    stopping_criteria=(
                        StoppingCriteriaList([
                            stopper
                        ])
                    ),
                )

            eos_ids = (
                model.generation_config.eos_token_id
            )
            if isinstance(eos_ids, int):
                eos_ids = {eos_ids}
            else:
                eos_ids = set(eos_ids or [])

            for sequence, position in zip(
                generated,
                positions,
            ):
                token_ids = sequence[
                    prompt_tokens:
                ].tolist()

                trimmed = []
                finished_by_eos = False

                for token_id in token_ids:
                    trimmed.append(token_id)
                    if token_id in eos_ids:
                        finished_by_eos = True
                        break

                response = tokenizer.decode(
                    trimmed,
                    skip_special_tokens=True,
                ).strip()

                _, box_end = (
                    arc.first_final_choice_boxed(
                        response
                    )
                )
                stopped_at_box = box_end is not None

                if stopped_at_box:
                    response = response[
                        :box_end
                    ].rstrip()

                parsed, method = (
                    arc.parse_final_choice(response)
                )

                attempted += 1
                generated_total += len(trimmed)

                if parsed is not None:
                    success += 1
                    output_rows[position] = (
                        promote_recovery(
                            primary_rows[position],
                            response,
                            parsed,
                            method,
                            len(trimmed),
                            prompt_tokens,
                            finished_by_eos,
                            stopped_at_box,
                            seed,
                            recovery_config_sha256,
                        )
                    )
                else:
                    output_rows[position] = (
                        retain_after_failed_recovery(
                            primary_rows[position],
                            response,
                            method,
                            len(trimmed),
                            prompt_tokens,
                            finished_by_eos,
                            seed,
                            recovery_config_sha256,
                        )
                    )
        finally:
            arc.CURRENT_QUESTION = None

        print(
            f"{question_number}/"
            f"{len(target_items)} | "
            f"{question['source_id']} | "
            f"recovery={success}/{attempted}",
            flush=True,
        )

    elapsed = time.time() - started
    peak_gpu_gb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    atomic_jsonl(OUTPUT_PATH, output_rows)

    final_parsed = sum(
        row["parsed_answer_generation_audit"]
        is not None
        for row in output_rows
    )

    final_groups = defaultdict(set)
    for row in output_rows:
        answer = row[
            "parsed_answer_generation_audit"
        ]
        if answer is not None:
            final_groups[
                row["question_uid"]
            ].add(answer)

    cluster_counts = [
        len(final_groups[uid])
        for uid in question_by_uid
    ]

    methods = Counter(
        row["parse_method_generation_audit"]
        for row in output_rows
    )

    manifest = {
        "version": (
            "arc_challenge_qwen2_7b_"
            "full_k16_recovery_v1"
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "labels_loaded": False,
        "test_questions_used": True,
        "test_labels_used": False,
        "sealed_test_labels_loaded": False,
        "questions": len(question_by_uid),
        "candidates": len(output_rows),
        "primary_output_file": (
            config["source"][
                "primary_candidates"
            ]
        ),
        "primary_output_sha256": (
            config["source"][
                "primary_candidates_sha256"
            ]
        ),
        "recovery_config": str(
            CONFIG_PATH.relative_to(ROOT)
        ),
        "recovery_config_file_sha256": (
            sha256_file(CONFIG_PATH)
        ),
        "recovery_config_stable_sha256": (
            recovery_config_sha256
        ),
        "recovery": {
            "attempted": attempted,
            "success": success,
            "success_rate": (
                success / max(attempted, 1)
            ),
            "failed": attempted - success,
            "generated_tokens": (
                generated_total
            ),
        },
        "final_parse": {
            "parsed_candidates": final_parsed,
            "unparsed_candidates": (
                len(output_rows) - final_parsed
            ),
            "parse_rate": (
                final_parsed / len(output_rows)
            ),
            "methods": dict(methods),
        },
        "answer_clusters_per_question": {
            "min": min(cluster_counts),
            "median": float(np.median(
                cluster_counts
            )),
            "mean": float(np.mean(
                cluster_counts
            )),
            "max": max(cluster_counts),
            "multi_cluster_questions": sum(
                value > 1
                for value in cluster_counts
            ),
        },
        "elapsed_seconds": elapsed,
        "peak_gpu_gb": peak_gpu_gb,
        "output_file": str(
            OUTPUT_PATH.relative_to(ROOT)
        ),
        "output_sha256": sha256_file(
            OUTPUT_PATH
        ),
    }

    atomic_json(MANIFEST_PATH, manifest)

    print()
    print("===== ARC 无标签恢复完成 =====")
    print(json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ))
    print("输出：", OUTPUT_PATH)
    print("清单：", MANIFEST_PATH)


if __name__ == "__main__":
    main()

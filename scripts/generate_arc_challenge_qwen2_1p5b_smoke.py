from pathlib import Path
from collections import Counter, defaultdict
import json
import re
import sys

import numpy as np
import torch
from transformers import StoppingCriteria


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_fresh_math_qwen3_8b_nonthinking_smoke as base


CONFIG_PATH = (
    ROOT / "configs/"
    "arc_challenge_qwen2_1p5b_nonthinking_smoke_v1.json"
)
PARTS_DIR = (
    ROOT / "outputs/arc_challenge_v1/"
    "qwen2_1p5b_smoke_parts"
)
OUTPUT_PATH = (
    ROOT / "outputs/arc_challenge_v1/"
    "qwen2_1p5b_smoke_candidates.jsonl"
)
MANIFEST_PATH = (
    ROOT / "data/manifests/"
    "arc_challenge_qwen2_1p5b_nonthinking_smoke_v1.json"
)

CURRENT_QUESTION = None
ORIGINAL_GENERATE_QUESTION = base.generate_question


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def normalize_choice_text(value):
    value = str(value).strip()
    value = value.replace("\u2212", "-")
    value = value.replace("$", "")
    value = value.replace(r"\(", "")
    value = value.replace(r"\)", "")

    for _ in range(4):
        updated = re.sub(
            r"\\(?:text|textrm|mathrm|mathbf|"
            r"textbf|operatorname)\{([^{}]*)\}",
            r"\1",
            value,
        )
        if updated == value:
            break
        value = updated

    value = re.sub(
        r"(?i)^(?:final\s+answer|answer|option|choice)"
        r"\s*[:=\-]?\s*",
        "",
        value.strip(),
    )
    value = re.sub(
        r"^[\(\[\{\s]+",
        "",
        value,
    )
    value = re.sub(
        r"[\)\]\}\s.,;:!]+$",
        "",
        value,
    )
    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value.casefold()


def parse_choice_content(content):
    if CURRENT_QUESTION is None:
        return None

    normalized = normalize_choice_text(content)
    labels = CURRENT_QUESTION["choice_labels"]
    texts = CURRENT_QUESTION["choice_texts"]

    # 首选：只输出选项标签。
    for index, label in enumerate(labels):
        if normalized == normalize_choice_text(label):
            return {
                "index": index,
                "label": str(label),
                "method": "boxed_choice_label",
            }

    # 接受完整、唯一匹配的选项文本。
    text_matches = [
        index
        for index, text in enumerate(texts)
        if normalized == normalize_choice_text(text)
    ]
    if len(text_matches) == 1:
        index = text_matches[0]
        return {
            "index": index,
            "label": str(labels[index]),
            "method": "boxed_choice_text",
        }

    # 接受 “A. option text” 或 “A: option text”。
    for index, (label, text) in enumerate(
        zip(labels, texts)
    ):
        label_value = re.escape(
            normalize_choice_text(label)
        )
        match = re.fullmatch(
            label_value
            + r"\s*[\).:\-]\s*(.+)",
            normalized,
        )
        if (
            match is not None
            and normalize_choice_text(match.group(1))
            == normalize_choice_text(text)
        ):
            return {
                "index": index,
                "label": str(label),
                "method": (
                    "boxed_choice_label_and_text"
                ),
            }

    return None


def select_final_choice_box(text):
    valid_boxes = []

    for box in base.all_valid_boxed(text):
        parsed = parse_choice_content(
            box["content"]
        )
        if parsed is not None:
            item = dict(box)
            item["parsed"] = parsed
            valid_boxes.append(item)

    if not valid_boxes:
        return None

    cue_matches = list(re.finditer(
        r"(?i)(?:\*\*)?"
        r"final\s+answer"
        r"(?:\*\*)?",
        text,
    ))

    if cue_matches:
        cue_start = cue_matches[-1].start()
        after_cue = [
            box
            for box in valid_boxes
            if box["start"] >= cue_start
        ]
        if after_cue:
            return after_cue[0]

    think_close = text.rfind("</think>")
    if think_close >= 0:
        before_close = [
            box
            for box in valid_boxes
            if box["end"] <= think_close
        ]
        if before_close:
            return before_close[-1]

    return valid_boxes[-1]


def first_final_choice_boxed(text):
    selected = select_final_choice_box(text)

    if selected is None:
        return None, None

    return (
        selected["content"],
        selected["end"],
    )


def parse_final_choice(text):
    selected = select_final_choice_box(text)

    if selected is None:
        return None, "unparsed"

    parsed = selected["parsed"]

    return (
        f"choice:{parsed['index']}",
        parsed["method"],
    )


class ArcFinalChoiceStop(StoppingCriteria):
    """
    每个候选独立停止，并且只有合法选项才能触发停止。
    """

    def __init__(
        self,
        tokenizer,
        prompt_length,
        tail_tokens=1024,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)
        self.tail_tokens = int(tail_tokens)
        self.finished = []

    def __call__(
        self,
        input_ids,
        scores,
        **kwargs,
    ):
        batch_size = int(input_ids.shape[0])

        if len(self.finished) != batch_size:
            self.finished = [
                False
                for _ in range(batch_size)
            ]

        for index in range(batch_size):
            if self.finished[index]:
                continue

            sequence = input_ids[index]
            left = max(
                self.prompt_length,
                int(sequence.shape[0])
                - self.tail_tokens,
            )
            tail_text = self.tokenizer.decode(
                sequence[left:],
                skip_special_tokens=True,
            )

            if select_final_choice_box(
                tail_text
            ) is not None:
                self.finished[index] = True

        return torch.tensor(
            self.finished,
            dtype=torch.bool,
            device=input_ids.device,
        )


def generate_question(
    model,
    tokenizer,
    question,
    question_index,
    config,
    config_sha256,
):
    global CURRENT_QUESTION

    CURRENT_QUESTION = question

    try:
        part = ORIGINAL_GENERATE_QUESTION(
            model,
            tokenizer,
            question,
            question_index,
            config,
            config_sha256,
        )
    finally:
        CURRENT_QUESTION = None

    for row in part["candidates"]:
        parsed = row[
            "parsed_answer_generation_audit"
        ]

        choice_index = None
        choice_label = None

        if (
            isinstance(parsed, str)
            and parsed.startswith("choice:")
        ):
            choice_index = int(
                parsed.split(":", 1)[1]
            )
            choice_label = question[
                "choice_labels"
            ][choice_index]

        row.update({
            "generator": "Qwen2-1.5B",
            "task_type": (
                "multiple_choice_science"
            ),
            "source_split": question[
                "source_split"
            ],
            "logical_split": question[
                "logical_split"
            ],
            "data_role": question[
                "data_role"
            ],
            "source_id": question[
                "source_id"
            ],
            "question_text": question[
                "question"
            ],
            "choice_labels": question[
                "choice_labels"
            ],
            "choice_texts": question[
                "choice_texts"
            ],
            (
                "parsed_choice_index_"
                "generation_audit"
            ): choice_index,
            (
                "parsed_choice_label_"
                "generation_audit"
            ): choice_label,
            "generator_track": (
                "ARC_qwen2_1p5b_smoke_v1"
            ),
        })

    part["source_split"] = question[
        "source_split"
    ]
    part["logical_split"] = question[
        "logical_split"
    ]
    part["data_role"] = question[
        "data_role"
    ]

    return part



def model_aware_print(*args, **kwargs):
    translated = []

    for value in args:
        if isinstance(value, str):
            value = value.replace(
                "Qwen3-8B Non-Thinking",
                "Qwen2-1.5B",
            )
            value = value.replace(
                "Track B Non-Thinking",
                "ARC Qwen2-1.5B",
            )
        translated.append(value)

    __import__("builtins").print(
        *translated,
        **kwargs,
    )


def install_overrides():
    base.print = model_aware_print
    base.CONFIG_PATH = CONFIG_PATH
    base.PARTS_DIR = PARTS_DIR
    base.OUTPUT_PATH = OUTPUT_PATH
    base.MANIFEST_PATH = MANIFEST_PATH

    base.ThinkingFinalAnswerStop = (
        ArcFinalChoiceStop
    )
    base.first_final_boxed = (
        first_final_choice_boxed
    )
    base.parse_final_answer = (
        parse_final_choice
    )
    base.generate_question = (
        generate_question
    )


def preflight():
    config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    questions_path = (
        ROOT / config["dataset"]["questions"]
    )
    questions = read_jsonl(
        questions_path
    )

    if (
        base.sha256_file(questions_path)
        != config["dataset"][
            "questions_sha256"
        ]
    ):
        raise RuntimeError(
            "smoke 题目哈希不一致"
        )

    if (
        len(questions)
        != config["dataset"][
            "expected_questions"
        ]
    ):
        raise RuntimeError(
            "smoke 题数不一致"
        )

    uids = [
        row["question_uid"]
        for row in questions
    ]
    if len(uids) != len(set(uids)):
        raise RuntimeError(
            "smoke question_uid 重复"
        )

    for source in config["dataset"][
        "source_files"
    ].values():
        path = ROOT / source["path"]

        if "label" in source["path"].casefold():
            raise RuntimeError(
                "source_files 引用了标签路径"
            )

        if (
            base.sha256_file(path)
            != source["sha256"]
        ):
            raise RuntimeError(
                f"源问题哈希不一致：{path}"
            )

    model_config_path = (
        ROOT / config["model"]["path"]
        / "config.json"
    )
    if (
        base.sha256_file(model_config_path)
        != config["model"][
            "config_sha256"
        ]
    ):
        raise RuntimeError(
            "生成模型配置哈希不一致"
        )

    forbidden = {
        "answer",
        "answerkey",
        "gold",
        "target",
        "label",
    }

    for row in questions:
        overlap = forbidden.intersection(
            key.casefold()
            for key in row
        )
        if overlap:
            raise RuntimeError(
                "问题含标签字段："
                f"{sorted(overlap)}"
            )

        labels = row["choice_labels"]
        texts = row["choice_texts"]

        if len(labels) != len(texts):
            raise RuntimeError(
                "选项标签与文本长度不一致"
            )

        if len(set(labels)) != len(labels):
            raise RuntimeError(
                "选项标签重复"
            )

        prompt = base.render_prompt(
            config,
            row["problem"],
        )
        if "{problem}" in prompt:
            raise RuntimeError(
                "prompt 占位符未替换"
            )

        # 对每种合法标签做解析器自检。
        global CURRENT_QUESTION
        CURRENT_QUESTION = row

        try:
            for index, label in enumerate(
                labels
            ):
                probes = [
                    str(label),
                    f"({label})",
                    rf"\text{{{label}}}",
                    f"Option {label}",
                    (
                        f"{label}. "
                        f"{texts[index]}"
                    ),
                    str(texts[index]),
                ]

                for probe in probes:
                    parsed = parse_choice_content(
                        probe
                    )
                    if (
                        parsed is None
                        or parsed["index"] != index
                    ):
                        raise RuntimeError(
                            "解析器自检失败："
                            f"{row['source_id']} "
                            f"{probe!r}"
                        )
        finally:
            CURRENT_QUESTION = None

    print(
        "===== ARC smoke 静态预检通过 ====="
    )
    print("问题：", len(questions))
    print(
        "候选：",
        len(questions)
        * config["sampling"][
            "candidates_per_question"
        ],
    )
    print("标签读取：False")
    print(
        "分布：",
        dict(Counter(
            row["logical_split"]
            for row in questions
        )),
    )
    print(
        "选项数：",
        dict(Counter(
            len(row["choice_labels"])
            for row in questions
        )),
    )
    print(
        "标签格式：",
        dict(Counter(
            (
                "numeric"
                if all(
                    str(label).isdigit()
                    for label
                    in row["choice_labels"]
                )
                else "alphabetic"
            )
            for row in questions
        )),
    )


def finalize_manifest():
    rows = read_jsonl(OUTPUT_PATH)
    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    clusters = defaultdict(set)
    ordered_uids = []

    for row in rows:
        uid = row["question_uid"]

        if uid not in clusters:
            ordered_uids.append(uid)

        answer = row[
            "parsed_answer_generation_audit"
        ]
        if answer is not None:
            clusters[uid].add(answer)

    cluster_counts = [
        len(clusters[uid])
        for uid in ordered_uids
    ]

    k = manifest[
        "candidates_per_question"
    ]
    question_rows = rows[::k]

    manifest.update({
        "version": (
            "arc_challenge_qwen2_1p5b_"
            "nonthinking_smoke_v1"
        ),
        "task_type": (
            "multiple_choice_science"
        ),
        "generator_track": (
            "ARC_qwen2_1p5b_smoke_v1"
        ),
        "labels_loaded": False,
        "sealed_test_labels_loaded": False,
        "dataset_questions": dict(
            Counter(
                row["logical_split"]
                for row in question_rows
            )
        ),
        "choice_clusters_per_question": {
            "min": min(cluster_counts),
            "median": float(np.median(
                cluster_counts
            )),
            "max": max(cluster_counts),
            (
                "questions_with_no_"
                "parsed_choice"
            ): sum(
                value == 0
                for value in cluster_counts
            ),
        },
    })

    base.atomic_json(
        MANIFEST_PATH,
        manifest,
    )

    print()
    print(
        "===== ARC smoke 最终清单 ====="
    )
    print(json.dumps(
        {
            key: manifest[key]
            for key in [
                "questions",
                "candidates",
                "dataset_questions",
                "parse",
                (
                    "choice_clusters_"
                    "per_question"
                ),
                "generated_tokens",
                (
                    "elapsed_this_run_"
                    "seconds"
                ),
                "peak_gpu_gb",
                "output_file",
                "output_sha256",
            ]
        },
        ensure_ascii=False,
        indent=2,
    ))
    print("清单：", MANIFEST_PATH)


def main():
    install_overrides()

    if "--preflight" in sys.argv:
        preflight()
        return

    preflight()
    base.main()
    finalize_manifest()


if __name__ == "__main__":
    main()

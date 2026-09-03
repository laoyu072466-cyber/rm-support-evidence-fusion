from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/generator/Qwen2-7B"
QUESTIONS_PATH = (
    ROOT / "data/external/fresh_math_2026/questions.jsonl"
)
OUTPUT_PATH = (
    ROOT
    / "outputs/fresh_math_2026/"
    "qwen2_7b_smoke_4x4_v2.jsonl"
)
MANIFEST_PATH = (
    ROOT
    / "outputs/fresh_math_2026/"
    "qwen2_7b_smoke_4x4_v2_manifest.json"
)

K = 4
TOTAL_SEQUENCE_LIMIT = 2048
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 0
SEED = 20260901


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_questions():
    with QUESTIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    selected = []
    for dataset in [
        "AIME_2026",
        "HMMT_FEB_2026",
    ]:
        selected.extend([
            row
            for row in rows
            if row["source_dataset"] == dataset
        ][:2])

    if len(selected) != 4:
        raise RuntimeError(
            f"冒烟题目数量异常：{len(selected)}"
        )

    return selected


def first_valid_boxed(text):
    marker = r"\boxed"
    search_from = 0

    while True:
        start = text.find(marker, search_from)
        if start < 0:
            return None, None

        brace_start = text.find(
            "{",
            start + len(marker),
        )
        if brace_start < 0:
            return None, None

        between = text[
            start + len(marker):brace_start
        ]
        if between.strip():
            search_from = start + len(marker)
            continue

        depth = 0
        for index in range(
            brace_start,
            len(text),
        ):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    answer = text[
                        brace_start + 1:index
                    ].strip()

                    if (
                        answer
                        and len(answer) <= 256
                        and r"\boxed" not in answer
                    ):
                        return answer, index + 1

                    search_from = index + 1
                    break
        else:
            return None, None


def parse_answer(text):
    boxed, _ = first_valid_boxed(text)
    if boxed is not None:
        return boxed, "boxed"

    hash_matches = re.findall(
        r"####\s*([^\n]+)",
        text,
    )
    if hash_matches:
        return hash_matches[-1].strip(), "hash"

    phrase_matches = re.findall(
        (
            r"(?:final answer|answer is)"
            r"\s*[:=]?\s*"
            r"([^\n.!]+)"
        ),
        text,
        flags=re.IGNORECASE,
    )
    if phrase_matches:
        answer = phrase_matches[-1].strip()
        if (
            answer
            and r"\boxed{}" not in answer
            and len(answer) <= 256
        ):
            return answer, "final_phrase"

    return None, "unparsed"


class FirstBoxedAnswerStop(StoppingCriteria):
    def __init__(
        self,
        tokenizer,
        prompt_tokens,
    ):
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens

    def __call__(
        self,
        input_ids,
        scores,
        **kwargs,
    ):
        completed = []

        for sequence in input_ids:
            response = self.tokenizer.decode(
                sequence[self.prompt_tokens:],
                skip_special_tokens=True,
            )
            _, box_end = first_valid_boxed(
                response
            )
            completed.append(box_end is not None)

        return torch.tensor(
            completed,
            dtype=torch.bool,
            device=input_ids.device,
        )


def main():
    for path in [OUTPUT_PATH, MANIFEST_PATH]:
        if path.exists():
            raise FileExistsError(
                f"为避免覆盖，已停止：{path}"
            )

    questions = read_questions()

    print("加载 Qwen2-7B tokenizer……", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载 Qwen2-7B 模型……", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.eval()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()

    started = time.time()
    results = []
    generated_lengths = []
    parse_methods = Counter()
    eos_count = 0
    boxed_stop_count = 0

    print()
    print("===== 无标签生成冒烟开始 =====")
    print("题目：4")
    print("每题候选：", K)
    print("标准答案读取：False")
    print()

    for question_number, question in enumerate(
        questions,
        start=1,
    ):
        prompt = (
            question["problem"]
            + "\n\nSolve the problem step by step. "
            + "Put the final answer in \\boxed{}."
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
        max_new_tokens = (
            TOTAL_SEQUENCE_LIMIT - prompt_tokens
        )

        if max_new_tokens < 256:
            raise RuntimeError(
                "提示过长："
                f"{question['question_uid']} "
                f"prompt_tokens={prompt_tokens}"
            )

        question_started = time.time()
        stopper = FirstBoxedAnswerStop(
            tokenizer,
            prompt_tokens,
        )

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                num_return_sequences=K,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([
                    stopper
                ]),
            )

        question_elapsed = (
            time.time() - question_started
        )

        eos_ids = model.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = {eos_ids}
        else:
            eos_ids = set(eos_ids or [])

        question_answers = []

        for candidate_index, sequence in enumerate(
            generated
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

            raw_response = tokenizer.decode(
                trimmed,
                skip_special_tokens=True,
            ).strip()

            _, box_end = first_valid_boxed(
                raw_response
            )
            stopped_at_box = box_end is not None

            if stopped_at_box:
                response = raw_response[
                    :box_end
                ].rstrip()
            else:
                response = raw_response

            parsed_answer, parse_method = (
                parse_answer(response)
            )

            result = {
                "question_uid": question[
                    "question_uid"
                ],
                "source_dataset": question[
                    "source_dataset"
                ],
                "problem_id": question["problem_id"],
                "candidate_index": candidate_index,
                "generator": "Qwen2-7B",
                "generator_track": "A",
                "seed": SEED,
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": max_new_tokens,
                "generated_tokens": len(trimmed),
                "finished_by_eos": finished_by_eos,
                "stopped_at_box": stopped_at_box,
                "parse_method": parse_method,
                "parsed_answer": parsed_answer,
                "solution_text": response,
            }
            results.append(result)

            generated_lengths.append(len(trimmed))
            parse_methods[parse_method] += 1
            eos_count += int(finished_by_eos)
            boxed_stop_count += int(
                stopped_at_box
            )

            if parsed_answer is not None:
                question_answers.append(parsed_answer)

        print(
            f"{question_number}/4 "
            f"{question['source_dataset']} "
            f"problem_id={question['problem_id']} | "
            f"prompt={prompt_tokens} | "
            f"time={question_elapsed:.1f}s | "
            f"parsed={len(question_answers)}/{K} | "
            f"answer_clusters="
            f"{len(set(question_answers))}",
            flush=True,
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in results:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    elapsed = time.time() - started

    manifest = {
        "version": "fresh_math_2026_qwen2_7b_smoke_v2",
        "blind_protocol": {
            "sealed_labels_loaded": False,
            "questions": 4,
            "candidates_per_question": K,
            "generator_track": "A",
        },
        "generation": {
            "model": str(MODEL_PATH),
            "seed": SEED,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "total_sequence_limit": (
                TOTAL_SEQUENCE_LIMIT
            ),
        },
        "results": {
            "candidates": len(results),
            "parsed_candidates": (
                len(results)
                - parse_methods["unparsed"]
            ),
            "parse_rate": (
                (
                    len(results)
                    - parse_methods["unparsed"]
                )
                / len(results)
            ),
            "parse_methods": dict(parse_methods),
            "eos_rate": eos_count / len(results),
            "boxed_stop_rate": (
                boxed_stop_count / len(results)
            ),
            "generated_tokens": {
                "min": min(generated_lengths),
                "mean": (
                    sum(generated_lengths)
                    / len(generated_lengths)
                ),
                "max": max(generated_lengths),
            },
            "elapsed_seconds": round(elapsed, 3),
            "peak_gpu_gb": round(
                torch.cuda.max_memory_allocated()
                / 1024**3,
                3,
            ),
        },
        "output_file": str(
            OUTPUT_PATH.relative_to(ROOT)
        ),
    }

    manifest["output_sha256"] = sha256_file(
        OUTPUT_PATH
    )

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
    print("===== 冒烟汇总 =====")
    print(
        json.dumps(
            manifest["results"],
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("===== 每道题第一个候选预览 =====")
    seen = set()
    for row in results:
        if row["question_uid"] in seen:
            continue
        seen.add(row["question_uid"])

        print()
        print(
            row["source_dataset"],
            "problem_id=",
            row["problem_id"],
            "parsed=",
            row["parsed_answer"],
        )
        print(row["solution_text"][:1000])

    print()
    print("输出：", OUTPUT_PATH)
    print("清单：", MANIFEST_PATH)


if __name__ == "__main__":
    main()

from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

ROOT = Path("/root/autodl-tmp/rm_traj_project")
sys.path.insert(0, str(ROOT / "scripts"))

from smoke_generate_fresh_math_qwen2_7b import (
    first_valid_boxed,
    parse_answer,
)


from recover_fresh_math_qwen3_8b_smoke import (
    build_messages,
    decode_candidate,
    recovery_mode,
)


CONFIG_PATH = (
    ROOT
    / "configs/fresh_math_2026_qwen3_8b_adaptive_k16.json"
)
PARTS_DIR = (
    ROOT
    / "outputs/fresh_math_2026/qwen3_8b_adaptive_k16_parts"
)
OUTPUT_PATH = (
    ROOT
    / "outputs/fresh_math_2026/"
    "qwen3_8b_adaptive_k16_candidates.jsonl"
)
MANIFEST_PATH = (
    ROOT
    / "data/manifests/"
    "fresh_math_2026_qwen3_8b_adaptive_k16.json"
)




def all_valid_boxed(text):
    spans = []
    starts = []

    for marker in [r"\boxed{", r"\fbox{"]:
        cursor = 0
        while True:
            position = text.find(marker, cursor)
            if position < 0:
                break
            starts.append(position)
            cursor = position + 1

    for position in sorted(set(starts)):
        content, relative_end = first_valid_boxed(
            text[position:]
        )
        if relative_end is not None:
            spans.append({
                "content": content,
                "start": position,
                "end": position + relative_end,
            })

    return spans


def select_final_boxed(text):
    """
    接受以下两种完成信号：

    1. Final Answer 提示之后的 boxed；
    2. </think> 之前最后一个 boxed。
    """
    boxes = all_valid_boxed(text)
    if not boxes:
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
            box for box in boxes
            if box["start"] >= cue_start
        ]
        if after_cue:
            selected = after_cue[0].copy()
            selected["reason"] = "final_answer_cue"
            return selected

    think_close = text.rfind("</think>")
    if think_close >= 0:
        before_close = [
            box for box in boxes
            if box["end"] <= think_close
        ]
        if before_close:
            selected = before_close[-1].copy()
            selected["reason"] = "last_box_before_think_end"
            return selected

    # Non-thinking Prompt 明确禁止在中间步骤
    # 使用 boxed，因此最后一个完整 boxed 可以视为终答。
    if boxes:
        selected = boxes[-1].copy()
        selected["reason"] = "nonthinking_last_box"
        return selected

    return None


def first_final_boxed(text):
    selected = select_final_boxed(text)
    if selected is None:
        return None, None

    return selected["content"], selected["end"]


def parse_final_answer(text):
    selected = select_final_boxed(text)

    if selected is not None:
        synthetic = (
            r"\boxed{"
            + selected["content"]
            + "}"
        )
        return parse_answer(synthetic)

    think_close = text.rfind("</think>")
    if think_close >= 0:
        return parse_answer(
            text[
                think_close + len("</think>"):
            ]
        )

    return None, "unparsed"


class ThinkingFinalAnswerStop(StoppingCriteria):
    """
    每个候选独立停止。

    Final Answer + boxed 可以在 think 内结束；
    或在 </think> 出现后选择此前最后一个 boxed。
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

            if select_final_boxed(
                tail_text
            ) is not None:
                self.finished[index] = True

        return torch.tensor(
            self.finished,
            dtype=torch.bool,
            device=input_ids.device,
        )

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def percentile(values, q):
    return float(
        np.percentile(
            np.asarray(values, dtype=np.float64),
            q,
        )
    )


def atomic_json(path, value):
    temporary = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_questions(path):
    with path.open("r", encoding="utf-8") as file:
        rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    uids = [row["question_uid"] for row in rows]
    if len(uids) != len(set(uids)):
        raise RuntimeError("question_uid 重复")

    return rows


def part_path(question_index, question):
    return PARTS_DIR / (
        f"{question_index:03d}_"
        f"{question['question_uid']}.json"
    )


def validate_part(
    path,
    question,
    config_sha256,
    expected_k,
):
    try:
        content = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return False

    if content.get(
        "generation_config_sha256"
    ) != config_sha256:
        return False

    if content.get("question_uid") != question[
        "question_uid"
    ]:
        return False

    candidates = content.get("candidates")
    if not isinstance(candidates, list):
        return False

    if len(candidates) != expected_k:
        return False

    indices = sorted(
        row.get("candidate_index")
        for row in candidates
    )
    return indices == list(range(expected_k))


def render_prompt(config, problem):
    template = config["prompt"]["template"]

    if template.count("{problem}") != 1:
        raise RuntimeError(
            "prompt 模板必须且只能包含一个 "
            "{problem} 占位符"
        )

    return template.replace(
        "{problem}",
        problem,
    )



def recover_candidates(
    model,
    tokenizer,
    candidates,
    question_index,
    config,
):
    recovery = config["recovery"]

    for row in candidates:
        row["primary_generated_tokens"] = row[
            "generated_tokens"
        ]
        row["recovery_attempted"] = False
        row["recovery_success"] = False
        row["recovery_generated_tokens"] = 0
        row["generation_stage"] = "primary"

    unfinished = [
        row for row in candidates
        if row[
            "parsed_answer_generation_audit"
        ] is None
    ]

    summary = {
        "primary_parsed": (
            len(candidates) - len(unfinished)
        ),
        "attempted": len(unfinished),
        "success": 0,
        "final_parsed": 0,
        "modes": {},
        "recovery_tokens": 0,
    }

    if not unfinished:
        summary["final_parsed"] = len(candidates)
        return candidates, summary

    plans = []

    for row in unfinished:
        mode, score = recovery_mode(row)
        plans.append({
            "row": row,
            "mode": mode,
            "repetition_score": score,
        })

    mode_counts = Counter(
        plan["mode"] for plan in plans
    )
    summary["modes"] = dict(mode_counts)

    def apply_result(
        row,
        result,
        mode,
        repetition_score,
        seed,
    ):
        row["recovery_attempted"] = True
        row["recovery_generated_tokens"] = (
            result["generated_tokens"]
        )
        row["recovery_mode"] = mode
        row["recovery_seed"] = seed
        row["repetition_score"] = repetition_score

        if result["answer"] is None:
            row["generation_stage"] = (
                "primary_unresolved"
            )
            return False

        row["solution_text"] = result["response"]
        row[
            "parsed_answer_generation_audit"
        ] = result["answer"]
        row[
            "parse_method_generation_audit"
        ] = result["method"]
        row["finished_by_eos"] = result[
            "finished_by_eos"
        ]
        row["stopped_at_box"] = result[
            "stopped_at_box"
        ]
        row["recovery_success"] = True
        row["generation_stage"] = mode
        return True

    continuation = [
        plan for plan in plans
        if plan["mode"] == "continue_finalize"
    ]

    for plan in continuation:
        row = plan["row"]
        messages = build_messages(
            row,
            "continue_finalize",
        )

        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
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
            + question_index * 1000
            + row["candidate_index"]
        )
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        stopper = ThinkingFinalAnswerStop(
            tokenizer,
            prompt_tokens,
        )

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=recovery[
                    "max_new_tokens"
                ],
                do_sample=True,
                temperature=recovery[
                    "temperature"
                ],
                top_p=recovery["top_p"],
                top_k=recovery["top_k"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=(
                    model.generation_config.eos_token_id
                ),
                stopping_criteria=StoppingCriteriaList([
                    stopper
                ]),
            )

        result = decode_candidate(
            tokenizer,
            model,
            generated[0],
            prompt_tokens,
        )

        apply_result(
            row,
            result,
            "continue_finalize",
            plan["repetition_score"],
            seed,
        )

    restart = [
        plan for plan in plans
        if plan["mode"] == "restart_efficient"
    ]
    restart.sort(
        key=lambda plan: plan["row"][
            "candidate_index"
        ]
    )

    restart_batch = recovery[
        "restart_batch_candidates"
    ]

    for batch_start in range(
        0,
        len(restart),
        restart_batch,
    ):
        batch = restart[
            batch_start:
            batch_start + restart_batch
        ]
        if not batch:
            continue

        messages = build_messages(
            batch[0]["row"],
            "restart_efficient",
        )
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
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
            + question_index * 1000
            + 500
            + batch_start
        )
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        stopper = ThinkingFinalAnswerStop(
            tokenizer,
            prompt_tokens,
        )

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                num_return_sequences=len(batch),
                max_new_tokens=recovery[
                    "max_new_tokens"
                ],
                do_sample=True,
                temperature=recovery[
                    "temperature"
                ],
                top_p=recovery["top_p"],
                top_k=recovery["top_k"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=(
                    model.generation_config.eos_token_id
                ),
                stopping_criteria=StoppingCriteriaList([
                    stopper
                ]),
            )

        for sequence, plan in zip(
            generated,
            batch,
        ):
            row = plan["row"]
            result = decode_candidate(
                tokenizer,
                model,
                sequence,
                prompt_tokens,
            )

            apply_result(
                row,
                result,
                "restart_efficient",
                plan["repetition_score"],
                seed,
            )

    summary["success"] = sum(
        row["recovery_success"]
        for row in candidates
    )
    summary["final_parsed"] = sum(
        row[
            "parsed_answer_generation_audit"
        ] is not None
        for row in candidates
    )
    summary["recovery_tokens"] = sum(
        row["recovery_generated_tokens"]
        for row in candidates
    )

    return candidates, summary

def generate_question(
    model,
    tokenizer,
    question,
    question_index,
    config,
    config_sha256,
):
    sampling = config["sampling"]
    k = sampling["candidates_per_question"]
    batch_candidates = sampling[
        "batch_candidates"
    ]

    prompt = render_prompt(
        config,
        question["problem"],
    )
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
    ).to("cuda")

    prompt_tokens = int(
        inputs["input_ids"].shape[1]
    )
    max_new_tokens = (
        sampling["total_sequence_limit"]
        - prompt_tokens
    )
    if max_new_tokens < 256:
        raise RuntimeError(
            f"{question['question_uid']} 提示过长："
            f"{prompt_tokens}"
        )

    candidates = []

    for batch_start in range(
        0,
        k,
        batch_candidates,
    ):
        current_batch = min(
            batch_candidates,
            k - batch_start,
        )
        batch_seed = (
            sampling["base_seed"]
            + question_index * 100
            + batch_start
        )

        torch.manual_seed(batch_seed)
        torch.cuda.manual_seed_all(batch_seed)

        stopper = ThinkingFinalAnswerStop(
            tokenizer,
            prompt_tokens,
        )

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                num_return_sequences=current_batch,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=sampling[
                    "temperature"
                ],
                top_p=sampling["top_p"],
                top_k=sampling["top_k"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=model.generation_config.eos_token_id,
                stopping_criteria=StoppingCriteriaList([
                    stopper
                ]),
            )

        eos_ids = model.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = {eos_ids}
        else:
            eos_ids = set(eos_ids or [])

        for local_index, sequence in enumerate(
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

            _, box_end = first_final_boxed(
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
                parse_final_answer(response)
            )

            # 下游奖励模型接收普通解答文本，
            # 不保留未闭合的 thinking 包装。
            response = re.sub(
                r"^\s*<think>\s*",
                "",
                response,
                count=1,
            )
            response = response.replace(
                "</think>",
                "",
            ).strip()

            candidate_index = (
                batch_start + local_index
            )

            candidates.append({
                "question_uid": question[
                    "question_uid"
                ],
                "source_dataset": question[
                    "source_dataset"
                ],
                "data_role": "fresh_exploratory_test",
                "problem_id": question["problem_id"],
                "problem": question["problem"],
                "candidate_index": candidate_index,
                "solution_text": response,
                "parsed_answer_generation_audit": (
                    parsed_answer
                ),
                "parse_method_generation_audit": (
                    parse_method
                ),
                "generator": "Qwen3-8B",
                "generator_track": "B_adaptive",
                "seed": batch_seed,
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": max_new_tokens,
                "generated_tokens": len(trimmed),
                "finished_by_eos": finished_by_eos,
                "stopped_at_box": stopped_at_box,
                "has_think_end": (
                    "</think>" in raw_response
                ),
                "generation_config_sha256": (
                    config_sha256
                ),
            })

    candidates.sort(
        key=lambda row: row["candidate_index"]
    )

    candidates, recovery_summary = (
        recover_candidates(
            model,
            tokenizer,
            candidates,
            question_index,
            config,
        )
    )

    return {
        "question_uid": question["question_uid"],
        "source_dataset": question[
            "source_dataset"
        ],
        "problem_id": question["problem_id"],
        "generation_config_sha256": config_sha256,
        "recovery_summary": recovery_summary,
        "candidates": candidates,
    }


def main():
    started = time.time()
    config = json.loads(
        CONFIG_PATH.read_text(encoding="utf-8")
    )
    config_sha256 = stable_hash(config)

    questions_path = (
        ROOT / config["dataset"]["questions"]
    )
    if sha256_file(questions_path) != config[
        "dataset"
    ]["questions_sha256"]:
        raise RuntimeError("冻结题目哈希不一致")

    questions = read_questions(questions_path)
    expected_questions = config["dataset"][
        "expected_questions"
    ]
    if len(questions) != expected_questions:
        raise RuntimeError(
            f"题数异常：{len(questions)}"
        )

    question_indices = config["dataset"].get(
        "question_indices"
    )
    if question_indices is not None:
        if (
            len(question_indices)
            != len(set(question_indices))
        ):
            raise RuntimeError(
                "冒烟问题索引存在重复"
            )
        if any(
            index < 0 or index >= len(questions)
            for index in question_indices
        ):
            raise RuntimeError(
                "冒烟问题索引越界"
            )
        questions = [
            questions[index]
            for index in question_indices
        ]

    # 在加载大模型前预检全部 prompt。
    rendered_prompts = [
        render_prompt(
            config,
            question["problem"],
        )
        for question in questions
    ]
    if any(
        "{problem}" in prompt
        for prompt in rendered_prompts
    ):
        raise RuntimeError(
            "存在未替换的 problem 占位符"
        )
    print(
        "Prompt 渲染预检通过：",
        len(rendered_prompts),
        flush=True,
    )

    k = config["sampling"][
        "candidates_per_question"
    ]
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    missing = []
    completed_before_start = 0

    for question_index, question in enumerate(
        questions
    ):
        path = part_path(question_index, question)
        if path.exists() and validate_part(
            path,
            question,
            config_sha256,
            k,
        ):
            completed_before_start += 1
        else:
            missing.append((question_index, question))

    print("===== Qwen3-8B 自适应 K16 全量生成 =====")
    print("问题总数：", len(questions))
    print("每题候选：", k)
    print("已完成问题：", completed_before_start)
    print("待生成问题：", len(missing))
    print("密封标签读取：False")
    print("配置哈希：", config_sha256)

    peak_gpu = 0.0

    if missing:
        model_path = ROOT / config["model"]["path"]

        print("加载 tokenizer……", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = (
                tokenizer.eos_token
            )

        chat_hash = hashlib.sha256(
            (tokenizer.chat_template or "").encode(
                "utf-8"
            )
        ).hexdigest()
        if chat_hash != config["model"][
            "chat_template_sha256"
        ]:
            raise RuntimeError(
                "chat template 哈希不一致"
            )

        print("加载 Qwen3-8B Non-Thinking……", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
            local_files_only=True,
        )
        model.eval()
        torch.cuda.reset_peak_memory_stats()

        run_started = time.time()

        for run_index, (
            question_index,
            question,
        ) in enumerate(missing, start=1):
            question_started = time.time()

            part = generate_question(
                model,
                tokenizer,
                question,
                question_index,
                config,
                config_sha256,
            )
            atomic_json(
                part_path(question_index, question),
                part,
            )

            candidates = part["candidates"]
            recovery_summary = part[
                "recovery_summary"
            ]
            parsed = sum(
                row[
                    "parsed_answer_generation_audit"
                ] is not None
                for row in candidates
            )
            unique_answers = len({
                str(
                    row[
                        "parsed_answer_generation_audit"
                    ]
                ).strip()
                for row in candidates
                if row[
                    "parsed_answer_generation_audit"
                ] is not None
            })

            elapsed = time.time() - question_started
            average = (
                (time.time() - run_started)
                / run_index
            )
            remaining = len(missing) - run_index
            eta_minutes = (
                average * remaining / 60
            )

            print(
                f"{completed_before_start + run_index}"
                f"/{len(questions)} | "
                f"{question['source_dataset']} "
                f"id={question['problem_id']} | "
                f"parsed={parsed}/{k} | "
                f"clusters={unique_answers} | "
                f"recovery="
                f"{recovery_summary['success']}/"
                f"{recovery_summary['attempted']} | "
                f"time={elapsed:.1f}s | "
                f"ETA={eta_minutes:.1f}min",
                flush=True,
            )

        peak_gpu = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

    all_candidates = []
    part_hashes = {}

    for question_index, question in enumerate(
        questions
    ):
        path = part_path(question_index, question)

        if not validate_part(
            path,
            question,
            config_sha256,
            k,
        ):
            raise RuntimeError(
                f"无效或缺失 part：{path}"
            )

        part = json.loads(
            path.read_text(encoding="utf-8")
        )
        all_candidates.extend(
            part["candidates"]
        )
        part_hashes[path.name] = sha256_file(path)

    expected_candidates = len(questions) * k
    if len(all_candidates) != expected_candidates:
        raise RuntimeError(
            "候选数异常："
            f"{len(all_candidates)} "
            f"!= {expected_candidates}"
        )

    temporary = OUTPUT_PATH.with_suffix(
        ".jsonl.tmp"
    )
    with temporary.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in all_candidates:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )
    temporary.replace(OUTPUT_PATH)

    lengths = [
        row["generated_tokens"]
        for row in all_candidates
    ]
    parse_methods = Counter(
        row["parse_method_generation_audit"]
        for row in all_candidates
    )
    parsed_count = sum(
        row[
            "parsed_answer_generation_audit"
        ] is not None
        for row in all_candidates
    )
    boxed_count = sum(
        row["stopped_at_box"]
        for row in all_candidates
    )

    recovery_attempted = sum(
        row.get("recovery_attempted", False)
        for row in all_candidates
    )
    recovery_success = sum(
        row.get("recovery_success", False)
        for row in all_candidates
    )
    recovery_tokens = sum(
        int(row.get(
            "recovery_generated_tokens",
            0,
        ))
        for row in all_candidates
    )
    recovery_modes = Counter(
        row.get("recovery_mode")
        for row in all_candidates
        if row.get("recovery_attempted")
    )

    cluster_counts = defaultdict(set)
    for row in all_candidates:
        answer = row[
            "parsed_answer_generation_audit"
        ]
        if answer is not None:
            cluster_counts[row["question_uid"]].add(
                str(answer).strip()
            )


    cluster_sizes = [
        len(cluster_counts[
            question["question_uid"]
        ])
        for question in questions
    ]

    manifest = {
        "version": (
            "fresh_math_2026_qwen3_8b_adaptive_k16_candidates_v1"
        ),
        "generator_track": "B_adaptive",
        "labels_loaded": False,
        "generation_config": str(
            CONFIG_PATH.relative_to(ROOT)
        ),
        "generation_config_sha256": config_sha256,
        "questions": len(questions),
        "candidates_per_question": k,
        "candidates": len(all_candidates),
        "dataset_questions": dict(Counter(
            row["source_dataset"]
            for row in questions
        )),
        "parse": {
            "parsed_candidates": parsed_count,
            "parse_rate": (
                parsed_count / len(all_candidates)
            ),
            "methods": dict(parse_methods),
            "boxed_stop_rate": (
                boxed_count / len(all_candidates)
            ),
        },
        "recovery": {
            "attempted": recovery_attempted,
            "success": recovery_success,
            "success_rate": (
                recovery_success
                / recovery_attempted
                if recovery_attempted
                else 0.0
            ),
            "modes": dict(recovery_modes),
            "generated_tokens": recovery_tokens,
            "final_unparsed": (
                len(all_candidates) - parsed_count
            ),
        },
        "generated_tokens": {
            "min": min(lengths),
            "median": percentile(lengths, 50),
            "p90": percentile(lengths, 90),
            "p95": percentile(lengths, 95),
            "max": max(lengths),
            "mean": float(np.mean(lengths)),
        },
        "answer_clusters_per_question": {
            "min": min(cluster_sizes),
            "median": percentile(
                cluster_sizes,
                50,
            ),
            "max": max(cluster_sizes),
            "questions_with_no_parsed_answer": sum(
                value == 0
                for value in cluster_sizes
            ),
        },
        "output_file": str(
            OUTPUT_PATH.relative_to(ROOT)
        ),
        "output_sha256": sha256_file(OUTPUT_PATH),
        "part_files": len(part_hashes),
        "part_sha256": part_hashes,
        "elapsed_this_run_seconds": round(
            time.time() - started,
            3,
        ),
        "peak_gpu_gb": round(peak_gpu, 3),
    }

    atomic_json(MANIFEST_PATH, manifest)

    print()
    print("===== Track B 自适应 K16 全量生成完成 =====")
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in [
                    "questions",
                    "candidates",
                    "dataset_questions",
                    "parse",
                    "recovery",
                    "generated_tokens",
                    "answer_clusters_per_question",
                    "elapsed_this_run_seconds",
                    "peak_gpu_gb",
                    "output_file",
                    "output_sha256",
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("清单：", MANIFEST_PATH)


if __name__ == "__main__":
    main()

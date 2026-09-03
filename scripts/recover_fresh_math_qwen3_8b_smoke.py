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
    StoppingCriteriaList,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_fresh_math_qwen3_8b_nonthinking_smoke import (
    ThinkingFinalAnswerStop,
    parse_final_answer,
    select_final_boxed,
)


MODEL_PATH = (
    ROOT / "models/generator/Qwen3-8B"
)
SOURCE_PATH = (
    ROOT / "outputs/fresh_math_2026/"
    "qwen3_8b_nonthinking_smoke_candidates.jsonl"
)
OUTPUT_PATH = (
    ROOT / "outputs/fresh_math_2026/"
    "qwen3_8b_adaptive_recovery_smoke_candidates.jsonl"
)
MANIFEST_PATH = (
    ROOT / "data/manifests/"
    "fresh_math_2026_qwen3_8b_"
    "adaptive_recovery_smoke.json"
)

MAX_NEW_TOKENS = 1536
BASE_SEED = 20260905


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def atomic_text(path, text):
    temporary = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )
    temporary.write_text(
        text,
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def repetition_score(text):
    tail = text[-2500:].lower()

    patterns = [
        "→ no",
        "not an integer",
        "\\quad \\text{not",
        "try ",
        "divisible",
    ]

    return sum(
        tail.count(pattern)
        for pattern in patterns
    )


def recovery_mode(row):
    score = repetition_score(
        row["solution_text"]
    )

    if score >= 10:
        return "restart_efficient", score

    return "continue_finalize", score


def render_primary_prompt(problem):
    return (
        problem
        + "\n\n"
        + "Solve the problem independently, clearly, "
        + "and efficiently. Do not restart or repeat "
        + "the solution. Do not use \\boxed{} for "
        + "intermediate results. Use \\boxed{} exactly "
        + "once, only for the final answer. Finish with "
        + "exactly:\n**Final Answer**\n"
        + "\\boxed{final answer}"
    )


def build_messages(row, mode):
    problem = row["problem"]

    if mode == "continue_finalize":
        return [
            {
                "role": "user",
                "content": render_primary_prompt(
                    problem
                ),
            },
            {
                "role": "assistant",
                "content": row["solution_text"],
            },
            {
                "role": "user",
                "content": (
                    "Stop now. Use the work already "
                    "completed above; do not recompute or "
                    "repeat it. State the resulting final "
                    "answer only, in exactly this form:\n"
                    "**Final Answer**\n"
                    "\\boxed{final answer}"
                ),
            },
        ]

    return [{
        "role": "user",
        "content": (
            problem
            + "\n\n"
            + "Solve this problem again from scratch "
            + "using an efficient symbolic argument. "
            + "A previous attempt failed because it "
            + "enumerated a long sequence of individual "
            + "cases. Do not enumerate candidates one by "
            + "one. Use an algebraic, modular, counting, "
            + "or structural shortcut as appropriate. "
            + "Keep the solution under 1000 tokens. "
            + "Do not use \\boxed{} for intermediate "
            + "results. Finish with exactly:\n"
            + "**Final Answer**\n"
            + "\\boxed{final answer}"
        ),
    }]


def decode_candidate(
    tokenizer,
    model,
    sequence,
    prompt_tokens,
):
    eos_ids = model.generation_config.eos_token_id

    if isinstance(eos_ids, int):
        eos_ids = {eos_ids}
    else:
        eos_ids = set(eos_ids or [])

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

    selected = select_final_boxed(
        raw_response
    )
    stopped_at_box = selected is not None

    if selected is not None:
        response = raw_response[
            :selected["end"]
        ].rstrip()
    else:
        response = raw_response

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

    answer, method = parse_final_answer(
        raw_response
    )

    return {
        "response": response,
        "answer": answer,
        "method": method,
        "generated_tokens": len(trimmed),
        "finished_by_eos": finished_by_eos,
        "stopped_at_box": stopped_at_box,
    }


def main():
    started = time.time()
    rows = read_jsonl(SOURCE_PATH)

    unfinished = [
        row for row in rows
        if row.get(
            "parsed_answer_generation_audit"
        ) is None
    ]

    print("===== Qwen3 自适应恢复冒烟 =====")
    print("原候选：", len(rows))
    print("待恢复：", len(unfinished))
    print("密封标签读取：False")

    plans = []

    for row in unfinished:
        mode, score = recovery_mode(row)
        plans.append({
            "row": row,
            "mode": mode,
            "repetition_score": score,
        })
        print(
            row["source_dataset"],
            f"id={row['problem_id']}",
            f"candidate={row['candidate_index']}",
            f"mode={mode}",
            f"repeat={score}",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.eval()
    torch.cuda.reset_peak_memory_stats()

    recovered = {}

    # 续写候选逐条处理；每条上下文不同。
    continuation = [
        plan for plan in plans
        if plan["mode"] == "continue_finalize"
    ]

    for plan_index, plan in enumerate(
        continuation
    ):
        row = plan["row"]
        messages = build_messages(
            row,
            plan["mode"],
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

        seed = BASE_SEED + plan_index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        stopper = ThinkingFinalAnswerStop(
            tokenizer,
            prompt_tokens,
        )

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
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
        result.update({
            "mode": plan["mode"],
            "repetition_score": plan[
                "repetition_score"
            ],
            "seed": seed,
        })

        key = (
            row["question_uid"],
            row["candidate_index"],
        )
        recovered[key] = result

        print(
            "恢复：",
            row["source_dataset"],
            f"id={row['problem_id']}",
            f"candidate={row['candidate_index']}",
            f"parsed={result['answer'] is not None}",
            f"tokens={result['generated_tokens']}",
            flush=True,
        )

    # 相同问题的独立重启候选可以批量生成。
    restart_groups = defaultdict(list)

    for plan in plans:
        if plan["mode"] == "restart_efficient":
            restart_groups[
                plan["row"]["question_uid"]
            ].append(plan)

    seed_offset = 100

    for group_index, group in enumerate(
        restart_groups.values()
    ):
        group.sort(
            key=lambda item: item["row"][
                "candidate_index"
            ]
        )
        first = group[0]
        messages = build_messages(
            first["row"],
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
            BASE_SEED
            + seed_offset
            + group_index * 100
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
                num_return_sequences=len(group),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=(
                    model.generation_config.eos_token_id
                ),
                stopping_criteria=StoppingCriteriaList([
                    stopper
                ]),
            )

        for local_index, (
            sequence,
            plan,
        ) in enumerate(zip(generated, group)):
            row = plan["row"]
            result = decode_candidate(
                tokenizer,
                model,
                sequence,
                prompt_tokens,
            )
            result.update({
                "mode": "restart_efficient",
                "repetition_score": plan[
                    "repetition_score"
                ],
                "seed": seed,
                "local_index": local_index,
            })

            key = (
                row["question_uid"],
                row["candidate_index"],
            )
            recovered[key] = result

            print(
                "恢复：",
                row["source_dataset"],
                f"id={row['problem_id']}",
                f"candidate={row['candidate_index']}",
                f"parsed={result['answer'] is not None}",
                f"tokens={result['generated_tokens']}",
                flush=True,
            )

    merged = []

    for row in rows:
        key = (
            row["question_uid"],
            row["candidate_index"],
        )
        result = recovered.get(key)
        updated = dict(row)

        updated["primary_generated_tokens"] = row[
            "generated_tokens"
        ]
        updated["recovery_attempted"] = (
            result is not None
        )

        if (
            result is not None
            and result["answer"] is not None
        ):
            updated["solution_text"] = result[
                "response"
            ]
            updated[
                "parsed_answer_generation_audit"
            ] = result["answer"]
            updated[
                "parse_method_generation_audit"
            ] = result["method"]
            updated["generator_track"] = (
                "B_nonthinking_adaptive"
            )
            updated["generation_stage"] = result[
                "mode"
            ]
            updated["recovery_seed"] = result[
                "seed"
            ]
            updated["recovery_generated_tokens"] = (
                result["generated_tokens"]
            )
            updated["repetition_score"] = result[
                "repetition_score"
            ]
            updated["finished_by_eos"] = result[
                "finished_by_eos"
            ]
            updated["stopped_at_box"] = result[
                "stopped_at_box"
            ]
        else:
            updated["generation_stage"] = "primary"

        merged.append(updated)

    output_text = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
        ) + "\n"
        for row in merged
    )
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    atomic_text(OUTPUT_PATH, output_text)

    parsed_before = sum(
        row.get(
            "parsed_answer_generation_audit"
        ) is not None
        for row in rows
    )
    parsed_after = sum(
        row.get(
            "parsed_answer_generation_audit"
        ) is not None
        for row in merged
    )
    recovery_success = sum(
        result["answer"] is not None
        for result in recovered.values()
    )

    groups = defaultdict(list)
    for row in merged:
        groups[row["question_uid"]].append(row)

    cluster_counts = []
    for items in groups.values():
        cluster_counts.append(len({
            str(row[
                "parsed_answer_generation_audit"
            ])
            for row in items
            if row.get(
                "parsed_answer_generation_audit"
            ) is not None
        }))

    manifest = {
        "version": (
            "fresh_math_2026_qwen3_8b_"
            "adaptive_recovery_smoke_v1"
        ),
        "labels_loaded": False,
        "questions": len(groups),
        "candidates": len(merged),
        "primary_parsed": parsed_before,
        "primary_parse_rate": (
            parsed_before / len(rows)
        ),
        "recovery_attempted": len(recovered),
        "recovery_success": recovery_success,
        "recovery_success_rate": (
            recovery_success / len(recovered)
            if recovered else 0.0
        ),
        "final_parsed": parsed_after,
        "final_parse_rate": (
            parsed_after / len(merged)
        ),
        "questions_with_parsed_candidate": sum(
            count > 0 for count in cluster_counts
        ),
        "answer_clusters_per_question": {
            "min": min(cluster_counts),
            "median": float(np.median(
                cluster_counts
            )),
            "max": max(cluster_counts),
        },
        "recovery_modes": dict(Counter(
            result["mode"]
            for result in recovered.values()
        )),
        "elapsed_seconds": round(
            time.time() - started,
            3,
        ),
        "peak_gpu_gb": round(
            torch.cuda.max_memory_allocated()
            / 1024**3,
            3,
        ),
        "output_file": str(
            OUTPUT_PATH.relative_to(ROOT)
        ),
        "output_sha256": sha256_file(
            OUTPUT_PATH
        ),
    }

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    atomic_text(
        MANIFEST_PATH,
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
    )

    print()
    print("===== 自适应恢复结果 =====")
    print(json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ))
    print("结果：", OUTPUT_PATH)
    print("清单：", MANIFEST_PATH)


if __name__ == "__main__":
    main()

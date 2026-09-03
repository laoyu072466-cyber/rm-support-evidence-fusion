from pathlib import Path
from collections import OrderedDict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
import re
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cisc_ptrue_smoke.json"
OUTPUT_DIR = ROOT / "data/cache/cisc_ptrue_smoke_v1"
MANIFEST_PATH = (
    ROOT / "data/manifests/cisc_ptrue_smoke_v1.json"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
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
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def boxed_spans(text):
    spans = []
    for marker in [r"\boxed{", r"\fbox{"]:
        start = 0
        while True:
            index = text.find(marker, start)
            if index < 0:
                break
            cursor = index + len(marker)
            depth = 1
            while cursor < len(text) and depth:
                if text[cursor] == "{":
                    depth += 1
                elif text[cursor] == "}":
                    depth -= 1
                cursor += 1
            if depth == 0:
                content = text[
                    index + len(marker):cursor - 1
                ].strip()
                if content:
                    spans.append((index, cursor))
            start = index + 1
    return sorted(spans)


def truncate_through_answer(text):
    hash_matches = list(re.finditer(
        r"####\s*[^\n\r]+",
        text,
    ))
    box_matches = boxed_spans(text)

    endings = []
    if hash_matches:
        endings.append((
            hash_matches[-1].end(),
            "hash",
        ))
    if box_matches:
        endings.append((
            box_matches[-1][1],
            "boxed",
        ))

    if not endings:
        return text.strip(), "full_response"

    end, method = max(endings, key=lambda item: item[0])
    return text[:end].rstrip(), method


def project_row(row):
    return {
        "question_uid": row["question_uid"],
        "problem_id": row.get("problem_id"),
        "candidate_index": int(row["candidate_index"]),
        "problem": str(row["problem"]),
        "solution_text": str(row["solution_text"]),
    }


def select_smoke_rows(path, questions, candidates):
    groups = OrderedDict()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = project_row(json.loads(line))
            groups.setdefault(
                row["question_uid"], []
            ).append(row)

    selected = []
    for uid, rows in list(groups.items())[:questions]:
        rows.sort(key=lambda row: row["candidate_index"])
        if len(rows) < candidates:
            raise RuntimeError(
                f"{uid} 只有 {len(rows)} 个候选"
            )
        selected.extend(rows[:candidates])

    expected = questions * candidates
    if len(selected) != expected:
        raise RuntimeError(
            f"候选数不匹配：{len(selected)} != {expected}"
        )
    return selected


def render_confidence_prompt(
    tokenizer,
    row,
    instruction,
    confidence_suffix,
):
    response, answer_method = truncate_through_answer(
        row["solution_text"]
    )
    messages = [
        {
            "role": "user",
            "content": row["problem"] + instruction,
        },
        {
            "role": "assistant",
            "content": response + confidence_suffix,
        },
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    if not formatted.endswith("("):
        raise RuntimeError(
            "CISC prompt 没有停在左括号后"
        )
    return formatted, response, answer_method


def score_prompts(
    model,
    tokenizer,
    prompts,
    option_ids,
    max_input_tokens,
):
    lengths = [
        len(tokenizer.encode(
            prompt,
            add_special_tokens=False,
        ))
        for prompt in prompts
    ]
    if max(lengths) > max_input_tokens:
        raise RuntimeError(
            f"输入超过上限：{max(lengths)} > "
            f"{max_input_tokens}"
        )

    batch = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to("cuda")

    with torch.inference_mode():
        output = model(
            **batch,
            use_cache=False,
        )
        last_positions = (
            batch["attention_mask"].sum(dim=1) - 1
        )
        batch_indices = torch.arange(
            len(prompts),
            device=last_positions.device,
        )
        next_logits = output.logits[
            batch_indices,
            last_positions,
        ].float()
        binary_logits = next_logits[:, option_ids]
        binary_probs = torch.softmax(
            binary_logits,
            dim=-1,
        )

    results = []
    for index in range(len(prompts)):
        logit_0 = float(binary_logits[index, 0].item())
        logit_1 = float(binary_logits[index, 1].item())
        p_true = float(binary_probs[index, 1].item())
        results.append({
            "input_tokens": int(lengths[index]),
            "logit_0": logit_0,
            "logit_1": logit_1,
            "logit_confidence": logit_1 - logit_0,
            "p_true_binary_normalized": p_true,
        })

    del output, batch, next_logits
    del binary_logits, binary_probs
    return results


def synthetic_probe(
    model,
    tokenizer,
    config,
    option_ids,
):
    rows = [
        {
            "question_uid": "synthetic_correct",
            "problem": "What is 1+1?",
            "solution_text": (
                "We compute 1+1=2. "
                "Therefore the answer is \\boxed{2}."
            ),
        },
        {
            "question_uid": "synthetic_wrong",
            "problem": "What is 1+1?",
            "solution_text": (
                "We compute 1+1=3. "
                "Therefore the answer is \\boxed{3}."
            ),
        },
    ]
    prompts = []
    for row in rows:
        prompt, _, _ = render_confidence_prompt(
            tokenizer,
            row,
            config["input_instruction"],
            config["confidence_suffix"],
        )
        prompts.append(prompt)
    scores = score_prompts(
        model,
        tokenizer,
        prompts,
        option_ids,
        config["max_input_tokens"],
    )
    return {
        row["question_uid"]: score
        for row, score in zip(rows, scores)
    }


def read_existing(path, config_sha256):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["config_sha256"] != config_sha256:
                raise RuntimeError(
                    f"旧缓存配置不一致：{path}"
                )
            rows.append(row)
    return rows


def summarize_rows(rows):
    values = np.asarray([
        row["p_true_binary_normalized"]
        for row in rows
    ], dtype=np.float64)
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "std": float(values.std()),
    }


def run_experiment(config, spec, config_sha256):
    dataset_path = ROOT / spec["dataset"]
    model_path = ROOT / spec["model"]
    output_path = OUTPUT_DIR / f"{spec['name']}.jsonl"

    selected = select_smoke_rows(
        dataset_path,
        spec["questions"],
        spec["candidates_per_question"],
    )
    expected_keys = {
        (row["question_uid"], row["candidate_index"])
        for row in selected
    }
    existing = read_existing(
        output_path,
        config_sha256,
    )
    completed_keys = {
        (row["question_uid"], row["candidate_index"])
        for row in existing
    }
    pending = [
        row for row in selected
        if (
            row["question_uid"],
            row["candidate_index"],
        ) not in completed_keys
    ]

    if completed_keys - expected_keys:
        raise RuntimeError(
            f"缓存含额外候选：{output_path}"
        )

    print()
    print("=" * 76)
    print(spec["name"])
    print("模型：", model_path)
    print("候选：", len(selected))
    print("已完成：", len(existing))
    print("待评分：", len(pending))

    synthetic = None
    peak_gpu_gb = 0.0
    started = time.time()

    if pending:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        option_tokens = [
            tokenizer.encode(
                option,
                add_special_tokens=False,
            )
            for option in config["confidence_options"]
        ]
        if any(len(ids) != 1 for ids in option_tokens):
            raise RuntimeError(
                f"0/1 不是单 token：{option_tokens}"
            )
        option_ids = [ids[0] for ids in option_tokens]
        print("0/1 token ids：", option_ids)

        torch.cuda.reset_peak_memory_stats()
        print("加载模型……", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to("cuda")
        model.eval()

        synthetic = synthetic_probe(
            model,
            tokenizer,
            config,
            option_ids,
        )
        print(
            "synthetic correct p(True)=",
            f"{synthetic['synthetic_correct']['p_true_binary_normalized']:.6f}",
        )
        print(
            "synthetic wrong   p(True)=",
            f"{synthetic['synthetic_wrong']['p_true_binary_normalized']:.6f}",
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with output_path.open(
            "a",
            encoding="utf-8",
        ) as output_file:
            batch_size = int(spec["batch_size"])
            for start in range(0, len(pending), batch_size):
                batch_rows = pending[start:start + batch_size]
                prompts = []
                response_meta = []
                for row in batch_rows:
                    prompt, response, method = (
                        render_confidence_prompt(
                            tokenizer,
                            row,
                            config["input_instruction"],
                            config["confidence_suffix"],
                        )
                    )
                    prompts.append(prompt)
                    response_meta.append((response, method))

                scores = score_prompts(
                    model,
                    tokenizer,
                    prompts,
                    option_ids,
                    config["max_input_tokens"],
                )

                for row, meta, score in zip(
                    batch_rows,
                    response_meta,
                    scores,
                ):
                    response, method = meta
                    result = {
                        "experiment": spec["name"],
                        "model_name": spec["model_name"],
                        "question_uid": row["question_uid"],
                        "problem_id": row["problem_id"],
                        "candidate_index": row["candidate_index"],
                        "answer_truncation_method": method,
                        "response_characters_scored": len(response),
                        **score,
                        "config_sha256": config_sha256,
                    }
                    output_file.write(
                        json.dumps(
                            result,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                output_file.flush()
                os.fsync(output_file.fileno())
                done = len(existing) + start + len(batch_rows)
                print(
                    f"{spec['name']}: {done}/{len(selected)}",
                    flush=True,
                )

        peak_gpu_gb = float(
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    final_rows = read_existing(
        output_path,
        config_sha256,
    )
    final_keys = {
        (row["question_uid"], row["candidate_index"])
        for row in final_rows
    }
    if final_keys != expected_keys:
        raise RuntimeError(
            f"最终候选不完整：{len(final_keys)} "
            f"!= {len(expected_keys)}"
        )
    if not all(
        np.isfinite(row["p_true_binary_normalized"])
        for row in final_rows
    ):
        raise RuntimeError("出现非有限 P(True)")

    summary = {
        "name": spec["name"],
        "model_name": spec["model_name"],
        "dataset": spec["dataset"],
        "dataset_sha256": sha256_file(dataset_path),
        "questions": spec["questions"],
        "candidates": len(final_rows),
        "statistics": summarize_rows(final_rows),
        "synthetic_probe": synthetic,
        "peak_gpu_gb": round(peak_gpu_gb, 3),
        "elapsed_seconds": round(time.time() - started, 3),
        "output_file": str(output_path.relative_to(ROOT)),
        "output_sha256": sha256_file(output_path),
    }
    print(json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ))
    return summary


def main():
    started = time.time()
    config = json.loads(
        CONFIG_PATH.read_text(encoding="utf-8")
    )
    config_sha256 = stable_hash(config)

    print("===== CISC P(True) smoke =====")
    print("官方口径：二元 0/1 completion likelihood")
    print("配置哈希：", config_sha256)
    print("标签用于评分：", False)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    results = []
    for spec in config["experiments"]:
        results.append(run_experiment(
            config,
            spec,
            config_sha256,
        ))

    manifest = {
        "version": config["version"],
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "official_reference": config[
            "official_reference"
        ],
        "confidence_method": "CISC P(True)",
        "confidence_options": ["0", "1"],
        "option_normalization": (
            "softmax over next-token logits for 0 and 1"
        ),
        "labels_used_for_scoring": False,
        "config_file": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": config_sha256,
        "results": results,
        "elapsed_seconds": round(
            time.time() - started,
            3,
        ),
    }
    atomic_json(MANIFEST_PATH, manifest)
    print()
    print("===== Smoke 完成 =====")
    print("清单：", MANIFEST_PATH)


if __name__ == "__main__":
    main()

from pathlib import Path
from collections import defaultdict
import hashlib
import json
import time

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/processed/prototype_v2"
CACHE_ROOT = ROOT / "data/cache/generator_hidden_smoke_v1"

DATASETS = {
    "gsm_train_smoke": {
        "path": DATA_ROOT / "gsm_train.jsonl",
        "questions": 128,
        "model": "Qwen2-1.5B",
    },
    "gsm_pilot_smoke": {
        "path": DATA_ROOT / "gsm_pilot_validation.jsonl",
        "questions": 64,
        "model": "Qwen2-1.5B",
    },
    "gsm_id_smoke": {
        "path": DATA_ROOT / "gsm_id_test_mixed.jsonl",
        "questions": 64,
        "model": "Qwen2-1.5B",
    },
    "svamp_ood_smoke": {
        "path": DATA_ROOT / "svamp_ood_mixed.jsonl",
        "questions": 64,
        "model": "Qwen2-1.5B",
    },
    "math_train_smoke": {
        "path": DATA_ROOT / "math_train.jsonl",
        "questions": 128,
        "model": "Qwen2-7B",
    },
    "math_pilot_smoke": {
        "path": DATA_ROOT / "math_pilot_validation.jsonl",
        "questions": 64,
        "model": "Qwen2-7B",
    },
    "math_id_smoke": {
        "path": DATA_ROOT / "math_id_test_mixed.jsonl",
        "questions": 64,
        "model": "Qwen2-7B",
    },
}

PROMPTS = {
    "problem_solution": (
        "Problem:\n{problem}\n\nSolution:\n"
    ),
    "question_answer": (
        "Question: {problem}\nAnswer:\n"
    ),
}

MODEL_BATCH = {
    "Qwen2-1.5B": 16,
    "Qwen2-7B": 4,
}

MODEL_MAX_LENGTH = {
    "Qwen2-1.5B": 1024,
    "Qwen2-7B": 2048,
}

SELECTION_SEED = 20260901


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stable_key(uid, dataset):
    text = (
        f"{SELECTION_SEED}|{dataset}|{uid}"
    )
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def select_questions(rows, count, dataset):
    groups = defaultdict(list)

    for row in rows:
        groups[str(row["question_uid"])].append(row)

    selected_uids = sorted(
        groups,
        key=lambda uid: stable_key(uid, dataset),
    )[:count]

    selected = []
    for uid in selected_uids:
        selected.extend(groups[uid])

    return selected, selected_uids


def encode_record(tokenizer, row, template):
    prefix = template.format(
        problem=str(row["problem"])
    )
    response = str(row["solution_text"])

    prefix_ids = tokenizer.encode(
        prefix,
        add_special_tokens=True,
    )
    response_ids = tokenizer.encode(
        response,
        add_special_tokens=False,
    )

    input_ids = prefix_ids + response_ids

    return {
        "input_ids": input_ids,
        "prefix_length": len(prefix_ids),
        "response_length": len(response_ids),
    }


def pad_batch(encoded, pad_token_id):
    maximum = max(
        len(item["input_ids"])
        for item in encoded
    )

    input_ids = []
    attention_mask = []
    lengths = []
    prefix_lengths = []
    response_lengths = []

    for item in encoded:
        ids = item["input_ids"]
        padding = maximum - len(ids)

        input_ids.append(
            ids + [pad_token_id] * padding
        )
        attention_mask.append(
            [1] * len(ids) + [0] * padding
        )
        lengths.append(len(ids))
        prefix_lengths.append(
            item["prefix_length"]
        )
        response_lengths.append(
            item["response_length"]
        )

    return {
        "input_ids": torch.tensor(
            input_ids,
            dtype=torch.long,
            device="cuda",
        ),
        "attention_mask": torch.tensor(
            attention_mask,
            dtype=torch.long,
            device="cuda",
        ),
        "lengths": lengths,
        "prefix_lengths": prefix_lengths,
        "response_lengths": response_lengths,
    }


@torch.inference_mode()
def token_nll_features(
    model,
    final_hidden,
    input_ids,
    lengths,
    prefix_lengths,
):
    results = []

    for row_index in range(len(lengths)):
        total_length = lengths[row_index]
        prefix_length = prefix_lengths[row_index]

        if total_length <= prefix_length:
            raise RuntimeError("回答 token 为空")

        predictor_positions = torch.arange(
            prefix_length - 1,
            total_length - 1,
            device="cuda",
        )
        target_positions = torch.arange(
            prefix_length,
            total_length,
            device="cuda",
        )

        hidden = final_hidden[
            row_index,
            predictor_positions,
        ]
        targets = input_ids[
            row_index,
            target_positions,
        ]

        nll_parts = []

        for start in range(0, len(targets), 128):
            end = min(start + 128, len(targets))

            logits = model.lm_head(
                hidden[start:end]
            ).float()

            target_logits = logits.gather(
                1,
                targets[start:end].unsqueeze(1),
            ).squeeze(1)

            nll = (
                torch.logsumexp(logits, dim=1)
                - target_logits
            )
            nll_parts.append(
                nll.detach().cpu()
            )

            del logits, target_logits, nll

        values = torch.cat(nll_parts).numpy()

        results.append([
            float(np.mean(values)),
            float(np.quantile(values, 0.90)),
            float(np.max(values)),
            float(np.mean(values[-min(8, len(values)):])),
            float(np.std(values)),
        ])

    return np.asarray(results, dtype=np.float32)


@torch.inference_mode()
def extract_dataset(
    model,
    tokenizer,
    model_name,
    dataset_name,
    rows,
    prompt_name,
    template,
):
    batch_size = MODEL_BATCH[model_name]
    max_length = MODEL_MAX_LENGTH[model_name]

    output_dir = (
        CACHE_ROOT / model_name / prompt_name
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    layers = list(model.model.layers)
    layer_count = len(layers)

    context = {
        "terminal": None,
        "captured": None,
    }
    handles = []

    def make_hook(layer_index):
        def hook(module, inputs, output):
            hidden = (
                output[0]
                if isinstance(output, tuple)
                else output
            )

            batch_index = torch.arange(
                hidden.shape[0],
                device=hidden.device,
            )
            terminal = context["terminal"]

            context["captured"][layer_index] = (
                hidden[
                    batch_index,
                    terminal,
                ]
                .detach()
                .to(torch.float16)
                .cpu()
            )

        return hook

    for layer_index, layer in enumerate(layers):
        handles.append(
            layer.register_forward_hook(
                make_hook(layer_index)
            )
        )

    hidden_batches = []
    nll_batches = []
    metadata = []

    started = time.time()
    processed = 0

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]

        encoded = [
            encode_record(
                tokenizer,
                row,
                template,
            )
            for row in batch_rows
        ]

        too_long = [
            len(item["input_ids"])
            for item in encoded
            if len(item["input_ids"]) > max_length
        ]
        if too_long:
            raise RuntimeError(
                f"{dataset_name} 存在超过长度限制的样本："
                f"max={max(too_long)}, limit={max_length}"
            )

        batch = pad_batch(
            encoded,
            tokenizer.pad_token_id,
        )

        context["terminal"] = torch.tensor(
            [
                length - 1
                for length in batch["lengths"]
            ],
            dtype=torch.long,
            device="cuda",
        )
        context["captured"] = [
            None
            for _ in range(layer_count)
        ]

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            output = model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch[
                    "attention_mask"
                ],
                use_cache=False,
                return_dict=True,
            )

        block_states = torch.stack(
            context["captured"],
            dim=0,
        )

        batch_index = torch.arange(
            len(batch_rows),
            device="cuda",
        )
        final_normalized = output.last_hidden_state[
            batch_index,
            context["terminal"],
        ].detach().to(torch.float16).cpu()

        # 28 block 输出 + 最终归一化输出。
        terminal_states = torch.cat(
            [
                block_states,
                final_normalized.unsqueeze(0),
            ],
            dim=0,
        ).numpy()

        nll = token_nll_features(
            model,
            output.last_hidden_state,
            batch["input_ids"],
            batch["lengths"],
            batch["prefix_lengths"],
        )

        hidden_batches.append(terminal_states)
        nll_batches.append(nll)

        for row, encoded_item in zip(
            batch_rows,
            encoded,
        ):
            metadata.append({
                "question_uid": str(
                    row["question_uid"]
                ),
                "candidate_index": int(
                    row.get("candidate_index", -1)
                ),
                "problem_id": row.get("problem_id"),
                "label": int(row["label"]),
                "input_tokens": len(
                    encoded_item["input_ids"]
                ),
                "response_tokens": int(
                    encoded_item["response_length"]
                ),
            })

        processed += len(batch_rows)

        if (
            processed % (batch_size * 10) == 0
            or processed == len(rows)
        ):
            print(
                f"{model_name}/{prompt_name}/"
                f"{dataset_name}: "
                f"{processed}/{len(rows)}",
                flush=True,
            )

        context["terminal"] = None
        context["captured"] = None
        del output, block_states, final_normalized
        del terminal_states, nll, batch_index
        del batch, encoded
        torch.cuda.empty_cache()

    for handle in handles:
        handle.remove()

    hidden_array = np.concatenate(
        hidden_batches,
        axis=1,
    )
    nll_array = np.concatenate(
        nll_batches,
        axis=0,
    )
    labels = np.asarray(
        [item["label"] for item in metadata],
        dtype=np.int8,
    )

    hidden_path = (
        output_dir
        / f"{dataset_name}.terminal_layers_f16.npy"
    )
    nll_path = (
        output_dir
        / f"{dataset_name}.token_nll_f32.npy"
    )
    label_path = (
        output_dir
        / f"{dataset_name}.labels_i8.npy"
    )
    metadata_path = (
        output_dir
        / f"{dataset_name}.metadata.jsonl"
    )

    np.save(hidden_path, hidden_array)
    np.save(nll_path, nll_array)
    np.save(label_path, labels)

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for item in metadata:
            file.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                ) + "\n"
            )

    positive = labels == 1
    negative = labels == 0

    return {
        "dataset": dataset_name,
        "model": model_name,
        "prompt": prompt_name,
        "questions": len({
            item["question_uid"]
            for item in metadata
        }),
        "candidates": len(metadata),
        "hidden_shape": list(hidden_array.shape),
        "hidden_gb": round(
            hidden_array.nbytes / (1024 ** 3),
            4,
        ),
        "nll_columns": [
            "mean_nll",
            "p90_nll",
            "max_nll",
            "last8_mean_nll",
            "nll_std",
        ],
        "mean_nll_positive": float(
            np.mean(nll_array[positive, 0])
        ),
        "mean_nll_negative": float(
            np.mean(nll_array[negative, 0])
        ),
        "elapsed_seconds": round(
            time.time() - started,
            3,
        ),
        "files": {
            "hidden": str(hidden_path),
            "nll": str(nll_path),
            "labels": str(label_path),
            "metadata": str(metadata_path),
        },
    }


def main():
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    selected = {}
    for name, spec in DATASETS.items():
        rows = read_jsonl(spec["path"])
        subset, uids = select_questions(
            rows,
            spec["questions"],
            name,
        )
        selected[name] = {
            "rows": subset,
            "uids": uids,
            "model": spec["model"],
        }
        print(
            f"{name}: "
            f"问题={len(uids)}, "
            f"候选={len(subset)}"
        )

    manifest = {
        "version": "generator_hidden_smoke_v1",
        "selection_seed": SELECTION_SEED,
        "prompt_templates": PROMPTS,
        "results": {},
    }

    for model_name in [
        "Qwen2-1.5B",
        "Qwen2-7B",
    ]:
        model_path = (
            ROOT / "models/generator" / model_name
        )

        print("\n" + "=" * 80)
        print("加载：", model_path)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=True,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to("cuda")

        model.eval()
        model.requires_grad_(False)
        model.config.use_cache = False

        for prompt_name, template in PROMPTS.items():
            for dataset_name, item in selected.items():
                if item["model"] != model_name:
                    continue

                result = extract_dataset(
                    model,
                    tokenizer,
                    model_name,
                    dataset_name,
                    item["rows"],
                    prompt_name,
                    template,
                )

                key = (
                    f"{model_name}/"
                    f"{prompt_name}/"
                    f"{dataset_name}"
                )
                manifest["results"][key] = result

                print(json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                ))

        del model, tokenizer
        torch.cuda.empty_cache()

    manifest_path = (
        ROOT / "data/manifests/"
        "generator_hidden_smoke_v1.json"
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n生成模型隐藏状态冒烟提取完成。")
    print("清单：", manifest_path)
    print(
        "显存峰值 GB：",
        round(
            torch.cuda.max_memory_allocated()
            / (1024 ** 3),
            3,
        ),
    )


if __name__ == "__main__":
    main()

from pathlib import Path
import gc
import json
import sys
import time

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


ROOT = Path("/root/autodl-tmp/rm_traj_project")
sys.path.insert(0, str(ROOT / "scripts"))

from extract_generator_hidden_smoke import (
    encode_record,
    pad_batch,
    token_nll_features,
)


DATA_ROOT = ROOT / "data/processed/prototype_v2"
CACHE_ROOT = (
    ROOT / "data/cache/generator_cluster_features_v1"
)
MANIFEST = (
    ROOT
    / "data/manifests/generator_cluster_features_full_v1.json"
)

PROMPT_TEMPLATE = (
    "Problem:\n{problem}\n\nSolution:\n"
)

MODELS = {
    "Qwen2-1.5B": {
        "path": (
            ROOT / "models/generator/Qwen2-1.5B"
        ),
        "layer_index": 16,
        "layer_name": "block_17",
        "batch_size": 16,
        "max_length": 1024,
        "datasets": {
            "gsm_train": "gsm_train.jsonl",
            "gsm_pilot": (
                "gsm_pilot_validation.jsonl"
            ),
            "gsm_id_test": (
                "gsm_id_test_mixed.jsonl"
            ),
            "svamp_ood": (
                "svamp_ood_mixed.jsonl"
            ),
        },
    },
    "Qwen2-7B": {
        "path": (
            ROOT / "models/generator/Qwen2-7B"
        ),
        "layer_index": 18,
        "layer_name": "block_19",
        "batch_size": 2,
        "max_length": 2048,
        "datasets": {
            "math_train": "math_train.jsonl",
            "math_pilot": (
                "math_pilot_validation.jsonl"
            ),
            "math_id_test": (
                "math_id_test_mixed.jsonl"
            ),
        },
    },
}


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def existing_result(
    output_dir,
    prefix,
    candidate_count,
    hidden_size,
):
    paths = {
        "hidden": (
            output_dir
            / f"{prefix}.terminal_hidden_f16.npy"
        ),
        "nll": (
            output_dir
            / f"{prefix}.token_nll_f32.npy"
        ),
        "labels": (
            output_dir
            / f"{prefix}.labels_i8.npy"
        ),
        "metadata": (
            output_dir
            / f"{prefix}.metadata.jsonl"
        ),
    }

    if not all(path.exists() for path in paths.values()):
        return None

    try:
        hidden = np.load(
            paths["hidden"],
            mmap_mode="r",
        )
        nll = np.load(
            paths["nll"],
            mmap_mode="r",
        )
        labels = np.load(
            paths["labels"],
            mmap_mode="r",
        )

        metadata_count = sum(
            1
            for line in paths[
                "metadata"
            ].open("r", encoding="utf-8")
            if line.strip()
        )

        valid = (
            hidden.shape
            == (candidate_count, hidden_size)
            and nll.shape == (candidate_count, 5)
            and labels.shape == (candidate_count,)
            and metadata_count == candidate_count
        )

        if not valid:
            return None

        return {
            "candidates": candidate_count,
            "feature_shape": list(hidden.shape),
            "feature_gb": round(
                hidden.nbytes / (1024 ** 3),
                4,
            ),
            "skipped_existing": True,
            "files": {
                key: str(value)
                for key, value in paths.items()
            },
        }
    except Exception:
        return None


@torch.inference_mode()
def extract_dataset(
    model,
    tokenizer,
    model_name,
    model_spec,
    prefix,
    filename,
):
    rows = read_jsonl(DATA_ROOT / filename)
    candidate_count = len(rows)
    hidden_size = int(model.config.hidden_size)

    output_dir = CACHE_ROOT / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = existing_result(
        output_dir,
        prefix,
        candidate_count,
        hidden_size,
    )
    if existing is not None:
        print(
            f"{model_name}/{prefix}: "
            "已有完整缓存，自动跳过。",
            flush=True,
        )
        return {
            "model": model_name,
            "dataset": prefix,
            "source_file": filename,
            "layer_name": model_spec["layer_name"],
            **existing,
        }

    layer = model.model.layers[
        model_spec["layer_index"]
    ]
    batch_size = model_spec["batch_size"]
    max_length = model_spec["max_length"]

    context = {
        "terminal": None,
        "captured": None,
    }

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
        context["captured"] = (
            hidden[
                batch_index,
                context["terminal"],
            ]
            .detach()
            .to(torch.float16)
            .cpu()
        )

    handle = layer.register_forward_hook(hook)

    hidden_batches = []
    nll_batches = []
    metadata = []

    started = time.time()
    processed = 0
    peak_gpu = 0

    try:
        for start in range(
            0,
            candidate_count,
            batch_size,
        ):
            batch_rows = rows[
                start:start + batch_size
            ]

            encoded = [
                encode_record(
                    tokenizer,
                    row,
                    PROMPT_TEMPLATE,
                )
                for row in batch_rows
            ]

            longest = max(
                len(item["input_ids"])
                for item in encoded
            )
            if longest > max_length:
                raise RuntimeError(
                    f"{prefix}: 输入超过上限，"
                    f"length={longest}, "
                    f"limit={max_length}"
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
            context["captured"] = None

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

            if context["captured"] is None:
                raise RuntimeError(
                    f"{prefix}: Hook 未捕获隐藏状态"
                )

            hidden_batches.append(
                context["captured"].numpy()
            )

            nll = token_nll_features(
                model,
                output.last_hidden_state,
                batch["input_ids"],
                batch["lengths"],
                batch["prefix_lengths"],
            )
            nll_batches.append(nll)

            for row, item in zip(
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
                    "problem_id": row.get(
                        "problem_id"
                    ),
                    "label": int(row["label"]),
                    "input_tokens": len(
                        item["input_ids"]
                    ),
                    "response_tokens": int(
                        item["response_length"]
                    ),
                })

            processed += len(batch_rows)
            peak_gpu = max(
                peak_gpu,
                torch.cuda.max_memory_allocated(),
            )

            if (
                processed
                % max(batch_size * 25, 100)
                == 0
                or processed == candidate_count
            ):
                elapsed = time.time() - started
                print(
                    f"{model_name}/{prefix}: "
                    f"{processed}/{candidate_count}, "
                    f"{processed / elapsed:.1f} 候选/秒",
                    flush=True,
                )

            context["terminal"] = None
            context["captured"] = None

            del output, nll, batch, encoded
            torch.cuda.empty_cache()

    finally:
        handle.remove()

    hidden_array = np.concatenate(
        hidden_batches,
        axis=0,
    ).astype(np.float16, copy=False)
    nll_array = np.concatenate(
        nll_batches,
        axis=0,
    ).astype(np.float32, copy=False)
    labels = np.asarray(
        [item["label"] for item in metadata],
        dtype=np.int8,
    )

    if hidden_array.shape != (
        candidate_count,
        hidden_size,
    ):
        raise RuntimeError(
            f"{prefix}: 隐状态形状异常 "
            f"{hidden_array.shape}"
        )

    hidden_path = (
        output_dir
        / f"{prefix}.terminal_hidden_f16.npy"
    )
    nll_path = (
        output_dir
        / f"{prefix}.token_nll_f32.npy"
    )
    labels_path = (
        output_dir
        / f"{prefix}.labels_i8.npy"
    )
    metadata_path = (
        output_dir
        / f"{prefix}.metadata.jsonl"
    )

    np.save(hidden_path, hidden_array)
    np.save(nll_path, nll_array)
    np.save(labels_path, labels)

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for item in metadata:
            file.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    positive = labels == 1
    negative = labels == 0

    result = {
        "model": model_name,
        "dataset": prefix,
        "source_file": filename,
        "layer_name": model_spec["layer_name"],
        "prompt_template": PROMPT_TEMPLATE,
        "candidates": candidate_count,
        "feature_shape": list(hidden_array.shape),
        "feature_gb": round(
            hidden_array.nbytes / (1024 ** 3),
            4,
        ),
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
        "candidates_per_second": round(
            candidate_count
            / (time.time() - started),
            3,
        ),
        "peak_gpu_gb": round(
            peak_gpu / (1024 ** 3),
            3,
        ),
        "skipped_existing": False,
        "files": {
            "hidden": str(hidden_path),
            "nll": str(nll_path),
            "labels": str(labels_path),
            "metadata": str(metadata_path),
        },
    }

    print(json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    ))

    return result


def main():
    started = time.time()
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": (
            "generator_cluster_features_full_v1"
        ),
        "purpose": (
            "single selected generator layer plus "
            "token NLL for answer-cluster evidence"
        ),
        "prompt_policy": (
            "single canonical problem_solution prompt; "
            "smoke experiment showed near-identical "
            "behavior across two prompt variants"
        ),
        "models": {},
    }

    for model_name, model_spec in MODELS.items():
        print()
        print("=" * 76)
        print("加载生成模型：", model_name)

        tokenizer = AutoTokenizer.from_pretrained(
            model_spec["path"],
            trust_remote_code=True,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = (
                tokenizer.eos_token_id
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_spec["path"],
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to("cuda")

        model.eval()
        model.requires_grad_(False)

        torch.cuda.reset_peak_memory_stats()

        model_results = {}

        for prefix, filename in model_spec[
            "datasets"
        ].items():
            print()
            print("-" * 72)
            print(prefix)

            model_results[prefix] = extract_dataset(
                model,
                tokenizer,
                model_name,
                model_spec,
                prefix,
                filename,
            )

        manifest["models"][model_name] = {
            "layer_index_zero_based": (
                model_spec["layer_index"]
            ),
            "layer_name": model_spec[
                "layer_name"
            ],
            "hidden_size": int(
                model.config.hidden_size
            ),
            "datasets": model_results,
        }

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    manifest["elapsed_seconds"] = round(
        time.time() - started,
        3,
    )
    manifest["remaining_disk_gb"] = round(
        __import__("shutil").disk_usage(
            ROOT
        ).free
        / (1024 ** 3),
        3,
    )

    MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    MANIFEST.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print("全量生成模型簇特征提取完成。")
    print("清单：", MANIFEST)
    print("耗时秒：", manifest["elapsed_seconds"])
    print(
        "剩余磁盘GB：",
        manifest["remaining_disk_gb"],
    )


if __name__ == "__main__":
    main()

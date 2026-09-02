from pathlib import Path
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
import os
import time

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


ROOT = Path("/root/autodl-tmp/rm_traj_project")
CONFIG_PATH = (
    ROOT / "configs/multi_reward_full_scoring_v2.json"
)
DEVICE = torch.device("cuda")


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


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


def statistics(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )
    return {
        "count": len(values),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def count_jsonl(path):
    with path.open("rb") as file:
        return sum(
            1
            for line in file
            if line.strip()
        )


def preflight(config, model_key, split_names):
    if model_key not in config["models"]:
        raise RuntimeError(
            f"未知模型：{model_key}"
        )

    model_spec = config["models"][model_key]
    model_path = ROOT / model_spec["path"]

    if not (model_path / "config.json").exists():
        raise RuntimeError(
            f"模型不存在：{model_path}"
        )

    available = {
        item["name"]: item
        for item in config["splits"]
    }

    print("===== 多奖励模型评分预检 =====")
    print("模型：", model_spec["name"])
    print("路径：", model_path)
    print(
        "初始 batch：",
        model_spec["initial_batch_size"],
    )
    print("标签用于评分：False")

    total = 0

    for name in split_names:
        if name not in available:
            raise RuntimeError(
                f"未知数据集：{name}"
            )

        path = ROOT / available[name]["file"]
        if not path.exists():
            raise RuntimeError(
                f"数据不存在：{path}"
            )

        count = count_jsonl(path)
        total += count
        print(f"{name}: {count} 候选")

    print("合计：", total)
    return total


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if getattr(
        tokenizer,
        "chat_template",
        None,
    ) is None:
        template_path = (
            model_path / "chat_template.jinja"
        )
        if template_path.exists():
            tokenizer.chat_template = (
                template_path.read_text(
                    encoding="utf-8"
                )
            )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(model_path, tokenizer):
    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": 0},
        )
    )
    model.eval()
    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )
    model.config.use_cache = False
    return model


def score_batch(
    model,
    tokenizer,
    rows,
    indices,
):
    texts = []

    for index in indices:
        row = rows[int(index)]
        conversation = [
            {
                "role": "user",
                "content": str(row["problem"]),
            },
            {
                "role": "assistant",
                "content": str(
                    row["solution_text"]
                ),
            },
        ]
        texts.append(
            tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
        )

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        output = model(**encoded)
        logits = output.logits.float()

    if logits.ndim == 2:
        if logits.shape[1] != 1:
            raise RuntimeError(
                "奖励模型输出不是单标量："
                f"{tuple(logits.shape)}"
            )
        logits = logits[:, 0]

    logits = (
        logits.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if not np.all(np.isfinite(logits)):
        raise RuntimeError(
            "奖励模型产生非有限分数"
        )

    return logits


def score_split(
    model,
    tokenizer,
    model_output_dir,
    split_spec,
    initial_batch_size,
):
    name = split_spec["name"]
    data_path = ROOT / split_spec["file"]

    final_path = (
        model_output_dir
        / f"{name}.scores_f32.npy"
    )
    partial_path = (
        model_output_dir
        / f"{name}.scores_f32.partial.npy"
    )

    rows = read_jsonl(data_path)
    candidate_count = len(rows)

    print()
    print("=" * 76)
    print(name)
    print("候选：", candidate_count)

    if final_path.exists():
        final = np.load(
            final_path,
            mmap_mode="r",
        )
        if (
            final.shape != (candidate_count,)
            or not np.all(np.isfinite(final))
        ):
            raise RuntimeError(
                f"{name}: 已有最终缓存无效"
            )

        print("已完成，直接复用。")
        return {
            "candidates": candidate_count,
            "source_file": str(
                data_path.relative_to(ROOT)
            ),
            "source_sha256": sha256_file(
                data_path
            ),
            "score_file": str(
                final_path.relative_to(ROOT)
            ),
            "score_sha256": sha256_file(
                final_path
            ),
            "statistics": statistics(final),
            "reused": True,
            "elapsed_seconds": 0.0,
        }

    if partial_path.exists():
        scores = np.lib.format.open_memmap(
            partial_path,
            mode="r+",
        )
        if scores.shape != (candidate_count,):
            raise RuntimeError(
                f"{name}: 断点缓存形状错误 "
                f"{scores.shape}"
            )
    else:
        scores = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float32,
            shape=(candidate_count,),
        )
        scores[:] = np.nan
        scores.flush()

    pending = np.flatnonzero(
        ~np.isfinite(scores)
    ).tolist()

    pending.sort(
        key=lambda index: (
            len(str(rows[index]["problem"]))
            + len(str(
                rows[index]["solution_text"]
            ))
        )
    )

    already_done = (
        candidate_count - len(pending)
    )
    print("已完成断点：", already_done)
    print("待评分：", len(pending))

    position = 0
    batch_size = initial_batch_size
    started = time.time()
    last_report = already_done

    while position < len(pending):
        current = pending[
            position:position + batch_size
        ]

        try:
            logits = score_batch(
                model,
                tokenizer,
                rows,
                current,
            )

            scores[
                np.asarray(current, dtype=np.int64)
            ] = logits
            scores.flush()

            position += len(current)
            completed = (
                already_done + position
            )

            if (
                completed - last_report >= 128
                or completed == candidate_count
            ):
                run_speed = position / max(
                    time.time() - started,
                    1e-9,
                )
                print(
                    f"{name}: "
                    f"{completed}/{candidate_count}, "
                    f"{run_speed:.1f} 候选/秒, "
                    f"batch={batch_size}",
                    flush=True,
                )
                last_report = completed

        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()

            if batch_size == 1:
                raise

            batch_size = max(
                1,
                batch_size // 2,
            )
            print(
                "显存不足，batch_size 降为：",
                batch_size,
                flush=True,
            )

    if not np.all(np.isfinite(scores)):
        raise RuntimeError(
            f"{name}: 完成后仍存在缺失分数"
        )

    elapsed = time.time() - started

    scores.flush()
    del scores
    gc.collect()

    partial_path.replace(final_path)

    final = np.load(
        final_path,
        mmap_mode="r",
    )

    return {
        "candidates": candidate_count,
        "source_file": str(
            data_path.relative_to(ROOT)
        ),
        "source_sha256": sha256_file(
            data_path
        ),
        "score_file": str(
            final_path.relative_to(ROOT)
        ),
        "score_sha256": sha256_file(
            final_path
        ),
        "statistics": statistics(final),
        "reused": False,
        "elapsed_seconds": round(
            elapsed,
            3,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
    )
    args = parser.parse_args()

    config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    available_splits = {
        item["name"]: item
        for item in config["splits"]
    }

    split_names = (
        args.splits
        if args.splits
        else list(available_splits)
    )

    preflight(
        config,
        args.model,
        split_names,
    )

    if args.preflight:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    model_spec = config["models"][args.model]
    model_path = ROOT / model_spec["path"]

    output_root = (
        ROOT
        / config["output_root"]
        / model_spec["name"]
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        ROOT
        / config["manifest_root"]
        / (
            "multi_reward_scores_"
            f"{args.model}_v1.json"
        )
    )

    started = time.time()

    print()
    print("===== 多奖励模型全量评分 =====")
    print("模型：", model_spec["name"])
    print("标签用于评分：False")

    tokenizer = load_tokenizer(model_path)

    print("加载奖励模型……")
    model = load_model(
        model_path,
        tokenizer,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    manifest = {
        "version": (
            "multi_reward_full_scoring_v1"
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_key": args.model,
        "model_name": model_spec["name"],
        "model_path": str(
            model_path.relative_to(ROOT)
        ),
        "model_config_sha256": sha256_file(
            model_path / "config.json"
        ),
        "scoring": config["scoring"],
        "labels_used_for_scoring": False,
        "splits": {},
    }

    for split_name in split_names:
        result = score_split(
            model,
            tokenizer,
            output_root,
            available_splits[split_name],
            int(
                model_spec[
                    "initial_batch_size"
                ]
            ),
        )
        manifest["splits"][
            split_name
        ] = result

        manifest["peak_gpu_gb"] = round(
            torch.cuda.max_memory_allocated()
            / 1024 ** 3,
            3,
        )
        manifest["elapsed_seconds"] = round(
            time.time() - started,
            3,
        )
        atomic_json(
            manifest_path,
            manifest,
        )

    manifest["completed_splits"] = (
        split_names
    )
    manifest["peak_gpu_gb"] = round(
        torch.cuda.max_memory_allocated()
        / 1024 ** 3,
        3,
    )
    manifest["elapsed_seconds"] = round(
        time.time() - started,
        3,
    )

    atomic_json(
        manifest_path,
        manifest,
    )

    print()
    print("===== 评分完成 =====")
    print(json.dumps(
        {
            "model": manifest["model_name"],
            "completed_splits": (
                manifest["completed_splits"]
            ),
            "peak_gpu_gb": (
                manifest["peak_gpu_gb"]
            ),
            "elapsed_seconds": (
                manifest["elapsed_seconds"]
            ),
        },
        ensure_ascii=False,
        indent=2,
    ))
    print("清单：", manifest_path)

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

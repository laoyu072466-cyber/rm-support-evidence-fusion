from pathlib import Path
from collections import defaultdict
import gc
import json
import sys
import time

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bootstrap_answer_cluster_final as bootstrap
import eval_answer_cluster_generator_full as evaluation
import explore_answer_cluster_evidence_smoke as cluster


DEVICE = "cuda"
BATCH_SIZE = 8
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20260901
TIE_EPSILON = 1e-6

DATA_FILES = {
    "GSM8K_ID": (
        ROOT
        / "data/processed/prototype_v2/"
        "gsm_id_test_mixed.jsonl"
    ),
    "MATH_ID": (
        ROOT
        / "data/processed/prototype_v2/"
        "math_id_test_mixed.jsonl"
    ),
    "SVAMP_OOD": (
        ROOT
        / "data/processed/prototype_v2/"
        "svamp_ood_mixed.jsonl"
    ),
}

JUDGES = {
    "Qwen3_4B_V2": (
        ROOT
        / "models/reward/"
        "Skywork-Reward-V2-Qwen3-4B"
    ),
    "Qwen3_8B_V2": (
        ROOT
        / "models/reward/"
        "Skywork-Reward-V2-Qwen3-8B"
    ),
    "Llama_8B_V2": (
        ROOT
        / "models/reward/"
        "Skywork-Reward-V2-Llama-3.1-8B"
    ),
    "Llama_8B_v0p2": (
        ROOT
        / "models/reward/"
        "Skywork-Reward-Llama-3.1-8B-v0.2"
    ),
}

CACHE = (
    ROOT
    / "data/cache/"
    "answer_cluster_holdout_rewards_v1"
)
OUTPUT = (
    ROOT
    / "data/manifests/"
    "answer_cluster_holdout_rewards_v1.json"
)


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def json_safe(value):
    if (
        value is None
        or isinstance(value, (str, int, float, bool))
    ):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def reconstruct_final_selections():
    rows_by_dataset = {
        name: read_jsonl(path)
        for name, path in DATA_FILES.items()
    }

    changed = []
    reproduction = {}
    configurations = {}

    for domain_name, spec in evaluation.DOMAINS.items():
        print()
        print("=" * 76)
        print("重建最终选择：", domain_name)

        train = evaluation.load_dataset(
            f"{domain_name}_TRAIN",
            spec,
            *spec["train"],
        )
        pilot = evaluation.load_dataset(
            f"{domain_name}_PILOT",
            spec,
            *spec["pilot"],
        )

        selected, models, _ = (
            cluster.choose_configuration(
                train,
                pilot,
                bootstrap.FEATURE_INDICES,
            )
        )

        configurations[domain_name] = {
            key: json_safe(selected[key])
            for key in [
                "regularization",
                "beta",
                "threshold",
                "top1",
                "pair_macro_strict",
                "damage_rate",
            ]
        }

        print("配置：", configurations[domain_name])

        for dataset_name, split_spec in (
            spec["tests"].items()
        ):
            if dataset_name not in DATA_FILES:
                continue

            dataset = evaluation.load_dataset(
                dataset_name,
                spec,
                *split_spec,
            )
            method_scores = evaluation.predict_learned(
                dataset,
                bootstrap.FEATURE_INDICES,
                models,
                selected["beta"],
                selected["threshold"],
            )

            rows = rows_by_dataset[dataset_name]
            rm_scores = np.asarray(
                dataset["rm_scores"],
                dtype=np.float32,
            )
            method_scores = np.asarray(
                method_scores,
                dtype=np.float32,
            )

            if not (
                len(rows)
                == len(rm_scores)
                == len(method_scores)
            ):
                raise RuntimeError(
                    f"{dataset_name}: 候选数量不一致"
                )

            groups = defaultdict(list)
            for index, row in enumerate(rows):
                groups[str(row["question_uid"])].append(
                    index
                )

            raw_correct = 0
            new_correct = 0
            switch_count = 0
            correction_count = 0
            damage_count = 0

            for uid, indices in groups.items():
                indices_array = np.asarray(
                    indices,
                    dtype=np.int64,
                )
                raw_index = int(
                    indices_array[
                        np.argmax(rm_scores[indices_array])
                    ]
                )
                new_index = int(
                    indices_array[
                        np.argmax(
                            method_scores[indices_array]
                        )
                    ]
                )

                raw_label = int(
                    rows[raw_index]["label"]
                )
                new_label = int(
                    rows[new_index]["label"]
                )

                raw_correct += raw_label
                new_correct += new_label

                if raw_index == new_index:
                    continue

                switch_count += 1
                correction_count += int(
                    raw_label == 0 and new_label == 1
                )
                damage_count += int(
                    raw_label == 1 and new_label == 0
                )

                changed.append({
                    "dataset": dataset_name,
                    "question_uid": uid,
                    "raw_index": raw_index,
                    "new_index": new_index,
                    "raw_label": raw_label,
                    "new_label": new_label,
                })

            question_count = len(groups)
            reproduction[dataset_name] = {
                "questions": question_count,
                "raw_top1": (
                    raw_correct / question_count
                ),
                "new_top1": (
                    new_correct / question_count
                ),
                "top1_delta": (
                    (new_correct - raw_correct)
                    / question_count
                ),
                "switched_questions": switch_count,
                "switch_rate": (
                    switch_count / question_count
                ),
                "corrections": correction_count,
                "damages": damage_count,
            }

            print(
                f"{dataset_name}: "
                f"Top1={raw_correct / question_count:.6f}"
                f"->{new_correct / question_count:.6f}, "
                f"切换={switch_count}, "
                f"修复={correction_count}, "
                f"破坏={damage_count}"
            )

    return (
        rows_by_dataset,
        changed,
        reproduction,
        configurations,
    )


def cache_write(path, model_path, scores):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "model": str(model_path),
                "scores": scores,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def score_with_judge(
    judge_name,
    model_path,
    examples,
):
    cache_path = CACHE / f"{judge_name}.json"

    scores = {}
    if cache_path.exists():
        cached = json.loads(
            cache_path.read_text(encoding="utf-8")
        )
        if cached.get("model") == str(model_path):
            scores = {
                key: float(value)
                for key, value
                in cached.get("scores", {}).items()
            }

    pending = [
        item for item in examples
        if item["key"] not in scores
    ]

    if not pending:
        print(
            f"{judge_name}: 全部命中缓存，"
            f"候选={len(scores)}"
        )
        return scores, 0.0

    print()
    print("=" * 76)
    print("加载外部裁判：", judge_name)
    print("模型：", model_path)
    print("待评分候选：", len(pending))

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if getattr(tokenizer, "chat_template", None) is None:
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

    pending.sort(
        key=lambda item: (
            len(item["problem"])
            + len(item["solution"])
        )
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    position = 0
    batch_size = BATCH_SIZE
    started = time.time()

    while position < len(pending):
        batch = pending[
            position:position + batch_size
        ]

        try:
            texts = []
            for item in batch:
                conversation = [
                    {
                        "role": "user",
                        "content": item["problem"],
                    },
                    {
                        "role": "assistant",
                        "content": item["solution"],
                    },
                ]
                text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                texts.append(text)

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
                logits.detach().cpu().numpy()
            )

            for item, value in zip(batch, logits):
                scores[item["key"]] = float(value)

            position += len(batch)

            if (
                position % 128 < batch_size
                or position == len(pending)
            ):
                speed = position / (
                    time.time() - started
                )
                print(
                    f"{judge_name}: "
                    f"{position}/{len(pending)}, "
                    f"{speed:.1f} 候选/秒"
                )
                cache_write(
                    cache_path,
                    model_path,
                    scores,
                )

        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(
                "显存不足，batch_size 降为：",
                batch_size,
            )

    cache_write(cache_path, model_path, scores)

    peak_gpu_gb = (
        torch.cuda.max_memory_allocated()
        / 1024 ** 3
    )

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return scores, peak_gpu_gb


def bootstrap_mean(values, rng):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.full(
            BOOTSTRAP_SAMPLES,
            np.nan,
            dtype=np.float64,
        )

    indices = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_SAMPLES, len(values)),
    )
    return values[indices].mean(axis=1)


def ci(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [None, None]
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def analyze_judge(
    judge_name,
    scores,
    changed,
):
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
        + sum(ord(char) for char in judge_name)
    )

    results = {
        "datasets": {},
    }
    macro_preference_boot = []
    macro_alignment_boot = []

    for dataset_name in DATA_FILES:
        records = [
            item for item in changed
            if item["dataset"] == dataset_name
        ]

        margins = np.asarray([
            scores[
                f"{dataset_name}:{item['new_index']}"
            ]
            - scores[
                f"{dataset_name}:{item['raw_index']}"
            ]
            for item in records
        ], dtype=np.float64)

        effects = np.asarray([
            item["new_label"] - item["raw_label"]
            for item in records
        ], dtype=np.int8)

        preference = np.where(
            margins > TIE_EPSILON,
            1.0,
            np.where(
                margins < -TIE_EPSILON,
                0.0,
                0.5,
            ),
        )

        label_changed = effects != 0
        alignment = np.where(
            margins[label_changed]
            * effects[label_changed] > TIE_EPSILON,
            1.0,
            np.where(
                np.abs(margins[label_changed])
                <= TIE_EPSILON,
                0.5,
                0.0,
            ),
        )

        preference_boot = bootstrap_mean(
            preference,
            rng,
        )
        alignment_boot = bootstrap_mean(
            alignment,
            rng,
        )

        macro_preference_boot.append(
            preference_boot
        )
        macro_alignment_boot.append(
            alignment_boot
        )

        corrections = margins[effects == 1]
        damages = margins[effects == -1]
        neutral = margins[effects == 0]

        result = {
            "changed_questions": len(records),
            "preference_rate": float(
                preference.mean()
            ),
            "preference_rate_ci95": ci(
                preference_boot
            ),
            "win_rate": float(
                np.mean(margins > TIE_EPSILON)
            ),
            "tie_rate": float(
                np.mean(
                    np.abs(margins)
                    <= TIE_EPSILON
                )
            ),
            "loss_rate": float(
                np.mean(margins < -TIE_EPSILON)
            ),
            "mean_margin": float(margins.mean()),
            "median_margin": float(
                np.median(margins)
            ),
            "alignment_rate_on_label_changes": (
                float(alignment.mean())
                if len(alignment)
                else None
            ),
            "alignment_ci95": (
                ci(alignment_boot)
                if len(alignment)
                else [None, None]
            ),
            "correction_count": int(
                np.sum(effects == 1)
            ),
            "correction_mean_margin": (
                float(corrections.mean())
                if len(corrections)
                else None
            ),
            "damage_count": int(
                np.sum(effects == -1)
            ),
            "damage_mean_margin": (
                float(damages.mean())
                if len(damages)
                else None
            ),
            "neutral_count": int(
                np.sum(effects == 0)
            ),
            "neutral_mean_margin": (
                float(neutral.mean())
                if len(neutral)
                else None
            ),
        }

        results["datasets"][dataset_name] = result

        print(
            f"{judge_name}/{dataset_name}: "
            f"偏好新选择={result['preference_rate']:.4f} "
            f"CI={result['preference_rate_ci95']}, "
            f"标签对齐={result['alignment_rate_on_label_changes']}"
        )

    macro_preference_boot = np.mean(
        np.stack(macro_preference_boot),
        axis=0,
    )
    macro_alignment_boot = np.nanmean(
        np.stack(macro_alignment_boot),
        axis=0,
    )

    results["macro"] = {
        "preference_rate": float(np.mean([
            item["preference_rate"]
            for item in results["datasets"].values()
        ])),
        "preference_rate_ci95": ci(
            macro_preference_boot
        ),
        "alignment_rate_on_label_changes": (
            float(np.mean([
                item[
                    "alignment_rate_on_label_changes"
                ]
                for item
                in results["datasets"].values()
                if item[
                    "alignment_rate_on_label_changes"
                ] is not None
            ]))
        ),
        "alignment_ci95": ci(
            macro_alignment_boot
        ),
    }

    return results


def consensus_analysis(changed, all_scores):
    judge_names = list(all_scores)
    margins = np.stack([
        np.asarray([
            all_scores[judge][
                f"{item['dataset']}:{item['new_index']}"
            ]
            - all_scores[judge][
                f"{item['dataset']}:{item['raw_index']}"
            ]
            for item in changed
        ])
        for judge in judge_names
    ], axis=1)

    wins = np.sum(
        margins > TIE_EPSILON,
        axis=1,
    )
    losses = np.sum(
        margins < -TIE_EPSILON,
        axis=1,
    )
    effects = np.asarray([
        item["new_label"] - item["raw_label"]
        for item in changed
    ])

    majority = np.where(
        wins > losses,
        1,
        np.where(losses > wins, -1, 0),
    )

    label_changed = effects != 0
    aligned = majority[label_changed] * effects[
        label_changed
    ]

    result = {
        "judges": judge_names,
        "changed_questions": len(changed),
        "majority_prefers_new_rate": float(
            np.mean(majority == 1)
        ),
        "majority_prefers_raw_rate": float(
            np.mean(majority == -1)
        ),
        "majority_tie_rate": float(
            np.mean(majority == 0)
        ),
        "unanimous_new_rate": float(
            np.mean(wins == len(judge_names))
        ),
        "unanimous_raw_rate": float(
            np.mean(losses == len(judge_names))
        ),
        "label_change_alignment_rate": float(
            np.mean(aligned > 0)
            + 0.5 * np.mean(aligned == 0)
        ),
        "correction_majority_new_rate": float(
            np.mean(majority[effects == 1] == 1)
        ),
        "damage_majority_raw_rate": float(
            np.mean(majority[effects == -1] == -1)
        ),
        "datasets": {},
    }

    for dataset_name in DATA_FILES:
        mask = np.asarray([
            item["dataset"] == dataset_name
            for item in changed
        ])
        result["datasets"][dataset_name] = {
            "changed_questions": int(mask.sum()),
            "majority_prefers_new_rate": float(
                np.mean(majority[mask] == 1)
            ),
            "majority_prefers_raw_rate": float(
                np.mean(majority[mask] == -1)
            ),
            "tie_rate": float(
                np.mean(majority[mask] == 0)
            ),
        }

    return result


def main():
    started = time.time()

    (
        rows_by_dataset,
        changed,
        reproduction,
        configurations,
    ) = reconstruct_final_selections()

    print()
    print("总切换问题数：", len(changed))

    examples_by_key = {}
    for item in changed:
        dataset_name = item["dataset"]
        rows = rows_by_dataset[dataset_name]

        for row_index in [
            item["raw_index"],
            item["new_index"],
        ]:
            key = f"{dataset_name}:{row_index}"
            row = rows[row_index]
            examples_by_key[key] = {
                "key": key,
                "problem": str(row["problem"]),
                "solution": str(
                    row["solution_text"]
                ),
            }

    examples = list(examples_by_key.values())
    print("需评分的唯一候选数：", len(examples))

    all_scores = {}
    judge_results = {}
    peak_gpu = {}

    for judge_name, model_path in JUDGES.items():
        scores, peak = score_with_judge(
            judge_name,
            model_path,
            examples,
        )
        all_scores[judge_name] = scores
        peak_gpu[judge_name] = peak

        judge_results[judge_name] = analyze_judge(
            judge_name,
            scores,
            changed,
        )

    consensus = consensus_analysis(
        changed,
        all_scores,
    )

    output = {
        "version": (
            "answer_cluster_holdout_rewards_v1"
        ),
        "evaluation_scope": (
            "fixed final rm_support selections; "
            "changed questions only"
        ),
        "test_labels_used_for_tuning": False,
        "base_selection_reward_model": (
            "Skywork-Reward-V2-Qwen3-1.7B"
        ),
        "holdout_judges": {
            name: str(path)
            for name, path in JUDGES.items()
        },
        "selection_configurations": configurations,
        "selection_reproduction": reproduction,
        "unique_candidates_scored": len(examples),
        "judge_results": judge_results,
        "consensus": consensus,
        "peak_gpu_gb": peak_gpu,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "elapsed_seconds": round(
            time.time() - started,
            3,
        ),
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
            default=json_safe,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print("===== 四裁判共识 =====")
    print(
        json.dumps(
            consensus,
            ensure_ascii=False,
            indent=2,
        )
    )
    print("结果：", OUTPUT)
    print(
        "总耗时秒：",
        output["elapsed_seconds"],
    )


if __name__ == "__main__":
    main()

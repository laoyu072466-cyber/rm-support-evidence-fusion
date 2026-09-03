from pathlib import Path
from collections import defaultdict
import gc
import json
import sys
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_sgldsv_current_rm as base


CACHE_PATH = (
    ROOT / "data/cache/sgldsv_full_v1/"
    "Skywork-Reward-V2-Qwen3-1.7B/block_21"
)
TRAIN_RESULT_DIR = ROOT / "outputs/sgldsv_full"
CHECKPOINT_DIR = ROOT / "outputs/checkpoints"
PREDICTION_DIR = (
    ROOT / "outputs/final_predictions/sgldsv_full"
)
OUTPUT_PATH = (
    ROOT / "outputs/"
    "sgldsv_full_final_three_seeds.json"
)
MANIFEST_PATH = (
    ROOT / "data/manifests/"
    "sgldsv_full_final_three_seeds.json"
)

SEEDS = [42, 123, 456]
EVAL_BATCH_SIZE = 48

FAMILIES = {
    "GSM8K": {
        "validation": "gsm_pilot",
        "tests": {
            "GSM8K_ID": "gsm_id_test",
            "SVAMP_OOD": "svamp_ood",
        },
    },
    "MATH": {
        "validation": "math_pilot",
        "tests": {
            "MATH_ID": "math_id_test",
        },
    },
}


def checkpoint_path(family, seed):
    return (
        CHECKPOINT_DIR
        / f"sgldsv_{family.lower()}_seed{seed}.pt"
    )


def training_result_path(family, seed):
    return (
        TRAIN_RESULT_DIR
        / f"sgldsv_{family.lower()}_seed{seed}.json"
    )


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def get_state_dict(payload):
    if isinstance(payload, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "model",
            "best_state_dict",
        ]:
            if key in payload:
                candidate = payload[key]
                if isinstance(candidate, dict):
                    return candidate

        if payload and all(
            torch.is_tensor(value)
            for value in payload.values()
        ):
            return payload

    raise RuntimeError(
        "无法识别 checkpoint 中的模型参数结构"
    )


def extract_final_score(output, batch_size):
    if torch.is_tensor(output):
        tensor = output
    elif isinstance(output, dict):
        tensor = None

        for key in [
            "score",
            "scores",
            "final_score",
            "logits",
        ]:
            value = output.get(key)

            if torch.is_tensor(value):
                tensor = value
                break

        if tensor is None:
            raise RuntimeError(
                f"无法从字典输出识别分数：{output.keys()}"
            )
    elif isinstance(output, (tuple, list)):
        tensor = None

        for value in output:
            if (
                torch.is_tensor(value)
                and value.numel() == batch_size
            ):
                tensor = value
                break

        if tensor is None:
            raise RuntimeError(
                "无法从 tuple/list 输出识别最终分数"
            )
    else:
        raise RuntimeError(
            f"未知模型输出类型：{type(output)}"
        )

    tensor = tensor.reshape(-1)

    if len(tensor) != batch_size:
        raise RuntimeError(
            f"输出数量异常：{len(tensor)} != {batch_size}"
        )

    return tensor


def offset_range(offsets, index):
    if offsets.ndim == 1:
        return (
            int(offsets[index]),
            int(offsets[index + 1]),
        )

    if offsets.ndim == 2 and offsets.shape[1] == 2:
        return (
            int(offsets[index, 0]),
            int(offsets[index, 1]),
        )

    raise RuntimeError(
        f"未知 offsets 结构：{offsets.shape}"
    )


@torch.inference_mode()
def predict(model, prefix):
    paths = base.cache_paths(prefix)
    features = np.load(
        paths["features"],
        mmap_mode="r",
    )
    offsets = np.load(
        paths["offsets"],
        mmap_mode="r",
    )
    labels = np.load(
        paths["labels"],
        mmap_mode="r",
    )

    candidate_count = len(labels)
    output_scores = np.empty(
        candidate_count,
        dtype=np.float32,
    )

    model.eval()

    for start_index in range(
        0,
        candidate_count,
        EVAL_BATCH_SIZE,
    ):
        end_index = min(
            start_index + EVAL_BATCH_SIZE,
            candidate_count,
        )

        sequences = []

        for candidate_index in range(
            start_index,
            end_index,
        ):
            start, end = offset_range(
                offsets,
                candidate_index,
            )

            if end <= start:
                raise RuntimeError(
                    f"{prefix}:{candidate_index} "
                    "回答 token 为空"
                )

            array = np.asarray(
                features[start:end],
                dtype=np.float32,
            ).copy()

            sequences.append(
                torch.from_numpy(array)
            )

        lengths = torch.tensor(
            [len(sequence) for sequence in sequences],
            device=base.DEVICE,
        )

        padded = pad_sequence(
            sequences,
            batch_first=True,
        ).to(
            device=base.DEVICE,
            dtype=torch.float32,
        )

        positions = torch.arange(
            padded.shape[1],
            device=base.DEVICE,
        )
        mask = positions.unsqueeze(0) < lengths.unsqueeze(1)

        output = model(padded, mask)
        scores = extract_final_score(
            output,
            len(sequences),
        )

        output_scores[start_index:end_index] = (
            scores.float().cpu().numpy()
        )

        if (
            end_index % 1600 == 0
            or end_index == candidate_count
        ):
            print(
                f"{prefix}: "
                f"{end_index}/{candidate_count}"
            )

    return output_scores


def load_model(family, seed):
    path = checkpoint_path(family, seed)

    if not path.exists():
        raise FileNotFoundError(path)

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = get_state_dict(payload)

    model = base.SGLDSVHead().to(base.DEVICE)
    model.load_state_dict(
        state_dict,
        strict=True,
    )
    model.eval()

    return model


def transition_metrics(cache, raw_scores, new_scores):
    labels = to_numpy(cache.labels).astype(np.int8)
    metadata = cache.metadata
    groups = defaultdict(list)

    for index, row in enumerate(metadata):
        groups[str(row["question_uid"])].append(index)

    raw_correct = 0
    raw_wrong = 0
    corrected = 0
    damaged = 0

    for indices in groups.values():
        indices = np.asarray(indices, dtype=np.int64)

        raw_choice = indices[
            np.argmax(raw_scores[indices])
        ]
        new_choice = indices[
            np.argmax(new_scores[indices])
        ]

        raw_ok = int(labels[raw_choice]) == 1
        new_ok = int(labels[new_choice]) == 1

        if raw_ok:
            raw_correct += 1
            if not new_ok:
                damaged += 1
        else:
            raw_wrong += 1
            if new_ok:
                corrected += 1

    return {
        "raw_correct_questions": raw_correct,
        "raw_wrong_questions": raw_wrong,
        "corrected_questions": corrected,
        "damaged_questions": damaged,
        "correction_rate": (
            corrected / raw_wrong
            if raw_wrong else 0.0
        ),
        "damage_rate": (
            damaged / raw_correct
            if raw_correct else 0.0
        ),
    }


def evaluate_prefix(model, prefix):
    cache = base.CachedDataset(prefix)
    raw_scores = to_numpy(
        cache.rm_scores
    ).astype(np.float32)

    raw_metrics = base.ranking_metrics(
        cache,
        raw_scores,
    )
    sg_scores = predict(model, prefix)
    sg_metrics = base.ranking_metrics(
        cache,
        sg_scores,
    )
    transitions = transition_metrics(
        cache,
        raw_scores,
        sg_scores,
    )

    result = {
        "raw_rm": raw_metrics,
        "sgldsv": sg_metrics,
        "delta_top1": (
            sg_metrics["top1"]
            - raw_metrics["top1"]
        ),
        "delta_pair_macro_strict": (
            sg_metrics["pair_macro_strict"]
            - raw_metrics["pair_macro_strict"]
        ),
        **transitions,
    }

    del cache
    gc.collect()

    return result, sg_scores


def reproduce_validation():
    print("===== 先复现 Pilot 指标 =====")

    for family, config in FAMILIES.items():
        seed = 42
        model = load_model(family, seed)

        result, _ = evaluate_prefix(
            model,
            config["validation"],
        )

        saved = json.loads(
            training_result_path(
                family,
                seed,
            ).read_text(encoding="utf-8")
        )
        metric_candidates = []

        def collect_metrics(value, metric_path="root"):
            if isinstance(value, dict):
                if (
                    "top1" in value
                    and "pair_macro_strict" in value
                ):
                    metric_candidates.append(
                        (metric_path, value)
                    )

                for key, child in value.items():
                    collect_metrics(
                        child,
                        f"{metric_path}.{key}",
                    )
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    collect_metrics(
                        child,
                        f"{metric_path}[{index}]",
                    )

        collect_metrics(saved)

        if not metric_candidates:
            raise RuntimeError(
                f"{family} 训练结果中没有找到验证指标"
            )

        actual = result["sgldsv"]

        expected_path, expected = min(
            metric_candidates,
            key=lambda item: (
                abs(
                    float(item[1]["top1"])
                    - actual["top1"]
                )
                + abs(
                    float(
                        item[1][
                            "pair_macro_strict"
                        ]
                    )
                    - actual[
                        "pair_macro_strict"
                    ]
                )
            ),
        )

        top1_gap = abs(
            actual["top1"]
            - float(expected["top1"])
        )
        pair_gap = abs(
            actual["pair_macro_strict"]
            - float(
                expected["pair_macro_strict"]
            )
        )

        print(
            f"{family} seed=42 Pilot："
            f"匹配路径={expected_path}, "
            f"Top1 gap={top1_gap:.10f}, "
            f"Pair gap={pair_gap:.10f}"
        )

        del model
        gc.collect()
        torch.cuda.empty_cache()

        if top1_gap > 1e-7 or pair_gap > 1e-6:
            raise RuntimeError(
                f"{family} Pilot 无法复现，"
                "测试评估已停止"
            )

    print("Pilot 复现通过，开始冻结测试评估。\n")


def summarize(values):
    array = np.asarray(values, dtype=np.float64)

    return {
        "mean": float(array.mean()),
        "std": float(
            array.std(ddof=1)
            if len(array) > 1 else 0.0
        ),
        "values": [float(value) for value in array],
    }


def main():
    start = time.time()
    base.CACHE_PATH = CACHE_PATH
    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.set_float32_matmul_precision("high")

    reproduce_validation()

    seed_results = {
        str(seed): {}
        for seed in SEEDS
    }

    for family, config in FAMILIES.items():
        for seed in SEEDS:
            print("\n" + "=" * 72)
            print(
                f"评估 {family}，seed={seed}"
            )

            model = load_model(family, seed)

            for display, prefix in config["tests"].items():
                result, scores = evaluate_prefix(
                    model,
                    prefix,
                )
                seed_results[str(seed)][display] = result

                prediction_path = (
                    PREDICTION_DIR
                    / f"{display.lower()}_seed{seed}.npy"
                )
                np.save(prediction_path, scores)

                print(
                    f"{display}: "
                    f"Top1 "
                    f"{result['raw_rm']['top1']:.6f}"
                    f" -> "
                    f"{result['sgldsv']['top1']:.6f} "
                    f"({result['delta_top1']:+.6f}), "
                    f"Pair "
                    f"{result['raw_rm']['pair_macro_strict']:.6f}"
                    f" -> "
                    f"{result['sgldsv']['pair_macro_strict']:.6f} "
                    f"({result['delta_pair_macro_strict']:+.6f}), "
                    f"Damage={result['damage_rate']:.6f}"
                )

            del model
            gc.collect()
            torch.cuda.empty_cache()

    dataset_names = [
        "GSM8K_ID",
        "MATH_ID",
        "SVAMP_OOD",
    ]
    aggregate = {}

    for dataset in dataset_names:
        aggregate[dataset] = {}

        for metric in [
            "delta_top1",
            "delta_pair_macro_strict",
            "damage_rate",
            "correction_rate",
        ]:
            aggregate[dataset][metric] = summarize([
                seed_results[str(seed)][dataset][metric]
                for seed in SEEDS
            ])

        aggregate[dataset]["sgldsv_top1"] = summarize([
            seed_results[str(seed)][dataset][
                "sgldsv"
            ]["top1"]
            for seed in SEEDS
        ])
        aggregate[dataset]["sgldsv_pair"] = summarize([
            seed_results[str(seed)][dataset][
                "sgldsv"
            ]["pair_macro_strict"]
            for seed in SEEDS
        ])

    id_macro_top1 = []
    id_macro_pair = []
    all_macro_top1 = []
    all_macro_pair = []

    for seed in SEEDS:
        current = seed_results[str(seed)]

        id_macro_top1.append(np.mean([
            current["GSM8K_ID"]["sgldsv"]["top1"],
            current["MATH_ID"]["sgldsv"]["top1"],
        ]))
        id_macro_pair.append(np.mean([
            current["GSM8K_ID"]["sgldsv"][
                "pair_macro_strict"
            ],
            current["MATH_ID"]["sgldsv"][
                "pair_macro_strict"
            ],
        ]))
        all_macro_top1.append(np.mean([
            current[name]["sgldsv"]["top1"]
            for name in dataset_names
        ]))
        all_macro_pair.append(np.mean([
            current[name]["sgldsv"][
                "pair_macro_strict"
            ]
            for name in dataset_names
        ]))

    aggregate["ID_MACRO"] = {
        "top1": summarize(id_macro_top1),
        "pair_macro_strict": summarize(
            id_macro_pair
        ),
    }
    aggregate["ALL_DATASET_MACRO"] = {
        "top1": summarize(all_macro_top1),
        "pair_macro_strict": summarize(
            all_macro_pair
        ),
    }

    output = {
        "version": "sgldsv_full_final_three_seeds_v1",
        "protocol": {
            "backbone": (
                "Skywork-Reward-V2-Qwen3-1.7B"
            ),
            "block_number": 21,
            "separate_dataset_heads": True,
            "seeds": SEEDS,
            "checkpoint_selection": (
                "pilot_pair_macro_strict"
            ),
            "test_used_for_training_or_selection": False,
            "gsm_head_tests": [
                "GSM8K_ID",
                "SVAMP_OOD",
            ],
            "math_head_tests": ["MATH_ID"],
        },
        "per_seed": seed_results,
        "aggregate": aggregate,
        "elapsed_seconds": round(
            time.time() - start,
            3,
        ),
        "peak_gpu_gb": round(
            torch.cuda.max_memory_allocated()
            / 1024 ** 3,
            3,
        ),
    }

    text = json.dumps(
        output,
        ensure_ascii=False,
        indent=2,
    ) + "\n"

    OUTPUT_PATH.write_text(
        text,
        encoding="utf-8",
    )
    MANIFEST_PATH.write_text(
        text,
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("===== SG-LDSV 三随机种子最终结果 =====")

    for dataset in dataset_names:
        item = aggregate[dataset]
        print(
            f"{dataset}: "
            f"Top1="
            f"{item['sgldsv_top1']['mean']:.6f}"
            f" ± {item['sgldsv_top1']['std']:.6f}, "
            f"ΔTop1="
            f"{item['delta_top1']['mean']:+.6f}"
            f" ± {item['delta_top1']['std']:.6f}, "
            f"Pair="
            f"{item['sgldsv_pair']['mean']:.6f}"
            f" ± {item['sgldsv_pair']['std']:.6f}, "
            f"ΔPair="
            f"{item['delta_pair_macro_strict']['mean']:+.6f}"
            f" ± "
            f"{item['delta_pair_macro_strict']['std']:.6f}"
        )

    print("结果：", OUTPUT_PATH)
    print("清单：", MANIFEST_PATH)
    print("耗时秒：", output["elapsed_seconds"])
    print("显存峰值 GB：", output["peak_gpu_gb"])


if __name__ == "__main__":
    main()

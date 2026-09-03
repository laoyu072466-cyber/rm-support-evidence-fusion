from pathlib import Path
from collections import defaultdict
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/generator_hidden_smoke_v1"
RM_CACHE = (
    ROOT
    / "data/cache/trajectory_features_v1"
    / "Skywork-Reward-V2-Qwen3-1.7B"
    / "layer_28"
)
OUTPUT = (
    ROOT
    / "data/manifests/generator_hidden_probe_smoke_v1.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
SEEDS = [42, 123, 456]
PROMPTS = ["problem_solution", "question_answer"]
PAIR_CAP = 64
EPOCHS = 30
PAIR_BATCH = 256

DOMAINS = {
    "GSM8K": {
        "model": "Qwen2-1.5B",
        "train": "gsm_train_smoke",
        "pilot": "gsm_pilot_smoke",
        "tests": {
            "GSM8K_ID": "gsm_id_smoke",
            "SVAMP_OOD": "svamp_ood_smoke",
        },
        "rm_prefix": {
            "gsm_pilot_smoke": "gsm_pilot",
            "gsm_id_smoke": "gsm_id_test",
            "svamp_ood_smoke": "svamp_ood",
        },
    },
    "MATH": {
        "model": "Qwen2-7B",
        "train": "math_train_smoke",
        "pilot": "math_pilot_smoke",
        "tests": {
            "MATH_ID": "math_id_smoke",
        },
        "rm_prefix": {
            "math_pilot_smoke": "math_pilot",
            "math_id_smoke": "math_id_test",
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


def load_cache(model_name, prompt_name, dataset_name):
    directory = CACHE / model_name / prompt_name

    hidden = np.load(
        directory
        / f"{dataset_name}.terminal_layers_f16.npy",
        mmap_mode="r",
    )
    nll = np.asarray(
        np.load(
            directory
            / f"{dataset_name}.token_nll_f32.npy"
        ),
        dtype=np.float32,
    )
    labels = np.asarray(
        np.load(
            directory
            / f"{dataset_name}.labels_i8.npy"
        ),
        dtype=np.int8,
    )
    metadata = read_jsonl(
        directory / f"{dataset_name}.metadata.jsonl"
    )

    if hidden.shape[1] != len(labels):
        raise RuntimeError("hidden 与 label 数量不一致")
    if len(nll) != len(labels):
        raise RuntimeError("NLL 与 label 数量不一致")
    if len(metadata) != len(labels):
        raise RuntimeError("metadata 与 label 数量不一致")

    return {
        "hidden": hidden,
        "nll": nll,
        "labels": labels,
        "metadata": metadata,
    }


def groups_from_metadata(metadata):
    groups = defaultdict(list)
    for index, item in enumerate(metadata):
        groups[str(item["question_uid"])].append(index)
    return dict(groups)


def ranking_metrics(labels, metadata, scores):
    groups = groups_from_metadata(metadata)
    top1_values = []
    pair_values = []

    for indices in groups.values():
        indices = np.asarray(indices, dtype=np.int64)
        group_labels = labels[indices]
        group_scores = scores[indices]

        selected = int(np.argmax(group_scores))
        top1_values.append(
            float(group_labels[selected] == 1)
        )

        positives = group_scores[group_labels == 1]
        negatives = group_scores[group_labels == 0]

        comparisons = (
            positives[:, None] > negatives[None, :]
        )
        pair_values.append(float(np.mean(comparisons)))

    return {
        "questions": len(groups),
        "top1": float(np.mean(top1_values)),
        "pair_macro_strict": float(np.mean(pair_values)),
    }


def selection_effect(
    labels,
    metadata,
    raw_scores,
    alternative_scores,
):
    groups = groups_from_metadata(metadata)

    raw_correct = 0
    raw_wrong = 0
    damaged = 0
    corrected = 0
    switched = 0

    for indices in groups.values():
        indices = np.asarray(indices, dtype=np.int64)

        raw_choice = indices[
            int(np.argmax(raw_scores[indices]))
        ]
        new_choice = indices[
            int(np.argmax(alternative_scores[indices]))
        ]

        raw_is_correct = labels[raw_choice] == 1
        new_is_correct = labels[new_choice] == 1

        switched += int(raw_choice != new_choice)

        if raw_is_correct:
            raw_correct += 1
            damaged += int(not new_is_correct)
        else:
            raw_wrong += 1
            corrected += int(new_is_correct)

    return {
        "damage_rate": (
            damaged / raw_correct
            if raw_correct else 0.0
        ),
        "correction_rate": (
            corrected / raw_wrong
            if raw_wrong else 0.0
        ),
        "switch_rate": switched / len(groups),
        "net_corrected_questions": corrected - damaged,
    }


def load_rm_scores(smoke_metadata, prefix):
    full_scores = np.asarray(
        np.load(RM_CACHE / f"{prefix}.scores_f32.npy"),
        dtype=np.float32,
    )
    full_metadata = read_jsonl(
        RM_CACHE / f"{prefix}.metadata.jsonl"
    )

    table = {}
    for score, item in zip(full_scores, full_metadata):
        key = (
            str(item["question_uid"]),
            int(item.get("candidate_index", -1)),
        )
        table[key] = float(score)

    result = []
    for item in smoke_metadata:
        key = (
            str(item["question_uid"]),
            int(item.get("candidate_index", -1)),
        )
        if key not in table:
            raise KeyError(
                f"奖励分数中找不到候选：{key}"
            )
        result.append(table[key])

    return np.asarray(result, dtype=np.float32)


def build_pairs(labels, metadata, seed):
    rng = np.random.default_rng(seed)
    groups = groups_from_metadata(metadata)
    positive_indices = []
    negative_indices = []

    for indices in groups.values():
        indices = np.asarray(indices, dtype=np.int64)
        positives = indices[labels[indices] == 1]
        negatives = indices[labels[indices] == 0]

        pairs = np.asarray(
            [
                (positive, negative)
                for positive in positives
                for negative in negatives
            ],
            dtype=np.int64,
        )

        if len(pairs) > PAIR_CAP:
            chosen = rng.choice(
                len(pairs),
                size=PAIR_CAP,
                replace=False,
            )
            pairs = pairs[chosen]

        positive_indices.extend(pairs[:, 0].tolist())
        negative_indices.extend(pairs[:, 1].tolist())

    return (
        np.asarray(positive_indices, dtype=np.int64),
        np.asarray(negative_indices, dtype=np.int64),
    )


def fit_layer_probes(
    feature_array,
    labels,
    metadata,
    seed,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    features = torch.from_numpy(
        np.array(feature_array, copy=True)
    ).to(DEVICE, dtype=torch.float32)

    feature_mean = features.mean(
        dim=1,
        keepdim=True,
    )
    feature_std = features.std(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-4)

    features.sub_(feature_mean).div_(feature_std)

    layer_count, _, hidden_size = features.shape

    weights = torch.nn.Parameter(
        torch.zeros(
            layer_count,
            hidden_size,
            device=DEVICE,
            dtype=torch.float32,
        )
    )

    optimizer = torch.optim.AdamW(
        [weights],
        lr=0.03,
        weight_decay=1e-3,
    )

    positive, negative = build_pairs(
        labels,
        metadata,
        seed,
    )
    rng = np.random.default_rng(seed)

    final_loss = None

    for epoch in range(EPOCHS):
        order = rng.permutation(len(positive))
        losses = []

        for start in range(
            0,
            len(order),
            PAIR_BATCH,
        ):
            batch_order = order[
                start:start + PAIR_BATCH
            ]

            positive_tensor = torch.as_tensor(
                positive[batch_order],
                device=DEVICE,
            )
            negative_tensor = torch.as_tensor(
                negative[batch_order],
                device=DEVICE,
            )

            difference = (
                features[:, positive_tensor, :]
                - features[:, negative_tensor, :]
            )

            margin = torch.einsum(
                "lbh,lh->lb",
                difference,
                weights,
            )

            loss = F.softplus(-margin).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [weights],
                max_norm=5.0,
            )
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            del difference, margin, loss

        final_loss = float(np.mean(losses))

        if epoch in {0, 9, 19, EPOCHS - 1}:
            print(
                f"  seed={seed} epoch={epoch + 1:02d}/"
                f"{EPOCHS}, loss={final_loss:.6f}",
                flush=True,
            )

    result = {
        "weights": weights.detach().cpu().numpy(),
        "mean": feature_mean[
            :, 0, :
        ].detach().cpu().numpy(),
        "std": feature_std[
            :, 0, :
        ].detach().cpu().numpy(),
        "pairs": len(positive),
        "final_loss": final_loss,
    }

    del features, feature_mean, feature_std
    del weights, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def score_layers(feature_array, fitted):
    layer_count = feature_array.shape[0]
    candidate_count = feature_array.shape[1]
    result = np.empty(
        (layer_count, candidate_count),
        dtype=np.float32,
    )

    batch_size = 512

    for start in range(
        0,
        candidate_count,
        batch_size,
    ):
        end = min(
            start + batch_size,
            candidate_count,
        )

        features = np.asarray(
            feature_array[:, start:end, :],
            dtype=np.float32,
        )
        features = (
            features
            - fitted["mean"][:, None, :]
        ) / fitted["std"][:, None, :]

        result[:, start:end] = np.einsum(
            "lnh,lh->ln",
            features,
            fitted["weights"],
            optimize=True,
        )

    return result


def layer_name(index, layer_count):
    if index == layer_count - 1:
        return "final_norm"
    return f"block_{index + 1:02d}"


def evaluate_method(
    dataset,
    scores,
    raw_scores,
):
    metrics = ranking_metrics(
        dataset["labels"],
        dataset["metadata"],
        scores,
    )
    raw = ranking_metrics(
        dataset["labels"],
        dataset["metadata"],
        raw_scores,
    )
    effect = selection_effect(
        dataset["labels"],
        dataset["metadata"],
        raw_scores,
        scores,
    )

    return {
        "questions": metrics["questions"],
        "raw_top1": raw["top1"],
        "raw_pair_macro_strict": raw[
            "pair_macro_strict"
        ],
        "top1": metrics["top1"],
        "top1_delta": (
            metrics["top1"] - raw["top1"]
        ),
        "pair_macro_strict": metrics[
            "pair_macro_strict"
        ],
        "pair_delta": (
            metrics["pair_macro_strict"]
            - raw["pair_macro_strict"]
        ),
        **effect,
    }


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def summarize_seed_metrics(seed_metrics):
    fields = [
        "top1",
        "top1_delta",
        "pair_macro_strict",
        "pair_delta",
        "damage_rate",
        "correction_rate",
    ]
    return {
        field: mean_std(
            [item[field] for item in seed_metrics]
        )
        for field in fields
    }


def run_prompt(domain_name, spec, prompt_name):
    model_name = spec["model"]

    print()
    print("=" * 72)
    print(
        f"{domain_name} / {model_name} / {prompt_name}",
        flush=True,
    )

    train = load_cache(
        model_name,
        prompt_name,
        spec["train"],
    )
    pilot = load_cache(
        model_name,
        prompt_name,
        spec["pilot"],
    )
    tests = {
        display: load_cache(
            model_name,
            prompt_name,
            dataset_name,
        )
        for display, dataset_name
        in spec["tests"].items()
    }

    fitted_models = []
    pilot_layer_scores = []
    pilot_layer_metrics = []

    for seed in SEEDS:
        fitted = fit_layer_probes(
            train["hidden"],
            train["labels"],
            train["metadata"],
            seed,
        )
        fitted_models.append(fitted)

        layer_scores = score_layers(
            pilot["hidden"],
            fitted,
        )
        pilot_layer_scores.append(layer_scores)

        metrics = [
            ranking_metrics(
                pilot["labels"],
                pilot["metadata"],
                layer_scores[layer_index],
            )
            for layer_index
            in range(layer_scores.shape[0])
        ]
        pilot_layer_metrics.append(metrics)

    layer_count = pilot["hidden"].shape[0]
    layer_table = []

    for layer_index in range(layer_count):
        pair_values = [
            run[layer_index]["pair_macro_strict"]
            for run in pilot_layer_metrics
        ]
        top1_values = [
            run[layer_index]["top1"]
            for run in pilot_layer_metrics
        ]

        layer_table.append({
            "index": layer_index,
            "name": layer_name(
                layer_index,
                layer_count,
            ),
            "pilot_pair_mean": float(
                np.mean(pair_values)
            ),
            "pilot_pair_std": float(
                np.std(pair_values)
            ),
            "pilot_top1_mean": float(
                np.mean(top1_values)
            ),
            "pilot_top1_std": float(
                np.std(top1_values)
            ),
        })

    selected = max(
        layer_table,
        key=lambda item: (
            item["pilot_pair_mean"],
            item["pilot_top1_mean"],
        ),
    )
    selected_index = selected["index"]

    print("Pilot 最优的五层：")
    for item in sorted(
        layer_table,
        key=lambda value: (
            value["pilot_pair_mean"],
            value["pilot_top1_mean"],
        ),
        reverse=True,
    )[:5]:
        print(
            f"  {item['name']}: "
            f"Pair={item['pilot_pair_mean']:.6f}, "
            f"Top1={item['pilot_top1_mean']:.6f}",
            flush=True,
        )

    print(
        "最终选择层：",
        selected["name"],
        flush=True,
    )

    all_sets = {
        "PILOT": pilot,
        **tests,
    }
    dataset_name_map = {
        "PILOT": spec["pilot"],
        **{
            display: dataset_name
            for display, dataset_name
            in spec["tests"].items()
        },
    }

    evaluations = {}

    for display, dataset in all_sets.items():
        dataset_name = dataset_name_map[display]
        rm_prefix = spec["rm_prefix"][dataset_name]
        raw_scores = load_rm_scores(
            dataset["metadata"],
            rm_prefix,
        )

        hidden_seed_scores = [
            score_layers(
                dataset["hidden"],
                fitted,
            )[selected_index]
            for fitted in fitted_models
        ]
        hidden_seed_metrics = [
            evaluate_method(
                dataset,
                scores,
                raw_scores,
            )
            for scores in hidden_seed_scores
        ]
        hidden_ensemble = np.mean(
            hidden_seed_scores,
            axis=0,
        )

        direct_nll_score = -dataset["nll"][:, 0]

        # 五种 NLL 统计组成一个小型线性探针。
        nll_train_features = train[
            "nll"
        ][None, :, :]
        nll_fitted_models = [
            fit_layer_probes(
                nll_train_features,
                train["labels"],
                train["metadata"],
                seed,
            )
            for seed in SEEDS
        ]
        nll_seed_scores = [
            score_layers(
                dataset["nll"][None, :, :],
                fitted,
            )[0]
            for fitted in nll_fitted_models
        ]
        nll_ensemble = np.mean(
            nll_seed_scores,
            axis=0,
        )

        evaluations[display] = {
            "raw_rm": ranking_metrics(
                dataset["labels"],
                dataset["metadata"],
                raw_scores,
            ),
            "negative_mean_nll": evaluate_method(
                dataset,
                direct_nll_score,
                raw_scores,
            ),
            "learned_nll5_ensemble": evaluate_method(
                dataset,
                nll_ensemble,
                raw_scores,
            ),
            "hidden_probe_seeds": hidden_seed_metrics,
            "hidden_probe_seed_summary": (
                summarize_seed_metrics(
                    hidden_seed_metrics
                )
            ),
            "hidden_probe_ensemble": evaluate_method(
                dataset,
                hidden_ensemble,
                raw_scores,
            ),
        }

        result = evaluations[display]
        print()
        print(display)
        print(
            "  Raw RM: "
            f"Top1={result['raw_rm']['top1']:.6f}, "
            f"Pair={result['raw_rm']['pair_macro_strict']:.6f}"
        )
        for method_name in [
            "negative_mean_nll",
            "learned_nll5_ensemble",
            "hidden_probe_ensemble",
        ]:
            item = result[method_name]
            print(
                f"  {method_name}: "
                f"Top1={item['top1']:.6f} "
                f"({item['top1_delta']:+.6f}), "
                f"Pair={item['pair_macro_strict']:.6f} "
                f"({item['pair_delta']:+.6f}), "
                f"Damage={item['damage_rate']:.6f}"
            )

    return {
        "domain": domain_name,
        "model": model_name,
        "prompt": prompt_name,
        "selection_rule": (
            "pilot mean pair-macro primary, "
            "pilot mean top1 secondary"
        ),
        "selected_layer": selected,
        "all_layers": layer_table,
        "evaluations": evaluations,
    }


def main():
    started = time.time()
    torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    results = {}
    selected_prompts = {}

    for domain_name, spec in DOMAINS.items():
        domain_results = {}

        for prompt_name in PROMPTS:
            result = run_prompt(
                domain_name,
                spec,
                prompt_name,
            )
            domain_results[prompt_name] = result

        selected_prompt = max(
            domain_results,
            key=lambda prompt: (
                domain_results[prompt][
                    "selected_layer"
                ]["pilot_pair_mean"],
                domain_results[prompt][
                    "selected_layer"
                ]["pilot_top1_mean"],
            ),
        )

        selected_prompts[domain_name] = selected_prompt
        results[domain_name] = domain_results

        print()
        print(
            f"{domain_name} Pilot 最终提示模板："
            f"{selected_prompt}"
        )

    output = {
        "version": "generator_hidden_probe_smoke_v1",
        "scope": (
            "small train/pilot/test subsets; "
            "test was not used for layer or prompt selection"
        ),
        "seeds": SEEDS,
        "pair_cap_per_question": PAIR_CAP,
        "epochs": EPOCHS,
        "selection_primary": (
            "pilot pair_macro_strict averaged over seeds"
        ),
        "selected_prompts": selected_prompts,
        "results": results,
        "elapsed_seconds": round(
            time.time() - started,
            3,
        ),
        "peak_gpu_gb": (
            round(
                torch.cuda.max_memory_allocated()
                / (1024 ** 3),
                3,
            )
            if torch.cuda.is_available()
            else 0.0
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("生成模型隐状态探针冒烟实验完成。")
    print("选择：", selected_prompts)
    print("结果：", OUTPUT)
    print("耗时秒：", output["elapsed_seconds"])
    print("显存峰值 GB：", output["peak_gpu_gb"])


if __name__ == "__main__":
    main()

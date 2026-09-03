from pathlib import Path
from collections import defaultdict
import hashlib
import json
import math

import numpy as np
import torch
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT / "configs/hidden_layer_selection.json"
FEATURE_ROOT = (
    PROJECT
    / "data/cache/layer_discovery_v1"
    / "Skywork-Reward-V2-Qwen3-1.7B"
)
CACHE_ROOT = (
    PROJECT
    / "data/cache/layer_selection_v1"
    / "Skywork-Reward-V2-Qwen3-1.7B"
)
OUTPUT_PATH = PROJECT / "outputs/layer_selection_1p7b.json"
MANIFEST_PATH = (
    PROJECT
    / "data/manifests/selected_hidden_layer_1p7b.json"
)

DATASETS = {
    "GSM8K": "gsm_layer_discovery",
    "MATH": "math_layer_discovery",
}

DEVICE = "cuda"


config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
cv_config = config["cross_validation"]
optimizer_config = config["optimizer"]
selection_config = config["selection"]

SEED = int(cv_config["seed"])
FOLDS = int(cv_config["folds"])
EPOCHS = int(optimizer_config["epochs"])
PAIR_BATCH_SIZE = int(
    optimizer_config["pair_batch_size"]
)
LEARNING_RATE = float(
    optimizer_config["learning_rate"]
)
WEIGHT_DECAY = float(
    optimizer_config["weight_decay"]
)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.set_float32_matmul_precision("high")

CACHE_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stable_hash(text):
    raw = f"{SEED}|{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def assign_question_folds(metadata):
    fold_ids = np.empty(len(metadata), dtype=np.int64)

    for dataset in DATASETS:
        question_ids = sorted({
            row["question_uid"]
            for row in metadata
            if row["dataset"] == dataset
        })

        question_ids.sort(
            key=lambda uid: stable_hash(
                f"{dataset}|{uid}"
            )
        )

        question_to_fold = {
            uid: index % FOLDS
            for index, uid in enumerate(question_ids)
        }

        for index, row in enumerate(metadata):
            if row["dataset"] == dataset:
                fold_ids[index] = question_to_fold[
                    row["question_uid"]
                ]

    return fold_ids


def build_training_pairs(
    metadata,
    labels,
    train_indices,
):
    groups = defaultdict(list)

    for index in train_indices:
        row = metadata[index]
        key = (
            row["dataset"],
            row["question_uid"],
        )
        groups[key].append(index)

    question_counts = {
        dataset: len({
            question_uid
            for dataset_name, question_uid in groups
            if dataset_name == dataset
        })
        for dataset in DATASETS
    }

    positive_indices = []
    negative_indices = []
    pair_weights = []

    for (dataset, _), indices in groups.items():
        positives = [
            index
            for index in indices
            if labels[index] == 1
        ]
        negatives = [
            index
            for index in indices
            if labels[index] == 0
        ]

        pair_count = len(positives) * len(negatives)

        if pair_count == 0:
            continue

        question_weight = (
            0.5
            / question_counts[dataset]
            / pair_count
        )

        for positive in positives:
            for negative in negatives:
                positive_indices.append(positive)
                negative_indices.append(negative)
                pair_weights.append(question_weight)

    positive_indices = np.asarray(
        positive_indices,
        dtype=np.int64,
    )
    negative_indices = np.asarray(
        negative_indices,
        dtype=np.int64,
    )
    pair_weights = np.asarray(
        pair_weights,
        dtype=np.float32,
    )

    pair_weights /= pair_weights.mean()

    return (
        positive_indices,
        negative_indices,
        pair_weights,
    )


def ranking_metrics(
    scores,
    labels,
    metadata,
    indices,
    dataset,
):
    groups = defaultdict(list)

    for index in indices:
        row = metadata[index]
        if row["dataset"] == dataset:
            groups[row["question_uid"]].append(index)

    top1_values = []
    pair_values = []

    for group_indices in groups.values():
        best_index = max(
            group_indices,
            key=lambda i: float(scores[i]),
        )
        top1_values.append(
            float(labels[best_index] == 1)
        )

        positives = [
            float(scores[i])
            for i in group_indices
            if labels[i] == 1
        ]
        negatives = [
            float(scores[i])
            for i in group_indices
            if labels[i] == 0
        ]

        comparisons = [
            float(positive > negative)
            for positive in positives
            for negative in negatives
        ]

        if comparisons:
            pair_values.append(
                sum(comparisons) / len(comparisons)
            )

    return {
        "questions": len(groups),
        "top1": float(np.mean(top1_values)),
        "pair_macro_strict": float(
            np.mean(pair_values)
        ),
    }


print("加载28层 discovery 特征……")

feature_parts = []
label_parts = []
metadata = []

for dataset, stem in DATASETS.items():
    features = np.load(
        FEATURE_ROOT / f"{stem}.terminal_states_f16.npy",
        mmap_mode="r",
    )
    labels = np.load(
        FEATURE_ROOT / f"{stem}.labels_i8.npy"
    )
    rows = read_jsonl(
        FEATURE_ROOT / f"{stem}.metadata.jsonl"
    )

    if features.shape[1] != len(labels):
        raise RuntimeError("特征与标签数量不一致")

    if len(rows) != len(labels):
        raise RuntimeError("元数据与标签数量不一致")

    feature_parts.append(np.asarray(features))
    label_parts.append(labels)

    for row in rows:
        row = dict(row)
        row["dataset"] = dataset
        metadata.append(row)

features_numpy = np.concatenate(
    feature_parts,
    axis=1,
)
labels = np.concatenate(label_parts).astype(np.int8)

del feature_parts

num_layers, candidate_count, hidden_size = (
    features_numpy.shape
)

if num_layers != 28 or hidden_size != 2048:
    raise RuntimeError(
        f"特征形状异常：{features_numpy.shape}"
    )

fold_ids = assign_question_folds(metadata)

print("总候选：", candidate_count)
print("特征形状：", features_numpy.shape)
print(
    "各折候选数：",
    {
        fold: int(np.sum(fold_ids == fold))
        for fold in range(FOLDS)
    },
)

print("把特征放入显卡……")
features_gpu = torch.from_numpy(
    features_numpy
).to(
    device=DEVICE,
    dtype=torch.float16,
)

del features_numpy
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

oof_scores = np.full(
    (num_layers, candidate_count),
    np.nan,
    dtype=np.float32,
)
fold_training = []

for fold in range(FOLDS):
    print()
    print("=" * 72)
    print(f"训练第 {fold + 1}/{FOLDS} 折")

    train_indices = np.flatnonzero(fold_ids != fold)
    validation_indices = np.flatnonzero(
        fold_ids == fold
    )

    (
        positive_indices,
        negative_indices,
        pair_weights,
    ) = build_training_pairs(
        metadata,
        labels,
        train_indices,
    )

    print("训练问题候选：", len(train_indices))
    print("验证问题候选：", len(validation_indices))
    print("训练正负对：", len(positive_indices))

    train_index_tensor = torch.tensor(
        train_indices,
        device=DEVICE,
        dtype=torch.long,
    )

    training_features = (
        features_gpu
        .index_select(1, train_index_tensor)
        .float()
    )

    feature_std = (
        training_features
        .std(dim=1, unbiased=False)
        .clamp_min(1e-3)
    )

    del training_features

    probe_weights = torch.nn.Parameter(
        torch.zeros(
            num_layers,
            hidden_size,
            device=DEVICE,
            dtype=torch.float32,
        )
    )

    optimizer = torch.optim.AdamW(
        [probe_weights],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LEARNING_RATE / 100,
    )

    pair_count = len(positive_indices)
    final_epoch_loss = None

    for epoch in range(EPOCHS):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            SEED + fold * 1000 + epoch
        )
        order = torch.randperm(
            pair_count,
            generator=generator,
        ).numpy()

        epoch_loss_sum = 0.0
        epoch_examples = 0

        for start in range(
            0,
            pair_count,
            PAIR_BATCH_SIZE,
        ):
            batch_order = order[
                start:start + PAIR_BATCH_SIZE
            ]

            positive_tensor = torch.tensor(
                positive_indices[batch_order],
                device=DEVICE,
                dtype=torch.long,
            )
            negative_tensor = torch.tensor(
                negative_indices[batch_order],
                device=DEVICE,
                dtype=torch.long,
            )
            weight_tensor = torch.tensor(
                pair_weights[batch_order],
                device=DEVICE,
                dtype=torch.float32,
            )

            positive_features = (
                features_gpu[:, positive_tensor, :]
                .float()
            )
            negative_features = (
                features_gpu[:, negative_tensor, :]
                .float()
            )

            differences = (
                positive_features - negative_features
            ) / feature_std[:, None, :]

            logits = torch.einsum(
                "lbd,ld->lb",
                differences,
                probe_weights,
            )

            losses = F.softplus(-logits)
            loss = (
                losses
                * weight_tensor[None, :]
            ).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = len(batch_order)
            epoch_loss_sum += (
                float(loss.item()) * batch_size
            )
            epoch_examples += batch_size

            del (
                positive_features,
                negative_features,
                differences,
                logits,
                losses,
            )

        scheduler.step()
        final_epoch_loss = (
            epoch_loss_sum / epoch_examples
        )

        if epoch in {0, 19, 39, EPOCHS - 1}:
            print(
                f"epoch {epoch + 1:02d}/{EPOCHS}, "
                f"loss={final_epoch_loss:.6f}",
                flush=True,
            )

    validation_tensor = torch.tensor(
        validation_indices,
        device=DEVICE,
        dtype=torch.long,
    )

    with torch.inference_mode():
        validation_features = (
            features_gpu
            .index_select(1, validation_tensor)
            .float()
            / feature_std[:, None, :]
        )
        validation_scores = torch.einsum(
            "lnd,ld->ln",
            validation_features,
            probe_weights,
        )

    oof_scores[:, validation_indices] = (
        validation_scores.cpu().numpy()
    )

    fold_training.append({
        "fold": fold,
        "training_candidates": len(train_indices),
        "validation_candidates": len(
            validation_indices
        ),
        "training_pairs": pair_count,
        "final_epoch_loss": final_epoch_loss,
    })

    del (
        probe_weights,
        optimizer,
        scheduler,
        validation_features,
        validation_scores,
        feature_std,
    )
    torch.cuda.empty_cache()

if np.isnan(oof_scores).any():
    raise RuntimeError("OOF 分数存在缺失")

all_indices = np.arange(candidate_count)
layer_results = []

for layer_offset in range(num_layers):
    layer_number = layer_offset + 1
    scores = oof_scores[layer_offset]

    gsm = ranking_metrics(
        scores,
        labels,
        metadata,
        all_indices,
        "GSM8K",
    )
    math_result = ranking_metrics(
        scores,
        labels,
        metadata,
        all_indices,
        "MATH",
    )


    macro_pair = (
        gsm["pair_macro_strict"]
        + math_result["pair_macro_strict"]
    ) / 2
    macro_top1 = (
        gsm["top1"]
        + math_result["top1"]
    ) / 2

    fold_macro_scores = []

    for fold in range(FOLDS):
        fold_indices = np.flatnonzero(
            fold_ids == fold
        )
        fold_gsm = ranking_metrics(
            scores,
            labels,
            metadata,
            fold_indices,
            "GSM8K",
        )
        fold_math = ranking_metrics(
            scores,
            labels,
            metadata,
            fold_indices,
            "MATH",
        )
        fold_macro_scores.append(
            (
                fold_gsm["pair_macro_strict"]
                + fold_math["pair_macro_strict"]
            )
            / 2
        )

    standard_error = float(
        np.std(
            fold_macro_scores,
            ddof=1,
        )
        / math.sqrt(FOLDS)
    )

    layer_results.append({
        "layer": layer_number,
        "gsm8k": gsm,
        "math": math_result,
        "dataset_macro_pair_macro_strict": (
            macro_pair
        ),
        "dataset_macro_top1": macro_top1,
        "fold_macro_pair_scores": fold_macro_scores,
        "standard_error": standard_error,
    })

best_result = max(
    layer_results,
    key=lambda row: (
        row["dataset_macro_pair_macro_strict"],
        row["dataset_macro_top1"],
    ),
)
final_result = layer_results[-1]

macro_improvement = (
    best_result["dataset_macro_pair_macro_strict"]
    - final_result["dataset_macro_pair_macro_strict"]
)

per_dataset_ok = (
    best_result["gsm8k"]["pair_macro_strict"]
    >= final_result["gsm8k"]["pair_macro_strict"]
    - float(
        selection_config[
            "maximum_allowed_per_dataset_regression"
        ]
    )
    and
    best_result["math"]["pair_macro_strict"]
    >= final_result["math"]["pair_macro_strict"]
    - float(
        selection_config[
            "maximum_allowed_per_dataset_regression"
        ]
    )
)

if (
    best_result["layer"] != 28
    and macro_improvement
    >= float(
        selection_config[
            "minimum_macro_improvement_over_final"
        ]
    )
    and per_dataset_ok
):
    selected_layer = best_result["layer"]
    selection_reason = (
        "Best intermediate layer cleared the "
        "pre-registered improvement criteria."
    )
else:
    selected_layer = 28
    selection_reason = (
        "Intermediate-layer advantage did not clear "
        "the pre-registered threshold; use final layer."
    )

result = {
    "version": "hidden_layer_selection_result_v1",
    "model": "Skywork-Reward-V2-Qwen3-1.7B",
    "scope": "layer_discovery_only",
    "candidate_count": candidate_count,
    "fold_training": fold_training,
    "layer_results": layer_results,
    "best_raw_layer": best_result["layer"],
    "reference_final_layer": 28,
    "macro_improvement_over_final": (
        macro_improvement
    ),
    "per_dataset_guard_passed": per_dataset_ok,
    "selected_layer": selected_layer,
    "selection_reason": selection_reason,
    "peak_gpu_gb": round(
        torch.cuda.max_memory_allocated() / 1024**3,
        3,
    ),
}

np.save(
    CACHE_ROOT / "oof_scores_f32.npy",
    oof_scores,
)

OUTPUT_PATH.write_text(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

MANIFEST_PATH.write_text(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print()
print("=" * 72)
print("===== 所有层结果 =====")

for row in layer_results:
    print(
        f"Layer {row['layer']:02d} | "
        f"GSM Pair={row['gsm8k']['pair_macro_strict']:.4f} | "
        f"MATH Pair={row['math']['pair_macro_strict']:.4f} | "
        f"Macro={row['dataset_macro_pair_macro_strict']:.4f} | "
        f"Top1={row['dataset_macro_top1']:.4f}"
    )

print()
print("===== 选层结论 =====")
print("原始最优层：", best_result["layer"])
print(
    "最优层宏平均 Pair-Macro：",
    round(
        best_result[
            "dataset_macro_pair_macro_strict"
        ],
        6,
    ),
)
print(
    "第28层宏平均 Pair-Macro：",
    round(
        final_result[
            "dataset_macro_pair_macro_strict"
        ],
        6,
    ),
)
print(
    "相对第28层提升：",
    round(macro_improvement, 6),
)
print("最终选择层：", selected_layer)
print("原因：", selection_reason)
print(
    "显存峰值 GB：",
    result["peak_gpu_gb"],
)
print("结果文件：", OUTPUT_PATH)

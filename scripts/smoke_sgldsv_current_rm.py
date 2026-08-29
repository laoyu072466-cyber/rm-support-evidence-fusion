from pathlib import Path
from collections import defaultdict
import copy
import gc
import hashlib
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


ROOT = Path("/root/autodl-tmp/rm_traj_project")
MODEL_PATH = (
    ROOT / "models/reward/"
    "Skywork-Reward-V2-Qwen3-1.7B"
)
DATA_PATH = ROOT / "data/processed/prototype_v2"
CACHE_PATH = (
    ROOT / "data/cache/"
    "sgldsv_smoke_v1/"
    "Skywork-Reward-V2-Qwen3-1.7B/"
    "block_21"
)
OUTPUT_PATH = ROOT / "outputs/sgldsv_smoke_current_rm.json"
CHECKPOINT_PATH = (
    ROOT / "outputs/checkpoints/"
    "sgldsv_smoke_current_rm_seed42.pt"
)

DEVICE = torch.device("cuda")
BLOCK_NUMBER = 21
BLOCK_INDEX = BLOCK_NUMBER - 1
HIDDEN_SIZE = 2048

TRAIN_QUESTIONS = 64
VALIDATION_QUESTIONS = 32
EXTRACTION_BATCH_SIZE = 32
PAIR_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 48
PAIR_CAP_PER_QUESTION = 256

SEED = 42
MAX_EPOCHS = 5
PATIENCE = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_float32_matmul_precision("high")


def stable_hash(text):
    return hashlib.sha256(
        str(text).encode("utf-8")
    ).hexdigest()


def read_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def select_questions(rows, question_count):
    groups = defaultdict(list)

    for row in rows:
        groups[row["question_uid"]].append(row)

    ordered_uids = sorted(
        groups,
        key=stable_hash,
    )
    selected_uids = ordered_uids[:question_count]

    selected_rows = []

    for uid in selected_uids:
        labels = {
            int(row["label"])
            for row in groups[uid]
        }

        if labels != {0, 1}:
            raise RuntimeError(
                f"问题 {uid} 不是正负混合问题"
            )

        selected_rows.extend(groups[uid])

    return selected_rows, selected_uids


def build_tokenized_record(tokenizer, row):
    solution = str(row["solution_text"])
    conversation = [
        {
            "role": "user",
            "content": str(row["problem"]),
        },
        {
            "role": "assistant",
            "content": solution,
        },
    ]

    rendered = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False,
    )
    template_ids = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=False,
    )

    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    if input_ids != template_ids:
        raise RuntimeError(
            "渲染文本重新分词后与 "
            "apply_chat_template 不一致"
        )

    rendered_solution = solution.lstrip("\n")
    solution_start = rendered.rfind(
        rendered_solution
    )

    if solution_start < 0:
        raise RuntimeError(
            "无法在 chat template 中定位回答正文"
        )

    solution_end = (
        solution_start + len(rendered_solution)
    )

    response_positions = []

    for token_index, (left, right) in enumerate(
        offsets
    ):
        if right <= left:
            continue

        overlap = (
            right > solution_start
            and left < solution_end
        )
        if overlap:
            response_positions.append(token_index)

    if not response_positions:
        raise RuntimeError("回答正文没有映射到 token")

    if len(input_ids) > 40960:
        raise RuntimeError("输入超过模型上下文")

    return {
        "input_ids": input_ids,
        "response_positions": response_positions,
        "input_length": len(input_ids),
        "response_length": len(response_positions),
        "rendered_solution": rendered_solution,
    }


def cache_paths(prefix):
    return {
        "features": (
            CACHE_PATH
            / f"{prefix}.response_states_f16.npy"
        ),
        "offsets": (
            CACHE_PATH
            / f"{prefix}.offsets_i64.npy"
        ),
        "scores": (
            CACHE_PATH
            / f"{prefix}.rm_scores_f32.npy"
        ),
        "labels": (
            CACHE_PATH
            / f"{prefix}.labels_i8.npy"
        ),
        "metadata": (
            CACHE_PATH
            / f"{prefix}.metadata.jsonl"
        ),
    }


def extract_cache(
    backbone,
    tokenizer,
    rows,
    prefix,
):
    paths = cache_paths(prefix)
    CACHE_PATH.mkdir(parents=True, exist_ok=True)

    tokenized = []
    input_lengths = []
    response_lengths = []

    print(f"\n准备 {prefix} 的输入与响应 token mask……")

    for index, row in enumerate(rows):
        item = build_tokenized_record(
            tokenizer,
            row,
        )
        tokenized.append(item)
        input_lengths.append(item["input_length"])
        response_lengths.append(
            item["response_length"]
        )

        if index == 0:
            response_ids = [
                item["input_ids"][position]
                for position
                in item["response_positions"]
            ]
            decoded = tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
            )
            print(
                "第一条回答 token 解码预览：",
                repr(decoded[:240]),
            )
            print(
                "第一条原回答预览：",
                repr(
                    item["rendered_solution"][:240]
                ),
            )

    captured = {}

    def hook_fn(module, inputs, output):
        if isinstance(output, tuple):
            captured["hidden"] = output[0]
        else:
            captured["hidden"] = output

    layer = backbone.model.layers[BLOCK_INDEX]
    hook_handle = layer.register_forward_hook(hook_fn)

    feature_parts = []
    offsets = []
    rm_scores = np.empty(
        len(rows),
        dtype=np.float32,
    )
    labels = np.asarray(
        [int(row["label"]) for row in rows],
        dtype=np.int8,
    )
    metadata = []

    endpoint = 0
    start_time = time.time()

    try:
        for batch_start in range(
            0,
            len(rows),
            EXTRACTION_BATCH_SIZE,
        ):
            batch_end = min(
                batch_start + EXTRACTION_BATCH_SIZE,
                len(rows),
            )
            batch_items = tokenized[
                batch_start:batch_end
            ]

            padded = tokenizer.pad(
                {
                    "input_ids": [
                        item["input_ids"]
                        for item in batch_items
                    ]
                },
                padding=True,
                return_tensors="pt",
            )
            padded = {
                key: value.to(DEVICE)
                for key, value in padded.items()
            }

            captured.clear()

            with torch.inference_mode():
                output = backbone(**padded)

            hidden = captured.get("hidden")

            if hidden is None:
                raise RuntimeError(
                    "第21层 forward hook 没有捕获状态"
                )

            logits = (
                output.logits
                .detach()
                .float()
                .view(-1)
                .cpu()
                .numpy()
            )
            rm_scores[batch_start:batch_end] = logits

            for local_index, item in enumerate(
                batch_items
            ):
                positions = torch.tensor(
                    item["response_positions"],
                    device=DEVICE,
                    dtype=torch.long,
                )
                response_states = hidden[
                    local_index
                ].index_select(
                    0,
                    positions,
                )
                response_states = (
                    response_states
                    .detach()
                    .to(torch.float16)
                    .cpu()
                    .numpy()
                )

                left = endpoint
                right = left + len(response_states)
                offsets.append([left, right])
                endpoint = right
                feature_parts.append(response_states)

                row_index = batch_start + local_index
                row = rows[row_index]

                metadata.append({
                    "question_uid": row["question_uid"],
                    "problem_id": row.get("problem_id"),
                    "candidate_index": row.get(
                        "candidate_index"
                    ),
                    "label": int(row["label"]),
                    "input_tokens": item["input_length"],
                    "response_tokens": item[
                        "response_length"
                    ],
                })

            completed = batch_end
            elapsed = time.time() - start_time
            print(
                f"{prefix}: {completed}/{len(rows)}, "
                f"{completed / max(elapsed, 1e-6):.1f} "
                "候选/秒",
                flush=True,
            )

            del hidden, output, padded
            captured.clear()

    finally:
        hook_handle.remove()

    feature_array = np.concatenate(
        feature_parts,
        axis=0,
    )
    offset_array = np.asarray(
        offsets,
        dtype=np.int64,
    )

    if feature_array.shape != (
        endpoint,
        HIDDEN_SIZE,
    ):
        raise RuntimeError(
            f"缓存形状错误：{feature_array.shape}"
        )

    np.save(paths["features"], feature_array)
    np.save(paths["offsets"], offset_array)
    np.save(paths["scores"], rm_scores)
    np.save(paths["labels"], labels)

    with paths["metadata"].open(
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

    result = {
        "prefix": prefix,
        "candidates": len(rows),
        "response_tokens": int(endpoint),
        "feature_shape": list(feature_array.shape),
        "feature_gb": round(
            feature_array.nbytes / 1024 ** 3,
            4,
        ),
        "input_tokens": {
            "min": int(np.min(input_lengths)),
            "median": float(
                np.median(input_lengths)
            ),
            "max": int(np.max(input_lengths)),
        },
        "response_token_length": {
            "min": int(np.min(response_lengths)),
            "median": float(
                np.median(response_lengths)
            ),
            "max": int(np.max(response_lengths)),
        },
        "elapsed_seconds": round(
            time.time() - start_time,
            3,
        ),
        "files": {
            key: str(value)
            for key, value in paths.items()
        },
    }

    print(json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    ))
    return result


class SGLDSVHead(nn.Module):
    def __init__(self, hidden_size=2048):
        super().__init__()

        self.projection = nn.Linear(
            hidden_size,
            128,
            bias=False,
        )
        self.projection_norm = nn.LayerNorm(
            128,
            elementwise_affine=False,
        )

        self.transition_embedding = nn.Linear(
            128,
            32,
        )
        self.source_gate = nn.Linear(
            128,
            1,
        )

        self.local_scorer = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )

        self.endpoint_scorer = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.local_scale = nn.Parameter(
            torch.tensor(1.0)
        )

    def forward(self, states, mask):
        states = states.float()
        mask = mask.bool()

        projected = self.projection_norm(
            self.projection(states)
        )

        batch_size, token_count, _ = states.shape

        if token_count >= 2:
            transitions = (
                projected[:, 1:]
                - projected[:, :-1]
            )
            embedded = F.relu(
                self.transition_embedding(
                    transitions
                )
            )

            valid_transition = (
                mask[:, 1:]
                & mask[:, :-1]
            )

            gate = torch.sigmoid(
                self.source_gate(
                    projected[:, :-1]
                ).squeeze(-1)
            )
            weights = (
                gate
                * valid_transition.float()
            )

            denominator = (
                weights.sum(dim=1, keepdim=True)
                .clamp_min(1e-8)
            )

            weighted_mean = (
                weights.unsqueeze(-1)
                * embedded
            ).sum(dim=1) / denominator

            centered = (
                embedded
                - weighted_mean.unsqueeze(1)
            )
            variance = (
                weights.unsqueeze(-1)
                * centered.square()
            ).sum(dim=1) / denominator

            weighted_std = torch.sqrt(
                variance.clamp_min(0.0) + 1e-8
            )

            local_summary = torch.cat(
                [weighted_mean, weighted_std],
                dim=-1,
            )
            local_score = self.local_scorer(
                local_summary
            ).squeeze(-1)

            has_transition = valid_transition.any(
                dim=1
            )
            local_score = torch.where(
                has_transition,
                local_score,
                torch.zeros_like(local_score),
            )
        else:
            local_score = torch.zeros(
                batch_size,
                device=states.device,
                dtype=torch.float32,
            )

        lengths = mask.sum(dim=1).long()
        last_index = (
            lengths - 1
        ).clamp_min(0)

        endpoint_state = states[
            torch.arange(
                batch_size,
                device=states.device,
            ),
            last_index,
        ]
        endpoint_score = self.endpoint_scorer(
            endpoint_state
        ).squeeze(-1)

        score = (
            endpoint_score
            + self.local_scale * local_score
        )

        return {
            "score": score,
            "endpoint_score": endpoint_score,
            "local_score": local_score,
        }


class CachedDataset:
    def __init__(self, prefix):
        paths = cache_paths(prefix)

        self.features = np.load(
            paths["features"],
            mmap_mode="r",
        )
        self.offsets = np.load(
            paths["offsets"],
        )
        self.rm_scores = np.load(
            paths["scores"],
        ).astype(np.float32)
        self.labels = np.load(
            paths["labels"],
        ).astype(np.int8)

        with paths["metadata"].open(
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        self.groups = defaultdict(list)

        for index, item in enumerate(self.metadata):
            self.groups[
                item["question_uid"]
            ].append(index)

        self.groups = dict(self.groups)

    def sequence(self, index):
        left, right = self.offsets[index]
        array = np.array(
            self.features[left:right],
            dtype=np.float32,
            copy=True,
        )
        return torch.from_numpy(array)


def pack_candidate_indices(cache, indices):
    sequences = [
        cache.sequence(int(index))
        for index in indices
    ]
    lengths = torch.tensor(
        [len(sequence) for sequence in sequences],
        dtype=torch.long,
    )

    padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0,
    ).to(DEVICE)

    positions = torch.arange(
        padded.shape[1],
        device=DEVICE,
    ).unsqueeze(0)

    mask = (
        positions
        < lengths.to(DEVICE).unsqueeze(1)
    )

    return padded, mask


def build_pairs(cache, seed):
    rng = np.random.default_rng(seed)
    all_pairs = []

    for uid, indices in cache.groups.items():
        positive = [
            index
            for index in indices
            if cache.labels[index] == 1
        ]
        negative = [
            index
            for index in indices
            if cache.labels[index] == 0
        ]

        pairs = np.asarray(
            [
                [pos, neg]
                for pos in positive
                for neg in negative
            ],
            dtype=np.int64,
        )

        if len(pairs) > PAIR_CAP_PER_QUESTION:
            selected = rng.choice(
                len(pairs),
                size=PAIR_CAP_PER_QUESTION,
                replace=False,
            )
            pairs = pairs[selected]

        all_pairs.append(pairs)

    return np.concatenate(all_pairs, axis=0)


@torch.no_grad()
def score_dataset(model, cache):
    model.eval()
    scores = np.empty(
        len(cache.labels),
        dtype=np.float32,
    )
    endpoint_scores = np.empty_like(scores)
    local_scores = np.empty_like(scores)

    for start in range(
        0,
        len(cache.labels),
        EVAL_BATCH_SIZE,
    ):
        end = min(
            start + EVAL_BATCH_SIZE,
            len(cache.labels),
        )
        indices = np.arange(start, end)
        states, mask = pack_candidate_indices(
            cache,
            indices,
        )

        output = model(states, mask)

        scores[start:end] = (
            output["score"]
            .cpu()
            .numpy()
        )
        endpoint_scores[start:end] = (
            output["endpoint_score"]
            .cpu()
            .numpy()
        )
        local_scores[start:end] = (
            output["local_score"]
            .cpu()
            .numpy()
        )

    return scores, endpoint_scores, local_scores


def ranking_metrics(cache, scores):
    top1 = []
    pair_macro = []

    for uid, indices in cache.groups.items():
        indices = np.asarray(indices)
        labels = cache.labels[indices]
        values = scores[indices]

        winner = int(np.argmax(values))
        top1.append(float(labels[winner] == 1))

        positive = values[labels == 1]
        negative = values[labels == 0]

        pair_macro.append(float(
            (
                positive[:, None]
                > negative[None, :]
            ).mean()
        ))

    return {
        "questions": len(cache.groups),
        "candidates": len(cache.labels),
        "top1": float(np.mean(top1)),
        "pair_macro_strict": float(
            np.mean(pair_macro)
        ),
        "positive_score_mean": float(
            np.mean(scores[cache.labels == 1])
        ),
        "negative_score_mean": float(
            np.mean(scores[cache.labels == 0])
        ),
    }


def train_smoke(train_cache, validation_cache):
    model = SGLDSVHead().to(DEVICE)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if parameter_count != 814564:
        raise RuntimeError(
            f"参数量错误：{parameter_count} != 814564"
        )

    pairs = build_pairs(train_cache, SEED)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    validation_raw = ranking_metrics(
        validation_cache,
        validation_cache.rm_scores,
    )

    print("\n===== SG-LDSV 训练 =====")
    print("可训练参数：", parameter_count)
    print("训练正负对：", len(pairs))
    print("验证集原始 RM：")
    print(json.dumps(
        validation_raw,
        ensure_ascii=False,
        indent=2,
    ))

    rng = np.random.default_rng(SEED)
    history = []
    best_pair = -math.inf
    best_epoch = None
    best_state = None
    patience_count = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        shuffled = pairs.copy()
        rng.shuffle(shuffled)
        losses = []

        for start in range(
            0,
            len(shuffled),
            PAIR_BATCH_SIZE,
        ):
            batch_pairs = shuffled[
                start:start + PAIR_BATCH_SIZE
            ]

            positive_indices = batch_pairs[:, 0]
            negative_indices = batch_pairs[:, 1]
            all_indices = np.concatenate(
                [
                    positive_indices,
                    negative_indices,
                ]
            )

            states, mask = pack_candidate_indices(
                train_cache,
                all_indices,
            )

            optimizer.zero_grad(set_to_none=True)
            scores = model(
                states,
                mask,
            )["score"]

            pair_count = len(batch_pairs)
            positive_scores = scores[:pair_count]
            negative_scores = scores[pair_count:]

            loss = F.softplus(
                -(
                    positive_scores
                    - negative_scores
                )
            ).mean()

            if not torch.isfinite(loss):
                raise RuntimeError("loss 非有限")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )
            optimizer.step()
            losses.append(float(loss.item()))

        validation_scores, _, _ = score_dataset(
            model,
            validation_cache,
        )
        validation_metrics = ranking_metrics(
            validation_cache,
            validation_scores,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(
                np.mean(losses)
            ),
            "validation": validation_metrics,
            "local_scale": float(
                model.local_scale.item()
            ),
        }
        history.append(epoch_record)

        print(
            f"epoch {epoch}/{MAX_EPOCHS} | "
            f"loss={epoch_record['train_loss']:.6f} | "
            f"val Top1="
            f"{validation_metrics['top1']:.6f} | "
            f"val Pair="
            f"{validation_metrics['pair_macro_strict']:.6f} | "
            f"lambda={epoch_record['local_scale']:.4f}",
            flush=True,
        )

        current_pair = validation_metrics[
            "pair_macro_strict"
        ]

        if current_pair > best_pair + 1e-12:
            best_pair = current_pair
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print("触发 patience=2，提前停止。")
            break

    if best_state is None:
        raise RuntimeError("没有产生 checkpoint")

    model.load_state_dict(best_state)

    validation_scores, endpoint_scores, local_scores = (
        score_dataset(
            model,
            validation_cache,
        )
    )
    validation_sgldsv = ranking_metrics(
        validation_cache,
        validation_scores,
    )

    torch.save(
        {
            "model_state_dict": best_state,
            "seed": SEED,
            "block_number": BLOCK_NUMBER,
            "parameter_count": parameter_count,
            "best_epoch": best_epoch,
            "validation": validation_sgldsv,
        },
        CHECKPOINT_PATH,
    )

    return {
        "parameter_count": parameter_count,
        "training_pairs": len(pairs),
        "best_epoch": best_epoch,
        "history": history,
        "validation_raw_rm": validation_raw,
        "validation_sgldsv": validation_sgldsv,
        "local_scale": float(
            model.local_scale.item()
        ),
        "endpoint_score_std": float(
            endpoint_scores.std()
        ),
        "local_score_std": float(
            local_scores.std()
        ),
        "checkpoint": str(CHECKPOINT_PATH),
    }


def main():
    overall_start = time.time()
    torch.cuda.reset_peak_memory_stats()

    train_rows = read_jsonl(
        DATA_PATH / "gsm_train.jsonl"
    )
    validation_rows = read_jsonl(
        DATA_PATH / "gsm_pilot_validation.jsonl"
    )

    selected_train, train_uids = select_questions(
        train_rows,
        TRAIN_QUESTIONS,
    )
    selected_validation, validation_uids = (
        select_questions(
            validation_rows,
            VALIDATION_QUESTIONS,
        )
    )

    print("===== SG-LDSV 当前 RM 冒烟实验 =====")
    print(
        f"训练：{len(train_uids)} 问题，"
        f"{len(selected_train)} 候选"
    )
    print(
        f"验证：{len(validation_uids)} 问题，"
        f"{len(selected_validation)} 候选"
    )
    print(
        f"冻结骨干：{MODEL_PATH}"
    )
    print("读取 block：", BLOCK_NUMBER)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "right"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            MODEL_PATH / "chat_template.jinja"
        ).read_text(encoding="utf-8")

    print("加载冻结奖励模型……")
    backbone = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
        )
        .to(DEVICE)
        .eval()
    )
    backbone.config.use_cache = False
    backbone.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    train_cache_summary = extract_cache(
        backbone,
        tokenizer,
        selected_train,
        "gsm_train_smoke",
    )
    validation_cache_summary = extract_cache(
        backbone,
        tokenizer,
        selected_validation,
        "gsm_pilot_smoke",
    )

    del backbone
    gc.collect()
    torch.cuda.empty_cache()

    print("\n冻结骨干已卸载，开始训练轻量头……")

    train_cache = CachedDataset(
        "gsm_train_smoke"
    )
    validation_cache = CachedDataset(
        "gsm_pilot_smoke"
    )

    training_result = train_smoke(
        train_cache,
        validation_cache,
    )

    result = {
        "version": "sgldsv_current_rm_smoke_v1",
        "scope": "64_train_questions_32_pilot_questions",
        "backbone": str(MODEL_PATH),
        "backbone_frozen": True,
        "block_number": BLOCK_NUMBER,
        "architecture": "exact SG-LDSV token-level adaptation",
        "train_selection_sha256": stable_hash(
            json.dumps(train_uids)
        ),
        "validation_selection_sha256": stable_hash(
            json.dumps(validation_uids)
        ),
        "cache": {
            "train": train_cache_summary,
            "validation": validation_cache_summary,
        },
        "training": training_result,
        "total_elapsed_seconds": round(
            time.time() - overall_start,
            3,
        ),
        "peak_gpu_gb": round(
            torch.cuda.max_memory_allocated()
            / 1024 ** 3,
            3,
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n===== SG-LDSV 冒烟最终结果 =====")
    print(json.dumps(
        {
            "raw_rm": training_result[
                "validation_raw_rm"
            ],
            "sgldsv": training_result[
                "validation_sgldsv"
            ],
            "best_epoch": training_result[
                "best_epoch"
            ],
            "local_scale": training_result[
                "local_scale"
            ],
            "parameter_count": training_result[
                "parameter_count"
            ],
            "total_elapsed_seconds": result[
                "total_elapsed_seconds"
            ],
            "peak_gpu_gb": result["peak_gpu_gb"],
            "checkpoint": training_result[
                "checkpoint"
            ],
            "result": str(OUTPUT_PATH),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

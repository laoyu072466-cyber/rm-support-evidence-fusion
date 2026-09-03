from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_multi_reward_reproduction as reproduction


CONFIG_PATH = (
    ROOT / "configs/"
    "controlled_reward_degradation_v1.json"
)
SCRIPT_PATH = Path(__file__).resolve()

PROTOCOL_TAG = (
    "controlled-reward-degradation-"
    "protocol-v1"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


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
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
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


def atomic_npy(path, values):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )
    with temporary.open("wb") as file:
        np.save(
            file,
            np.asarray(
                values,
                dtype=np.float32,
            ),
            allow_pickle=False,
        )
    temporary.replace(path)


def load_config():
    return json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )


def verify_protocol_tag():
    paths = [
        CONFIG_PATH,
        SCRIPT_PATH,
    ]

    for path in paths:
        relative = path.relative_to(ROOT)
        completed = subprocess.run(
            [
                "git",
                "show",
                f"{PROTOCOL_TAG}:{relative}",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )

        require(
            completed.returncode == 0,
            f"协议标签缺少：{relative}",
        )
        require(
            completed.stdout
            == path.read_bytes(),
            f"当前文件偏离协议标签：{relative}",
        )


def score_path(config, prefix):
    root = (
        ROOT
        / config["base_reward_model"][
            "score_root"
        ]
    )
    return (
        root
        / f"{prefix}.scores_f32.npy"
    )


def output_root(config):
    return (
        ROOT
        / config["outputs"]["score_root"]
    )


def variant_root(config, variant):
    return (
        output_root(config)
        / variant["id"]
    )


def variant_result_path(variant):
    model_key = (
        "controlled_" + variant["id"]
    )
    return (
        ROOT
        / "data/manifests/"
        / (
            "multi_reward_reproduction_"
            f"{model_key}_v1.json"
        )
    )


def count_jsonl(path):
    with path.open("rb") as file:
        return sum(
            1
            for line in file
            if line.strip()
        )


def preflight(require_frozen):
    config = load_config()

    print(
        "===== 受控奖励退化实验预检 ====="
    )
    print("分析类型：", config[
        "analysis_type"
    ])
    print(
        "基础 RM：",
        config["base_reward_model"]["name"],
    )
    print(
        "退化使用标签：",
        config["degradation"][
            "labels_used"
        ],
    )
    print(
        "Test 标签用于选择：",
        config["selection"][
            "test_labels_used_for_selection"
        ],
    )

    require(
        config["version"]
        == "controlled_reward_degradation_v1",
        "配置版本错误",
    )
    require(
        config["protocol_tag"]
        == PROTOCOL_TAG,
        "协议标签不一致",
    )
    require(
        config["degradation"][
            "labels_used"
        ] is False,
        "退化过程禁止使用标签",
    )
    require(
        config["selection"][
            "test_labels_used_for_selection"
        ] is False,
        "Test 标签禁止参与选择",
    )

    variants = config["degradation"][
        "variants"
    ]
    require(
        len(variants)
        == config["degradation"][
            "expected_variants"
        ] == 26,
        "退化变体数量错误",
    )
    require(
        len({
            item["id"]
            for item in variants
        }) == len(variants),
        "退化变体 ID 重复",
    )
    require(
        any(
            item["sigma"] == 0.0
            and item["seed"] == 0
            for item in variants
        ),
        "缺少零噪声对照",
    )

    for prefix, spec in (
        config["data"].items()
    ):
        data_path = ROOT / spec["file"]
        source_path = score_path(
            config,
            prefix,
        )

        require(
            data_path.exists(),
            f"缺少数据：{data_path}",
        )
        require(
            source_path.exists(),
            f"缺少源分数：{source_path}",
        )
        require(
            count_jsonl(data_path)
            == spec["count"],
            f"{prefix}: 数据行数错误",
        )
        require(
            sha256_file(data_path)
            == spec["data_sha256"],
            f"{prefix}: 数据哈希变化",
        )
        require(
            sha256_file(source_path)
            == spec[
                "source_score_sha256"
            ],
            f"{prefix}: 源分数哈希变化",
        )

        values = np.load(
            source_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        require(
            values.shape
            == (spec["count"],),
            f"{prefix}: 源分数形状错误",
        )
        require(
            bool(np.all(np.isfinite(
                values
            ))),
            f"{prefix}: 源分数非有限",
        )

    for key in [
        "score_manifest",
        "reference_result",
    ]:
        path = (
            ROOT
            / config["base_reward_model"][key]
        )
        require(
            path.exists(),
            f"缺少基础清单：{path}",
        )

    result_manifest = (
        ROOT
        / config["outputs"][
            "result_manifest"
        ]
    )
    require(
        not result_manifest.exists(),
        "最终结果已经存在，拒绝覆盖",
    )

    if require_frozen:
        verify_protocol_tag()

    print("Sigma：", config[
        "degradation"
    ]["sigma_grid"])
    print("变体：", len(variants))
    print(
        "CONTROLLED_DEGRADATION_"
        "PROTOCOL_READY"
    )

    return config


def read_group_metadata(path, expected):
    group_key = "question_uid"
    candidate_key = "candidate_index"

    groups = defaultdict(list)
    candidate_identities = set()
    rows = 0

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for row_index, line in enumerate(file):
            if not line.strip():
                continue

            row = json.loads(line)
            rows += 1

            require(
                group_key in row,
                f"{path}: 缺少 {group_key}",
            )
            require(
                candidate_key in row,
                f"{path}: 缺少 {candidate_key}",
            )

            uid = str(row[group_key])
            candidate_index = int(
                row[candidate_key]
            )
            identity = (
                uid,
                candidate_index,
            )

            require(
                identity
                not in candidate_identities,
                f"{path}: 候选身份重复 "
                f"{identity}",
            )
            candidate_identities.add(
                identity
            )
            groups[uid].append((
                candidate_index,
                row_index,
            ))

    require(
        rows == expected,
        f"{path}: 元数据行数错误",
    )

    ordered = {}

    for uid, items in groups.items():
        items.sort(
            key=lambda item: item[0]
        )
        ordered[uid] = [
            row_index
            for _, row_index in items
        ]

    return ordered


def deterministic_noise(
    variant,
    prefix,
    uid,
    count,
):
    key = (
        "controlled_reward_degradation_v1"
        f"|{variant['seed']}"
        f"|{prefix}"
        f"|{uid}"
    )
    digest = hashlib.sha256(
        key.encode("utf-8")
    ).digest()
    seed = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )
    rng = np.random.default_rng(seed)
    return rng.standard_normal(count)


def degraded_scores(
    original,
    groups,
    variant,
    prefix,
):
    result = np.empty(
        len(original),
        dtype=np.float32,
    )
    sigma = float(variant["sigma"])
    ranking_preserved = True
    zero_variance_questions = 0

    for uid, positions in groups.items():
        indices = np.asarray(
            positions,
            dtype=np.int64,
        )
        values = np.asarray(
            original[indices],
            dtype=np.float64,
        )

        mean = float(np.mean(values))
        std = float(np.std(values))

        if std <= 1e-12:
            standardized = np.zeros_like(
                values
            )
            zero_variance_questions += 1
        else:
            standardized = (
                values - mean
            ) / std

        if sigma == 0.0:
            degraded = standardized
            ranking_preserved = (
                ranking_preserved
                and int(np.argmax(values))
                == int(np.argmax(degraded))
            )
        else:
            noise = deterministic_noise(
                variant,
                prefix,
                uid,
                len(values),
            )
            degraded = (
                standardized
                + sigma * noise
            )

        result[indices] = degraded.astype(
            np.float32
        )

    require(
        bool(np.all(np.isfinite(result))),
        f"{variant['id']}/{prefix}: "
        "产生非有限分数",
    )

    if sigma == 0.0:
        require(
            ranking_preserved,
            f"{prefix}: 零噪声未保持排序",
        )

    return result, {
        "questions": len(groups),
        "zero_variance_questions": (
            zero_variance_questions
        ),
        "ranking_preserved": (
            ranking_preserved
            if sigma == 0.0
            else None
        ),
        "statistics": {
            "min": float(np.min(result)),
            "mean": float(np.mean(
                result,
                dtype=np.float64,
            )),
            "std": float(np.std(
                result,
                dtype=np.float64,
            )),
            "max": float(np.max(result)),
        },
    }


def prepare_scores(config):
    score_manifest_path = (
        ROOT
        / config["outputs"][
            "score_manifest"
        ]
    )

    metadata = {}
    source_scores = {}

    for prefix, spec in (
        config["data"].items()
    ):
        metadata[prefix] = (
            read_group_metadata(
                ROOT / spec["file"],
                spec["count"],
            )
        )
        source_scores[prefix] = np.load(
            score_path(config, prefix),
            allow_pickle=False,
        ).astype(np.float32)

    variant_records = []

    for number, variant in enumerate(
        config["degradation"]["variants"],
        start=1,
    ):
        root = variant_root(
            config,
            variant,
        )
        split_records = {}

        print(
            f"[{number:02d}/26] "
            f"{variant['id']}"
        )

        for prefix, spec in (
            config["data"].items()
        ):
            values, diagnostics = (
                degraded_scores(
                    source_scores[prefix],
                    metadata[prefix],
                    variant,
                    prefix,
                )
            )

            target = (
                root
                / f"{prefix}.scores_f32.npy"
            )
            atomic_npy(target, values)

            split_records[prefix] = {
                "file": str(
                    target.relative_to(ROOT)
                ),
                "sha256": sha256_file(
                    target
                ),
                "count": len(values),
                **diagnostics,
            }

        variant_manifest = {
            "version": (
                "controlled_reward_"
                "degradation_score_variant_v1"
            ),
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "variant": variant,
            "base_reward_model": config[
                "base_reward_model"
            ]["name"],
            "labels_used": False,
            "score_root": str(
                root.relative_to(ROOT)
            ),
            "splits": split_records,
        }

        manifest_path = (
            root / "score_manifest.json"
        )
        atomic_json(
            manifest_path,
            variant_manifest,
        )

        variant_records.append({
            **variant,
            "score_root": str(
                root.relative_to(ROOT)
            ),
            "score_manifest": str(
                manifest_path.relative_to(
                    ROOT
                )
            ),
            "score_manifest_sha256": (
                sha256_file(manifest_path)
            ),
        })

    score_manifest = {
        "version": (
            "controlled_reward_degradation_"
            "scores_v1"
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "configuration": str(
            CONFIG_PATH.relative_to(ROOT)
        ),
        "configuration_sha256": (
            sha256_file(CONFIG_PATH)
        ),
        "base_reward_model": config[
            "base_reward_model"
        ]["name"],
        "labels_used": False,
        "variants": variant_records,
    }
    atomic_json(
        score_manifest_path,
        score_manifest,
    )

    print("退化分数清单：", score_manifest_path)
    return score_manifest


def load_or_prepare_scores(config):
    path = (
        ROOT
        / config["outputs"][
            "score_manifest"
        ]
    )

    if not path.exists():
        return prepare_scores(config)

    manifest = json.loads(
        path.read_text(encoding="utf-8")
    )

    require(
        manifest["configuration_sha256"]
        == sha256_file(CONFIG_PATH),
        "已有退化分数配置哈希不一致",
    )
    require(
        len(manifest["variants"]) == 26,
        "已有退化分数变体数量错误",
    )

    for variant in manifest["variants"]:
        root = ROOT / variant["score_root"]
        manifest_path = (
            ROOT / variant["score_manifest"]
        )

        require(
            manifest_path.exists(),
            f"缺少变体清单：{manifest_path}",
        )
        require(
            sha256_file(manifest_path)
            == variant[
                "score_manifest_sha256"
            ],
            f"变体清单哈希错误："
            f"{variant['id']}",
        )

        reproduction.validate_score_files(
            root
        )

    print("复用已审计的退化分数。")
    return manifest


def run_variant(variant):
    model_key = (
        "controlled_" + variant["id"]
    )
    result_path = (
        ROOT
        / "data/manifests/"
        / (
            "multi_reward_reproduction_"
            f"{model_key}_v1.json"
        )
    )

    if result_path.exists():
        existing = json.loads(
            result_path.read_text(
                encoding="utf-8"
            )
        )
        require(
            existing["score_root"]
            == variant["score_root"],
            f"{variant['id']}: "
            "已有结果分数路径错误",
        )
        print(
            f"{variant['id']}: "
            "复用已有评价。"
        )
        return result_path

    reproduction.MODEL_SPECS = {
        model_key: {
            "name": (
                "Skywork-Reward-V2-Qwen3-4B"
                f" degraded sigma="
                f"{variant['sigma']}"
                f" seed={variant['seed']}"
            ),
            "score_root": (
                ROOT / variant["score_root"]
            ),
            "score_manifest": (
                ROOT
                / variant["score_manifest"]
            ),
        },
    }

    previous_argv = sys.argv[:]

    try:
        sys.argv = [
            str(SCRIPT_PATH),
            "--model",
            model_key,
        ]
        reproduction.main()
    finally:
        sys.argv = previous_argv

    require(
        result_path.exists(),
        f"{variant['id']}: 未产生结果",
    )
    return result_path


def correlation_summary(points):
    x = np.asarray([
        item["raw_top1"]
        for item in points
    ], dtype=np.float64)
    y = np.asarray([
        item["top1_delta"]
        for item in points
    ], dtype=np.float64)

    return {
        "points": len(points),
        "pearson": float(
            pearsonr(x, y).statistic
        ),
        "spearman": float(
            spearmanr(x, y).statistic
        ),
        "linear_slope": float(
            np.polyfit(x, y, 1)[0]
        ),
        "expected_direction": (
            "negative"
        ),
    }


def aggregate_results(
    config,
    score_manifest,
    result_paths,
    started,
):
    rows = []

    for variant, result_path in zip(
        score_manifest["variants"],
        result_paths,
    ):
        result = json.loads(
            result_path.read_text(
                encoding="utf-8"
            )
        )
        macro = result["test_macro"]

        rows.append({
            "id": variant["id"],
            "sigma": float(
                variant["sigma"]
            ),
            "seed": int(variant["seed"]),
            "score_root": (
                variant["score_root"]
            ),
            "score_manifest_sha256": (
                variant[
                    "score_manifest_sha256"
                ]
            ),
            "result_file": str(
                result_path.relative_to(ROOT)
            ),
            "result_sha256": sha256_file(
                result_path
            ),
            "raw_top1": macro["raw_top1"],
            "method_top1": (
                macro["method_top1"]
            ),
            "top1_delta": (
                macro["top1_delta"]
            ),
            "raw_pair": macro["raw_pair"],
            "method_pair": (
                macro["method_pair"]
            ),
            "pair_delta": (
                macro["pair_delta"]
            ),
            "damage_rate": (
                macro["damage_rate"]
            ),
            "correction_rate": (
                macro["correction_rate"]
            ),
            "best_at_4_delta": (
                macro["budget"]["k4"][
                    "best_at_k_delta"
                ]
            ),
            "best_at_8_delta": (
                macro["budget"]["k8"][
                    "best_at_k_delta"
                ]
            ),
        })

    reference_path = (
        ROOT
        / config["base_reward_model"][
            "reference_result"
        ]
    )
    reference = json.loads(
        reference_path.read_text(
            encoding="utf-8"
        )
    )
    reference_macro = reference[
        "test_macro"
    ]
    zero = next(
        item
        for item in rows
        if item["sigma"] == 0.0
    )

    zero_reproduction = {
        key: {
            "reference": float(
                reference_macro[key]
            ),
            "controlled_zero": float(
                zero[key]
            ),
            "absolute_error": abs(
                float(reference_macro[key])
                - float(zero[key])
            ),
        }
        for key in [
            "raw_top1",
            "method_top1",
            "top1_delta",
            "raw_pair",
            "method_pair",
            "pair_delta",
            "damage_rate",
            "correction_rate",
        ]
    }

    require(
        all(
            item["absolute_error"] < 1e-10
            for item in (
                zero_reproduction.values()
            )
        ),
        "Sigma=0 未复现基础 RM 结果",
    )

    by_sigma = []

    for sigma in config[
        "degradation"
    ]["sigma_grid"]:
        points = [
            item
            for item in rows
            if item["sigma"]
            == float(sigma)
        ]

        record = {
            "sigma": float(sigma),
            "replicates": len(points),
        }

        for key in [
            "raw_top1",
            "method_top1",
            "top1_delta",
            "pair_delta",
            "damage_rate",
            "correction_rate",
            "best_at_4_delta",
            "best_at_8_delta",
        ]:
            values = np.asarray([
                item[key]
                for item in points
            ], dtype=np.float64)
            record[key] = {
                "mean": float(
                    np.mean(values)
                ),
                "std": float(
                    np.std(values)
                ),
                "min": float(
                    np.min(values)
                ),
                "max": float(
                    np.max(values)
                ),
            }

        by_sigma.append(record)

    level_points = [{
        "raw_top1": item[
            "raw_top1"
        ]["mean"],
        "top1_delta": item[
            "top1_delta"
        ]["mean"],
    } for item in by_sigma]

    within_seed = {}

    zero_point = next(
        item
        for item in rows
        if item["sigma"] == 0.0
    )

    for seed in config[
        "degradation"
    ]["nonzero_sigma_seeds"]:
        points = [zero_point] + [
            item
            for item in rows
            if item["seed"] == seed
        ]
        points.sort(
            key=lambda item: item["sigma"]
        )
        within_seed[str(seed)] = (
            correlation_summary(points)
        )

    raw_means = [
        item["raw_top1"]["mean"]
        for item in by_sigma
    ]
    gain_means = [
        item["top1_delta"]["mean"]
        for item in by_sigma
    ]

    result = {
        "version": (
            "controlled_reward_degradation_"
            "results_v1"
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "analysis_type": config[
            "analysis_type"
        ],
        "configuration": str(
            CONFIG_PATH.relative_to(ROOT)
        ),
        "configuration_sha256": (
            sha256_file(CONFIG_PATH)
        ),
        "protocol_tag": PROTOCOL_TAG,
        "hypothesis": config[
            "hypothesis"
        ],
        "base_reward_model": config[
            "base_reward_model"
        ]["name"],
        "degradation_labels_used": False,
        "test_labels_used_for_selection": (
            False
        ),
        "test_labels_used_for_final_metrics": (
            True
        ),
        "confirmatory_new_test": False,
        "variants": rows,
        "zero_noise_reproduction": (
            zero_reproduction
        ),
        "by_sigma": by_sigma,
        "strength_gain_trend": {
            "all_variants": (
                correlation_summary(rows)
            ),
            "level_means": (
                correlation_summary(
                    level_points
                )
            ),
            "within_seed": within_seed,
            "raw_strength_strictly_decreases": (
                all(
                    right < left
                    for left, right
                    in zip(
                        raw_means,
                        raw_means[1:],
                    )
                )
            ),
            "gain_strictly_increases": (
                all(
                    right > left
                    for left, right
                    in zip(
                        gain_means,
                        gain_means[1:],
                    )
                )
            ),
        },
        "elapsed_seconds": round(
            time.time() - started,
            3,
        ),
        "interpretation_scope": (
            "Post-hoc controlled mechanism "
            "analysis on previously evaluated "
            "mathematical benchmarks."
        ),
    }

    output_path = (
        ROOT
        / config["outputs"][
            "result_manifest"
        ]
    )
    atomic_json(output_path, result)

    return output_path, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )
    args = parser.parse_args()

    config = preflight(
        require_frozen=(
            not args.preflight_only
        )
    )

    if args.preflight_only:
        print(
            "Preflight-only 完成；"
            "没有生成退化分数，"
            "没有运行 Test。"
        )
        return

    started = time.time()

    score_manifest = (
        load_or_prepare_scores(config)
    )

    result_paths = []

    for number, variant in enumerate(
        score_manifest["variants"],
        start=1,
    ):
        print()
        print("#" * 76)
        print(
            f"===== 评价变体 "
            f"{number}/26："
            f"{variant['id']} ====="
        )
        result_paths.append(
            run_variant(variant)
        )

    output_path, result = (
        aggregate_results(
            config,
            score_manifest,
            result_paths,
            started,
        )
    )

    print()
    print("=" * 76)
    print("===== 受控退化最终趋势 =====")
    print(json.dumps(
        {
            "zero_noise_reproduction": (
                result[
                    "zero_noise_reproduction"
                ]
            ),
            "by_sigma": result["by_sigma"],
            "strength_gain_trend": result[
                "strength_gain_trend"
            ],
        },
        ensure_ascii=False,
        indent=2,
    ))
    print()
    print(
        "CONTROLLED_REWARD_DEGRADATION_COMPLETE"
    )
    print("结果：", output_path)


if __name__ == "__main__":
    main()

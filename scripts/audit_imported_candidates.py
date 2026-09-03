from pathlib import Path
from collections import Counter
import json

ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT / "data/imported/"
    "SGLDSV_CANDIDATE_DATASETS_20260813"
)

FILES = [
    BASE / "gsm8k_gen/train.jsonl",
    BASE / "gsm8k_gen/val.jsonl",
    BASE / "gsm8k_gen/test.jsonl",
    BASE / "gsm8k_gen/train_pairs.jsonl",
    BASE / "gsm8k_gen/val_pairs.jsonl",
    BASE / "math_gen7b/train.jsonl",
    BASE / "math_gen7b/val.jsonl",
    BASE / "math_gen7b/test.jsonl",
    BASE / "math_gen7b/train_pairs.jsonl",
    BASE / "math_gen7b/val_pairs.jsonl",
    BASE / "svamp_gen/test.jsonl",
]

LABEL_KEYS = [
    "label", "correct", "is_correct",
    "final_answer_correct", "target", "preference",
]


def preview(value):
    if isinstance(value, str):
        value = value.replace("\n", "\\n")
        return repr(value[:240])
    if isinstance(value, list):
        if not value:
            return "list(len=0)"
        first = value[0]
        if isinstance(first, dict):
            return (
                f"list(len={len(value)}, "
                f"item_keys={sorted(first.keys())})"
            )
        return f"list(len={len(value)}, first={repr(first)[:120]})"
    if isinstance(value, dict):
        return f"dict(keys={sorted(value.keys())})"
    return repr(value)


for path in FILES:
    print("\n" + "=" * 80)
    print(path.relative_to(BASE))

    if not path.exists():
        print("文件不存在")
        continue

    count = 0
    invalid = 0
    first = None
    label_counts = {key: Counter() for key in LABEL_KEYS}
    list_lengths = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except Exception:
                invalid += 1
                continue

            count += 1

            if first is None:
                first = row

            if isinstance(row, dict):
                for key in LABEL_KEYS:
                    if key in row:
                        label_counts[key][str(row[key])] += 1

                for key, value in row.items():
                    if isinstance(value, list):
                        list_lengths.setdefault(key, []).append(len(value))

    print("有效行数：", count)
    print("无效行数：", invalid)

    if isinstance(first, dict):
        print("字段：", sorted(first.keys()))
        print("第一条数据预览：")
        for key, value in first.items():
            print(f"  {key}: {preview(value)}")

    for key, counter in label_counts.items():
        if counter:
            print(f"{key} 分布：", dict(counter))

    for key, lengths in list_lengths.items():
        if lengths:
            print(
                f"{key} 列表长度："
                f"最小={min(lengths)}, "
                f"最大={max(lengths)}, "
                f"平均={sum(lengths)/len(lengths):.2f}"
            )

print("\n" + "=" * 80)
print("统计文件：")

for path in sorted(BASE.glob("*/*stats*.json")):
    print("\n", path.relative_to(BASE))
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(content, ensure_ascii=False, indent=2)[:3000])
    except Exception as error:
        print("读取失败：", repr(error))

from pathlib import Path
import json
import os

from modelscope import snapshot_download

project = Path(__file__).resolve().parents[1]
old_root = Path("/root/autodl-tmp/vfrm/models")
new_root = project / "models/generator"

models = {
    "Qwen2-1.5B": "Qwen/Qwen2-1.5B",
    "Qwen2-7B": "Qwen/Qwen2-7B",
}

new_root.mkdir(parents=True, exist_ok=True)

for name, model_id in models.items():
    old_path = old_root / name
    new_path = new_root / name

    print("\n" + "=" * 72)
    print(name)

    if new_path.exists() and (
        new_path / "config.json"
    ).exists():
        print("项目目录中已经存在：", new_path)

    elif old_path.exists() and (
        old_path / "config.json"
    ).exists():
        print("发现原始模型：", old_path)
        if not new_path.exists():
            new_path.symlink_to(
                old_path,
                target_is_directory=True,
            )
            print("已建立软链接：", new_path)

    else:
        print("从魔塔下载：", model_id)
        snapshot_download(
            model_id=model_id,
            local_dir=str(new_path),
        )
        print("下载完成：", new_path)

    config_path = new_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    print("model_type:", config.get("model_type"))
    print(
        "architectures:",
        config.get("architectures"),
    )
    print(
        "hidden_size:",
        config.get("hidden_size"),
    )
    print(
        "num_hidden_layers:",
        config.get("num_hidden_layers"),
    )
    print(
        "max_position_embeddings:",
        config.get("max_position_embeddings"),
    )

    weight_files = sorted(
        new_path.glob("*.safetensors")
    )
    total = sum(
        path.stat().st_size
        for path in weight_files
    )
    print("权重文件数：", len(weight_files))
    print("权重 GB：", round(
        total / (1024 ** 3),
        3,
    ))

print("\n两个原始生成模型准备完成。")

# 运行主训练脚本 trainer.py
# 单卡（或调试）运行（推荐，按包方式运行，确保在项目根目录）：
python -m src.train.trainer \
    --config configs/model/freqdetr_base.yaml \
    --data configs/dataset/data.yaml

# 多卡（建议在 Linux 服务器上使用）示例：
# 使用 torchrun 启动（PyTorch >=1.10）：
# 4 GPUs 单节点示例（推荐，使用顶层 launcher 以避免导入/路径问题）：
# 推荐命令（更稳健）：
torchrun --nproc_per_node=4 python run_trainer.py \
    --config configs/model/freqdetr_base.yaml \
    --data configs/dataset/data.yaml

# 注意：
# - 在 Linux GPU 服务器上使用 NCCL 后端以获得最佳性能（通常已捆绑在 PyTorch+CUDA 安装中）。
# - 确保安装的 `torch` 与 CUDA 版本匹配，服务器需要正确配置 NVIDIA 驱动与 NVIDIA-Container-Toolkit（如使用容器）。

# 恢复训练示例：
# 当训练被中断或想继续训练时，从保存的 checkpoint 恢复：
torchrun --nproc_per_node=4 src/train/trainer.py \
    --config configs/model/freqdetr_base.yaml \
    --data configs/dataset/data.yaml \
    --resume ./checkpoints/freqdetr_best.pt


git status
git add .
git commit -m "version"    
git push

uv init
uv add -r requirements.txt
source .venv/bin/activate

## Quick start — minimal and portable

Notes: run commands from the project root. Use absolute dataset paths on servers.

1) Create / activate environment (one of):

Conda (recommended):

```bash
# create and activate (example)
# scripts/setup_env.sh --conda --name frepdetr
conda activate frepdetr
pip install -r requirements.txt
```

venv:

```bash
# create and activate
# scripts/setup_env.sh --venv --venv-path .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Single-GPU / debug (local)

```bash
python -m src.train.trainer \
  --config configs/model/freqdetr_base.yaml \
  --data configs/dataset/data.yaml
```

3) Multi-GPU (server) — use torchrun (PyTorch >=1.10)

```bash
torchrun --nproc_per_node=NUM python run_trainer.py \
  --config configs/model/freqdetr_base.yaml \
  --data /absolute/path/to/Maize/data.yaml
```

Replace NUM with your GPU count. Prefer absolute paths for `--data` on multi-node setups.

4) Resume from checkpoint

```bash
# add --resume with a saved checkpoint
python -m src.train.trainer \
  --config configs/model/freqdetr_base.yaml \
  --data configs/dataset/data.yaml \
  --resume ./checkpoints/freqdetr_best.pt
```

5) Common troubleshooting

- If you see FileNotFoundError for annotations, verify the COCO file exists:

```bash
ls -l /absolute/path/to/Maize/train/_annotations.coco.json
```

- On macOS, if `pycocotools` fails to build:

```bash
xcode-select --install
pip install cython
pip install pycocotools-binary
```

- On servers: ensure CUDA + drivers and NCCL are configured and `torch` matches your CUDA version.

6) Small git / workflow notes

Use normal git flow for checkpoints/versions (example):

```bash
git add . && git commit -m "train: <notes>" && git push
```

That's it — this file contains only the minimal, repeatable run commands. For more details see `README.md` or the configs under `configs/`.

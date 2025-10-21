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

## 环境与部署快速指南

1) 推荐快速创建环境（使用脚本）：

```bash
# 使用 conda 创建环境并安装依赖
scripts/setup_env.sh --conda --name frepdetr

# 或使用 venv
scripts/setup_env.sh --venv --venv-path .venv
source .venv/bin/activate
```

2) 数据路径建议
- 请在 `configs/dataset/data.yaml` 中使用绝对路径（例如 `/data/datasets/Maize`），或确保在启动时当前工作目录为项目根。
- 本项目在启动时会检查数据路径并在任何 rank 上不可见时优雅退出。

3) 常见问题
- 如果出现 `FileNotFoundError`，请先运行：
    ```bash
    ls -l /absolute/path/to/Maize/train/_annotations.coco.json
    ```
- 如果 `pycocotools` 安装失败（尤其在 macOS），请尝试：
    ```bash
    xcode-select --install
    pip install cython
    pip install pycocotools-binary
    ```

4) 在服务器上以 torchrun 启动（示例）

```bash
# 推荐：使用 launcher 来避免相对导入问题
torchrun --nproc_per_node=4 python run_trainer.py \
        --config configs/model/freqdetr_base.yaml \
        --data /absolute/path/to/Maize/data.yaml
```

# 运行主训练脚本 trainer.py
# 单卡（或调试）运行：
python src/train/trainer.py \
    --config configs/model/freqdetr_base.yaml \
    --data configs/dataset/maize.yaml

# 多卡（建议在 Linux 服务器上使用）示例：
# 使用 torchrun 启动（PyTorch >=1.10）：
# 4 GPUs 单节点示例：
# torchrun --nproc_per_node=4 src/train/trainer.py \
#     --config configs/model/freqdetr_base.yaml \
#     --data configs/dataset/maize.yaml

# 注意：
# - 在 Linux GPU 服务器上使用 NCCL 后端以获得最佳性能（通常已捆绑在 PyTorch+CUDA 安装中）。
# - 确保安装的 `torch` 与 CUDA 版本匹配，服务器需要正确配置 NVIDIA 驱动与 NVIDIA-Container-Toolkit（如使用容器）。

# 恢复训练示例：
# 当训练被中断或想继续训练时，从保存的 checkpoint 恢复：
# torchrun --nproc_per_node=4 src/train/trainer.py \
#     --config configs/model/freqdetr_base.yaml \
#     --data configs/dataset/maize.yaml \
#     --resume ./checkpoints/freqdetr_best.pt
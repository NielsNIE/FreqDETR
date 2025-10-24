总览（高层）
模型名：FreqDETR（项目内类 FreqDETR）
类型：DETR-style（transformer-based）单阶段目标检测器，使用 CNN backbone 提取特征后经 transformer encoder-decoder，decoder 输出 Nq queries 的特征向量，经 head 输出分类 logits 与边界框（cx,cy,w,h, 归一化到 [0,1]）。
关键实现文件：
模型：freqdetr.py
Backbone：cnn_backbone.py（ResNet-18 → conv 1x1 投影到 256）
Transformer encoder：encoder.py
Transformer decoder：decoder.py
Head：freqdetr_head.py
训练与评估流程：trainer.py
数据：datamodule.py（COCO 风格读取、Roboflow 兼容）
配置：freqdetr_base.yaml 与 data.yaml
架构细节（组件级）
Backbone — ResNet18Backbone
实现：基于 torchvision 的 ResNet-18（tv.resnet18）；提取 C5 特征（layer4 输出，通道 512）。
输出：通过 1x1 conv 将 512 → out_channels（默认 256），输出特征图大小约为输入 HxW 的 1/32（输入图像默认 640 → 特征 map 大小 20x20）。
可配置项（来自 config）：
out_channels（默认 256）
pretrained（是否加载预训练权重）
freeze_stages（冻结前若干 stage 的权重以做微调）
Positional Encoding / Transformer Encoder
TransformerEncoder 将 [B, C, H, W] 平铺为 [B, HW, C] 并通过 PyTorch 的 nn.TransformerEncoder（batch_first=True）进行自注意力编码。
Positional encoding 在文件内有 PositionalEncoding2D 的简易实现示例，但在 TransformerEncoder 中 self.pos = None（注：实际代码目前并没有激活 PE，采用 identity 投影）。也就是说，当前实现可能未显式加入位置编码（可视为一个简化的实现）。
参数（来自 config）：
d_model（256）
nheads（8）
enc_layers（3）
dim_feedforward（1024）
dropout（0.1）
Transformer Decoder
TransformerDecoder 使用 learnable queries（nn.Embedding(num_queries, d_model)）初始化 Nq queries（默认 100）。
使用 PyTorch 的 nn.TransformerDecoder 叠加若干 TransformerDecoderLayer（dec_layers=3）。
Decoder 的输出为 [B, Nq, C]，代表 Nq 个 query 对应的特征表示。
Head — FreqDetrHead
分类分支：nn.Linear(d_model, num_classes) → 输出 logits，默认 num_classes=1（可能是单类别的检测任务）。
回归分支：2 层 MLP（d_model → d_model → 4），末端 Sigmoid，将输出限制在 [0,1]，表示边框的 cx, cy, w, h（均为相对于图像宽高归一化的值）。
输出结构：字典 {"logits": [B,Nq,Cls], "boxes": [B,Nq,4]}。

数据处理与格式（数据流）
数据格式
支持 COCO 风格 JSON（.json），并兼容 Roboflow 导出的 <split>/_annotations.coco.json 风格。
配置文件：data.yaml 指定：
root（数据根）
img_dir_train / img_dir_val / img_dir_test
ann_train / ann_val / ann_test（相对路径或 None，若 None 则自动在 img_dir 下寻找 *.json）
classes（若省略，从 COCO JSON 中读取类别并做连续映射）
img_size（训练/推理尺寸，默认 640）
数据加载（COCODataset）
读取 COCO 注释，通过 pycocotools.COCO 进行解析。
将 COCO 的 bbox [x,y,w,h]（像素坐标）转换为归一化的 [cx,cy,w,h]：
cx = x + w/2, cy = y + h/2；再除以 W/H 得到 0..1 范围
图像增强/预处理通过 build_transforms（未展开在本次读取，但典型地包含缩放、裁剪、归一化、bboxes 转换）。
返回：image tensor（C,H,W）float（0..1）、target dict 包含 boxes (tensor Nx4)、labels (tensor N)。
批处理（collate_fn）会把 images stack 为 [B,C,H,W]，targets 保持为列表以保留变长。
DataLoader
build_dataloaders 根据是否处于分布式（检测 torch.distributed.is_initialized()）来决定是否使用 DistributedSampler。
train_loader 的 batch_size 来自 model config (cfg["batch_size"])。
val_loader batch_size = max(1, batch_size//2)

损失、匹配与评估（训练细节）
匹配策略（简化）
代码中实现了 match_by_greedy，一个基于 IoU 的贪心匹配器（非 Hungarian），操作：
将预测框和 GT box 转换为 xyxy（通过 box_cxcywh_to_xyxy），计算 generalized IoU 矩阵 [P,G]。
重复选择当前最大 IoU 对（pi, gi），并将所选行列置 -1 避免重复选择，直到 P 或 G 用尽。
这是一个简单/近似的匹配方法，相对于 Hungarian 可能不保证全局最优。
分类损失
在训练循环中构造 cls_targets（与 logits 同 shape [B,Nq,Cls]），将匹配到的 queries 对应位置赋 1（类别为 matched GT 的 label），其余为 0。
使用自定义 sigmoid_focal_loss（基于 BCEWithLogits + focal term），参数 alpha=0.25, gamma=2.0（参考 RetinaNet 的 focal loss）。
在 config 中 head.cls_loss 有字段 "focal" 或 "ce"，但训练代码固定使用 focal loss 的实现（若要切换需要代码修改）。
回归损失
对每个匹配对计算：
L1 loss（平均 L1）
GIoU loss：使用 generalized_box_iou 的 diag 得到每对 GIoU，然后使用 1 - mean(diag) 作为损失分量。
最终 loss = cls_loss + bbox_loss_l1 * l1_loss + bbox_loss_giou * giou_loss
系数在 freqdetr_base.yaml 中：
bbox_loss_l1 = 5.0
bbox_loss_giou = 2.0
混合精度
trainer.py 支持 AMP（自动混合精度），受 cfg amp 控制（默认 true），并在 device 为 CUDA 时启用 torch.cuda.amp.GradScaler()。
优化器与调度
使用 AdamW 优化器（torch.optim.AdamW），学习率 lr 与权重衰减 weight_decay 从 cfg 获取（lr=2e-4, weight_decay=1e-4）。
config 中 optim.scheduler 标注为 "cosine"，但训练脚本中没有看到 scheduler 的显式创建/step（可能未实现或在别处）；要使用 scheduler 需补充实现。
评价指标
在 evaluate 中，使用 ap50_greedy（代码在 metrics.py，未展开）计算 F1@50、precision、recall，评估流程：
对预测 logits 取 sigmoid，得到每 query 的最大类别分数作为置信度；
每个 image 取前 K (K=min(100, Nq)) 个预测用于评估；
计算 ap50_greedy（基于 IoU 0.5 的 greedy matching metrics）
支持在分布式下通过 dist.all_reduce 聚合指标。

训练循环与分布式行为（trainer.py）
启动与分布式初始化
入口通过 run_trainer.py（位于仓库根）或 python -m src.train.trainer 调用。
Trainer 读取 args：--config, --data, --local_rank, --resume。
使用 os.environ["WORLD_SIZE"] 判断是否分布式（>1 则认为分布式）；如果分布式：
backend = "nccl"（当 cuda 可用）或 "gloo"
使用 dist.init_process_group
若 CUDA，可设置 torch.device(f"cuda:{local_rank}") 并 torch.cuda.set_device(device)
数据路径检查
在启动前 trainer 会检查 data_cfg 指定的 root、img_dir、annotation 文件是否存在（并尝试若干候选路径），若检查失败会在 rank 0 打印错误并在所有 rank 上退出（通过 dist.all_reduce 保证一致性）。
Model / DDP
构造模型：FreqDETR(cfg).to(device)，如果分布式则用 DistributedDataParallel（DDP），device_ids 根据 CUDA 可用性设置。
只有主进程（rank 0 或非分布式）写日志与保存 checkpoint（使用 TensorBoard SummaryWriter）。
Checkpoint 格式（保存/恢复）
保存：字典 ck = {"epoch": epoch, "model_state": state_dict, "optimizer_state": ..., "best_f1": best_f1, "scaler_state": ... (if amp)}
model_state：在 DDP 情况下使用 model.module.state_dict()，否则 model.state_dict()。
checkpoint 文件默认保存在 freqdetr_best.pt
恢复：若 --resume 指定且文件存在，加载 ck：
加载 model_state（处理 DDP 包装），尝试加载 optimizer_state、scaler_state，并将 start_epoch = ck.get("epoch", 1) + 1
打印恢复信息
训练周期
每 epoch：
若分布式且使用了 sampler，则会调用 train_loader.sampler.set_epoch(epoch) 以保证每 epoch shuffle 不同。
调用 train_one_epoch 计算训练 loss（在代码中返回平均 loss）。
调用 evaluate 计算 val 指标（f1, p, r）。
主进程记录 scalar 到 TensorBoard 并保存 checkpoint（若 f1 改善则更新 best）。
训练结束打印 best F1 与文件路径。

超参数（来自 configs）
（以 freqdetr_base.yaml 为准）

seed: 42
device: auto
img_size: 640
epochs: 10
batch_size: 2 (注意：非常小，可能为实验/调试用)
num_workers: 0
lr: 2e-4
weight_decay: 1e-4
warmup_epochs: 2 (trainer 未必使用)
amp: true
log_dir: ./runs/freqdetr_base
Backbone:

out_channels: 256
pretrained: true
freeze_stages: 0
Transformer:

d_model: 256
nheads: 8
enc_layers: 3
dec_layers: 3
dim_feedforward: 1024
num_queries: 100
dropout: 0.1
Head:

num_classes: 1
bbox_loss_l1: 5.0
bbox_loss_giou: 2.0
cls_loss: "focal"
Optim:

optimizer: "adamw"
scheduler: "cosine" (未在 trainer 中看到实现)

设计权衡与局限（工程/研究角度）
简易匹配（greedy）代替 Hungarian：
优点：实现简单，速度较快。
缺点：可能导致亚最优分配，特别是在高度重叠或复杂场景，影响训练收敛与性能。标准 DETR 使用 Hungarian（匈牙利算法）以实现全局最优配对。
Positional encoding 的缺失或简化：
DETR/transformer 的性能很依赖明确的位置编码。当前代码里 TransformerEncoder 有一个 PositionalEncoding2D 示例但未启用；缺少位置编码会影响模型对空间结构的学习，可能降低性能。
Head 的设计（Sigmoid + 归一化 boxes）：
采用 Sigmoid 将 cx,cy,w,h 限制到 [0,1]，简洁易实现；但相比直接回归像素或对数尺度宽高可能存在细节差异（需要根据数据分布调整）。
COCO 类别映射与缺失数据处理：
若图像没有标注（len(bboxes)=0），代码会回退到伪目标（一个超小 box 与 label 0）以避免 loader 报错。这是实用但不是最优的做法（可能带来负样本误导）。
Scheduler 未实现：
config 指定了 cosine scheduler，但 trainer 未实际创建或 step scheduler（需补全以使用学习率调度策略）。

可改进的点（工程优先级）
把位置编码明确加回 encoder（或 decoder），并暴露可选参数来切换不同 PE（sine/cosine / learned）。
用 Hungarian matcher（linear_sum_assignment）替换 greedy matcher，以提升训练稳定性与性能。
在 trainer.py 中实现学习率调度器（例如 CosineAnnealingWarmRestarts 或 Cosine w/ warmup）。
将 heavy imports 延迟到 main()，改进 help 显示与快速验证场景（你之前已经要求不改核心算法，但这是一个低风险的代码组织改进）。
增加更完整的损失项（例如类别的 focal/CE 可配置、no-object class 的权重等），并在 head 中支持多类别输出（目前 code 似乎是单类别 leaning）。
扩展 eval pipeline 使用标准 COCO API（mAP @0.5:0.95）以便与其它工作对比（目前实现的是 ap50_greedy 的简单指标）。

Checkpoint 与部署注意
Checkpoint 包含 model_state、optimizer_state、epoch、best_f1、（如 AMP）scaler_state。
在 DDP 模式下保存 model.module.state_dict()，恢复时处理 DDP 包装。
除了本地训练，建议在多节点场景下使用 NCCL 后端，并保证 --data 指向在每个节点都可访问的绝对路径（或使用共享文件系统）。

总结（一句话）
FreqDETR 是一个基于 ResNet-18 + Transformer encoder-decoder 的简化 DETR 实现，使用 learnable queries 输出 Nq 边框与分类，训练采用 focal 分类损失和 L1+GIoU 回归损失，工程上对可用性做了很多实用处理（Roboflow 兼容、数据路径检查、DDP 支持），但在匹配算法（greedy）、位置编码、和 LR scheduler 等方面做了简化，存在若干可提升的研究/工程改进点。
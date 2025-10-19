import os, yaml, time, math
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from ..utils.seed import set_seed
from ..models.freqdetr import FreqDETR
from ..data.datamodule import build_dataloaders
from ..utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from ..utils.metrics import ap50_greedy

def get_device(pref="auto"):
    """
    自动选择合适的设备：
    - CUDA 优先（如果有 NVIDIA GPU）
    - 否则使用 MPS（Mac GPU）
    - 最后回退到 CPU
    """
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("⚙️ Using Apple Metal (MPS) backend")
        return torch.device("mps")
    print("⚙️ Using CPU backend")
    return torch.device("cpu")

def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    # inputs: [B,Nq,Cls], targets: [B,Nq,Cls] (0/1)
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob*targets + (1-prob)*(1-targets)
    loss = ce_loss * ((1-p_t)**gamma)
    if alpha >= 0:
        alpha_t = alpha*targets + (1-alpha)*(1-targets)
        loss = alpha_t * loss
    if reduction=="mean": return loss.mean()
    if reduction=="sum":  return loss.sum()
    return loss

def match_by_greedy(pred_boxes, gt_boxes):
    # very simple greedy matcher per image
    # returns matched indices (pi, gi) lists
    if gt_boxes.numel()==0 or pred_boxes.numel()==0:
        return [], []
    pb = box_cxcywh_to_xyxy(pred_boxes)
    gb = box_cxcywh_to_xyxy(gt_boxes)
    ious = generalized_box_iou(pb, gb)  # [P,G]
    matched_pi, matched_gi = [], []
    used_g = set()
    for _ in range(min(len(pb), len(gb))):
        i = torch.argmax(ious).item()
        pi = i // ious.shape[1]; gi = i % ious.shape[1]
        if gi in used_g:  # already taken
            ious[pi,:] = -1
            continue
        matched_pi.append(pi); matched_gi.append(gi)
        used_g.add(gi)
        ious[pi,:] = -1
        ious[:,gi] = -1
    return matched_pi, matched_gi

def train_one_epoch(model, loader, optimizer, device, cfg):
    model.train()
    loss_meter = 0.0
    for images, targets, _ in loader:
        images = images.to(device)
        out = model(images)  # logits [B,Nq,C], boxes [B,Nq,4]
        logits, boxes = out["logits"], out["boxes"]

        B, Nq, C = logits.shape
        cls_targets = torch.zeros_like(logits)

        # 构造简单匹配与损失
        l1_losses, giou_losses = [], []
        for b in range(B):
            gt = targets[b]
            gt_boxes = gt["boxes"].to(device)
            gt_labels = gt["labels"].to(device)
            if gt_boxes.numel()==0: continue
            pi, gi = match_by_greedy(boxes[b], gt_boxes)  # 索引
            if len(pi)==0: continue
            # 分类：将匹配到的query置为1
            cls_targets[b, torch.tensor(pi, device=device), gt_labels[:len(pi)]] = 1.0
            # 回归：仅对匹配对计算 L1 + GIoU
            pb = boxes[b][torch.tensor(pi, device=device)]
            gb = gt_boxes[torch.tensor(gi, device=device)]
            l1_losses.append(torch.abs(pb - gb).mean())
            giou = generalized_box_iou(box_cxcywh_to_xyxy(pb), box_cxcywh_to_xyxy(gb)).diag()
            giou_losses.append(1.0 - giou.mean())

        # 分类损失（未匹配的query为负样本）
        cls_loss = sigmoid_focal_loss(logits, cls_targets, alpha=0.25, gamma=2.0)

        if len(l1_losses)>0:
            l1_loss = torch.stack(l1_losses).mean()
            giou_loss = torch.stack(giou_losses).mean()
        else:
            l1_loss = torch.tensor(0.0, device=device)
            giou_loss = torch.tensor(0.0, device=device)

        loss = cls_loss + cfg["head"]["bbox_loss_l1"]*l1_loss + cfg["head"]["bbox_loss_giou"]*giou_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
    return loss_meter/len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    f1s, precs, recs = [], [], []
    for images, targets, _ in loader:
        images = images.to(device)
        out = model(images)
        logits, boxes = out["logits"].sigmoid(), out["boxes"]
        scores = logits.max(dim=-1).values  # [B,Nq]
        for b in range(images.size(0)):
            gt_boxes = targets[b]["boxes"].to(device)
            pred_boxes = boxes[b]
            pred_scores = scores[b]
            # 取前 K（简单做法）
            K = min(100, pred_boxes.size(0))
            f1, p, r = ap50_greedy(pred_boxes[:K], pred_scores[:K], gt_boxes)
            f1s.append(f1); precs.append(p); recs.append(r)
    return float(torch.tensor(f1s).mean()), float(torch.tensor(precs).mean()), float(torch.tensor(recs).mean())

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--data", type=str, default="configs/dataset/corn_local.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    data_cfg = yaml.safe_load(open(args.data, "r"))

    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device","auto"))

    # dataloaders
    train_loader, val_loader = build_dataloaders(data_cfg, {"batch_size":cfg["batch_size"], "num_workers":cfg["num_workers"]})

    # model/opt
    model = FreqDETR(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    writer = SummaryWriter(log_dir=cfg.get("log_dir","./runs/freqdetr"))

    best_f1, best_path = 0.0, None
    for epoch in range(1, cfg["epochs"]+1):
        tl = train_one_epoch(model, train_loader, optimizer, device, cfg)
        f1, p, r = evaluate(model, val_loader, device)
        writer.add_scalar("loss/train", tl, epoch)
        writer.add_scalar("val/f1", f1, epoch)
        writer.add_scalar("val/precision", p, epoch)
        writer.add_scalar("val/recall", r, epoch)
        print(f"[Epoch {epoch}] loss={tl:.4f}  F1@50={f1:.3f}  P={p:.3f}  R={r:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            os.makedirs("./checkpoints", exist_ok=True)
            best_path = f"./checkpoints/freqdetr_best.pt"
            torch.save(model.state_dict(), best_path)
    print("Best F1:", best_f1, "saved:", best_path)

if __name__ == "__main__":
    main()
import os, yaml, time, math
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from ..utils.seed import set_seed
from ..models.freqdetr import FreqDETR
from ..data.datamodule import build_dataloaders
from ..utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from ..utils.metrics import ap50_greedy
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
    scaler = None
    use_amp = cfg.get("amp", False) and device.type == "cuda"
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    for images, targets, _ in loader:
        images = images.to(device)
        if use_amp:
            with torch.cuda.amp.autocast():
                out = model(images)  # logits [B,Nq,C], boxes [B,Nq,4]
        else:
            out = model(images)
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
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
    # aggregate across distributed processes if available
    f1_mean = torch.tensor(f1s).mean() if len(f1s)>0 else torch.tensor(0.0)
    p_mean = torch.tensor(precs).mean() if len(precs)>0 else torch.tensor(0.0)
    r_mean = torch.tensor(recs).mean() if len(recs)>0 else torch.tensor(0.0)
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            # create tensors on device
            f1_t = f1_mean.to(device)
            p_t = p_mean.to(device)
            r_t = r_mean.to(device)
            dist.all_reduce(f1_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(p_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(r_t, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
            f1_mean = (f1_t / world_size).cpu()
            p_mean = (p_t / world_size).cpu()
            r_mean = (r_t / world_size).cpu()
    except Exception:
        pass
    return float(f1_mean), float(p_mean), float(r_mean)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--data", type=str, default="configs/dataset/corn_local.yaml")
    # torchrun will set LOCAL_RANK and WORLD_SIZE environment variables
    ap.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    data_cfg = yaml.safe_load(open(args.data, "r"))
    set_seed(cfg.get("seed", 42))

    # Distributed setup
    local_rank = args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    if is_distributed:
        dist_backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=dist_backend)

    # device setup
    if is_distributed and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = get_device(cfg.get("device","auto"))

    def startup_info():
        import platform, sys
        torch_version = torch.__version__
        python_version = sys.version.split()[0]
        os_name = platform.system()
        cuda_available = torch.cuda.is_available()
        n_gpus = torch.cuda.device_count() if cuda_available else 0
        amp_enabled = cfg.get("amp", False) and device.type == "cuda"
        print("="*60)
        print("Starting training")
        print(f"OS: {os_name}")
        print(f"Python: {python_version}")
        print(f"PyTorch: {torch_version}")
        print(f"Device: {device} (type={device.type})")
        if device.type == "cuda":
            print(f"CUDA available: {cuda_available}, GPUs: {n_gpus}")
            if n_gpus>0:
                names = [torch.cuda.get_device_name(i) for i in range(min(n_gpus,8))]
                print(f"GPU names (first {min(n_gpus,8)}): {names}")
        print(f"Distributed: {is_distributed} (world_size={world_size if is_distributed else 1})")
        print(f"Local rank: {local_rank}")
        print(f"AMP enabled: {amp_enabled}")
        print(f"Batch size (per process): {cfg.get('batch_size')}")
        print(f"Num workers: {cfg.get('num_workers')}")
        print(f"Epochs: {cfg.get('epochs')}  LR: {cfg.get('lr')}")
        print(f"Train root: {data_cfg.get('root')}  Train img dir: {data_cfg.get('img_dir_train')}")
        # model parameter count
        try:
            model_params = sum(p.numel() for p in model.parameters())
            print(f"Model params: {model_params:,}")
        except Exception:
            pass
        print("="*60)

    # dataloaders (will use DistributedSampler if dist is initialized)
    # Normalize dataset root to absolute path to avoid cwd-related issues
    try:
        data_cfg["root"] = os.path.abspath(data_cfg.get("root", "."))
    except Exception:
        data_cfg["root"] = data_cfg.get("root", ".")

    # Verify dataset paths exist and are accessible. This helps catch errors
    # early and avoids one rank failing silently and killing the whole job.
    def _check_dataset_paths(dc):
        import glob
        errs = []
        root = dc.get("root")
        if not os.path.exists(root):
            errs.append(f"root path not found: {root}")

        # image directories
        for k in ["img_dir_train", "img_dir_val", "img_dir_test"]:
            d = dc.get(k, None)
            if d:
                p = os.path.join(root, d)
                if not os.path.isdir(p):
                    errs.append(f"image dir not found for {k}: {p}")

        # annotation files (if provided explicitly)
        for kf, kd in [("ann_train", "img_dir_train"), ("ann_val", "img_dir_val"), ("ann_test", "img_dir_test")]:
            ann = dc.get(kf, None)
            imgdir = dc.get(kd, None)
            if ann:
                p = os.path.join(root, ann)
                if not os.path.isfile(p):
                    # also try inside the image dir
                    tried = [p]
                    if imgdir:
                        alt = os.path.join(root, imgdir, os.path.basename(ann))
                        tried.append(alt)
                    # try common Roboflow name
                    if imgdir:
                        rf = os.path.join(root, imgdir, "_annotations.coco.json")
                        tried.append(rf)
                    found = any(os.path.isfile(x) for x in tried)
                    if not found:
                        errs.append(f"annotation file for {kf} not found. tried: {tried}")
            else:
                # if ann not provided, check whether imgdir has any .json annotations (roboflow style)
                if imgdir:
                    jlist = glob.glob(os.path.join(root, imgdir, "*.json"))
                    if len(jlist) == 0:
                        errs.append(f"no annotation JSON found under image dir {os.path.join(root, imgdir)}")

        return errs

    errs = _check_dataset_paths(data_cfg)
    ok = 1 if len(errs) == 0 else 0

    # If distributed, reduce across ranks to ensure consistent view and then exit all if any rank failed
    if is_distributed:
        try:
            # choose device for reduction: use CUDA tensor if backend==nccl and cuda available
            use_cuda_tensor = False
            try:
                bk = dist.get_backend()
                use_cuda_tensor = (bk == "nccl") and (device.type == "cuda")
            except Exception:
                use_cuda_tensor = (device.type == "cuda")

            red_dev = device if use_cuda_tensor else torch.device("cpu")
            t = torch.tensor(ok, device=red_dev)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            ok_global = int(t.item())
        except Exception:
            # fallback: assume not ok if exception occurs
            ok_global = ok
    else:
        ok_global = ok

    if ok_global == 0:
        # rank 0 prints details
        try:
            rank = dist.get_rank() if is_distributed else 0
        except Exception:
            rank = 0
        if rank == 0:
            print("ERROR: dataset path checks failed:")
            for e in errs:
                print("  -", e)
            print("Please fix your configs/dataset data paths (use absolute paths) or ensure the data is available on all nodes.")
        # synchronize and then exit on all ranks
        try:
            if is_distributed:
                dist.barrier()
        except Exception:
            pass
        import sys
        sys.exit(1)

    train_loader, val_loader = build_dataloaders(data_cfg, {"batch_size":cfg["batch_size"], "num_workers":cfg["num_workers"]})

    # model/opt
    model = FreqDETR(cfg).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if device.type=="cuda" else None)

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # Only rank 0 writes logs & checkpoints
    is_main = (not is_distributed) or (dist.get_rank() == 0)
    writer = SummaryWriter(log_dir=cfg.get("log_dir","./runs/freqdetr")) if is_main else None

    # print startup info on main process
    if is_main:
        startup_info()

    best_f1, best_path = 0.0, None
    start_epoch = 1
    scaler = None
    use_amp = cfg.get("amp", False) and device.type == "cuda"
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # optionally resume
    if is_main and args.resume:
        if os.path.isfile(args.resume):
            ck = torch.load(args.resume, map_location=device)
            # load model state (handle DDP wrapping)
            model_state = ck.get("model_state", None)
            if model_state is not None:
                if hasattr(model, "module"):
                    model.module.load_state_dict(model_state)
                else:
                    model.load_state_dict(model_state)
            if "optimizer_state" in ck:
                try:
                    optimizer.load_state_dict(ck["optimizer_state"])
                except Exception:
                    print("Warning: failed to load optimizer state fully")
            if use_amp and "scaler_state" in ck and scaler is not None:
                try:
                    scaler.load_state_dict(ck["scaler_state"])
                except Exception:
                    print("Warning: failed to load scaler state")
            start_epoch = ck.get("epoch", 1) + 1
            best_f1 = ck.get("best_f1", 0.0)
            print(f"Resumed from {args.resume}, starting at epoch {start_epoch}")
        else:
            print(f"Resume checkpoint {args.resume} not found")
    for epoch in range(start_epoch, cfg["epochs"]+1):
        # if distributed with sampler, set epoch for shuffling
        try:
            if is_distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
        except Exception:
            pass

        tl = train_one_epoch(model, train_loader, optimizer, device, cfg)
        f1, p, r = evaluate(model, val_loader, device)
        if is_main and writer is not None:
            writer.add_scalar("loss/train", tl, epoch)
            writer.add_scalar("val/f1", f1, epoch)
            writer.add_scalar("val/precision", p, epoch)
            writer.add_scalar("val/recall", r, epoch)
        if is_main:
            print(f"[Epoch {epoch}] loss={tl:.4f}  F1@50={f1:.3f}  P={p:.3f}  R={r:.3f}")
            if f1 > best_f1:
                best_f1 = f1
            if is_main:
                os.makedirs("./checkpoints", exist_ok=True)
                best_path = f"./checkpoints/freqdetr_best.pt"
                # save full checkpoint (model, optimizer, scaler, epoch, best_f1)
                sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                ck = {
                    "epoch": epoch,
                    "model_state": sd,
                    "optimizer_state": optimizer.state_dict(),
                    "best_f1": best_f1,
                }
                if use_amp and scaler is not None:
                    try:
                        ck["scaler_state"] = scaler.state_dict()
                    except Exception:
                        pass
                torch.save(ck, best_path)

    if is_main:
        print("Best F1:", best_f1, "saved:", best_path)

if __name__ == "__main__":
    main()
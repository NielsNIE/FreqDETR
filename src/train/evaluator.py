import yaml, torch
from ..models.freqdetr import FreqDETR
from ..data.datamodule import build_dataloaders
from ..utils.seed import set_seed
from .trainer import evaluate

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--data", type=str, default="configs/dataset/corn_local.yaml")
    ap.add_argument("--ckpt", type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    data_cfg = yaml.safe_load(open(args.data, "r"))
    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device","cuda") if torch.cuda.is_available() else "cpu")

    _, val_loader = build_dataloaders(data_cfg, {"batch_size":cfg["batch_size"], "num_workers":cfg["num_workers"]})
    model = FreqDETR(cfg).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd)
    f1, p, r = evaluate(model, val_loader, device)
    print(f"Eval => F1@50={f1:.3f}, P={p:.3f}, R={r:.3f}")

if __name__ == "__main__":
    main()
import os, cv2, torch, glob
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from .transforms import build_transforms

def coco_xywh_to_cxcywh_norm(box, H, W):
    # COCO: [x,y,w,h] (像素) -> 归一化 [cx,cy,w,h]
    x, y, w, h = box
    cx = x + w/2.0
    cy = y + h/2.0
    return [cx/W, cy/H, w/W, h/H]

def _resolve_ann_path(root, img_dir, ann_path_from_cfg):
    """
    解析标注路径，兼容两种情况：
    1) 集中式：annotations/instances_*.json
    2) Roboflow：<split>/_annotations.coco.json（或任意 *.json）
    """
    # 1) 如果配置里给了 ann 且存在，直接用
    if ann_path_from_cfg:
        p = os.path.join(root, ann_path_from_cfg)
        if os.path.isfile(p):
            return p

    # 2) 尝试在 img_dir 下寻找 Roboflow 风格
    img_dir_abs = os.path.join(root, img_dir)
    cand1 = os.path.join(img_dir_abs, "_annotations.coco.json")
    if os.path.isfile(cand1):
        return cand1

    # 3) 兜底：img_dir 下任意 *.json
    cands = sorted(glob.glob(os.path.join(img_dir_abs, "*.json")))
    if len(cands) > 0:
        return cands[0]

    raise FileNotFoundError(
        f"找不到标注JSON。尝试过：\n"
        f"  - {os.path.join(root, str(ann_path_from_cfg) if ann_path_from_cfg else '')}\n"
        f"  - {cand1}\n"
        f"  - {img_dir_abs}/*.json"
    )

class COCODataset(Dataset):
    def __init__(self, root, img_dir, ann_file=None, img_size=640, classes=None, split="train", catid2contig=None):
        """
        root: 数据根目录
        img_dir: 图像文件夹（相对 root）
        ann_file: 标注JSON（相对 root，或 None 让函数自动解析）
        """
        self.root = root
        self.img_dir = os.path.join(root, img_dir)
        self.ann_file = _resolve_ann_path(root, img_dir, ann_file)
        self.transforms = build_transforms(img_size, split)
        self.coco = COCO(self.ann_file)

        # 类别映射（COCO 的 cat_id -> 连续ID）
        if classes is not None:
            name2idx = {name: i for i, name in enumerate(classes)}
            self.catid2contig = {}
            for cid, cat in self.coco.cats.items():
                name = cat["name"]
                if name in name2idx:
                    self.catid2contig[cid] = name2idx[name]
            self.classes = classes
        else:
            cats = [self.coco.cats[cid]["name"] for cid in self.coco.cats]
            cats_sorted = sorted(list(set(cats)))
            self.classes = cats_sorted
            name2idx = {n:i for i,n in enumerate(self.classes)}
            self.catid2contig = {cid: name2idx[self.coco.cats[cid]["name"]] for cid in self.coco.cats}

        if catid2contig is not None:
            self.catid2contig = catid2contig

        self.img_ids = list(self.coco.imgs.keys())

    def __len__(self): return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        file_name = info["file_name"]
        img_path = os.path.join(self.img_dir, file_name)

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        bboxes_coco = []
        class_labels = []
        for ann in anns:
            if "bbox" not in ann: continue
            x,y,w,h = ann["bbox"]
            if w <= 1 or h <= 1: continue
            cls = self.catid2contig.get(ann["category_id"], None)
            if cls is None:  # 该类不在映射里，跳过
                continue
            bboxes_coco.append([x,y,w,h])
            class_labels.append(int(cls))

        if len(bboxes_coco) == 0:
            bboxes_coco, class_labels = [[0.0, 0.0, 1e-3, 1e-3]], [0]

        aug = self.transforms(image=img, bboxes=bboxes_coco, class_labels=class_labels)
        image = aug["image"].float()/255.0
        H2, W2 = image.shape[1], image.shape[2]

        boxes_norm = [coco_xywh_to_cxcywh_norm(b, H2, W2) for b in aug["bboxes"]]
        boxes = torch.tensor(boxes_norm, dtype=torch.float32)
        labels = torch.tensor(aug["class_labels"], dtype=torch.long)

        target = {"boxes": boxes, "labels": labels}
        return image, target, img_path

def collate_fn(batch):
    images, targets, paths = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, targets, paths

def build_dataloaders(cfg_data, train_val_cfg):
    root = cfg_data["root"]; size = cfg_data.get("img_size", 640)

    train_set = COCODataset(
        root=root,
        img_dir=cfg_data["img_dir_train"],
        ann_file=cfg_data.get("ann_train", None),
        img_size=size,
        classes=cfg_data.get("classes", None),
        split="train",
        catid2contig=None
    )
    val_set = COCODataset(
        root=root,
        img_dir=cfg_data["img_dir_val"],
        ann_file=cfg_data.get("ann_val", None),
        img_size=size,
        classes=train_set.classes,
        split="val",
        catid2contig=train_set.catid2contig
    )

    train_loader = DataLoader(
        train_set, batch_size=train_val_cfg["batch_size"], shuffle=True,
        num_workers=train_val_cfg["num_workers"], pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=max(1, train_val_cfg["batch_size"]//2), shuffle=False,
        num_workers=train_val_cfg["num_workers"], pin_memory=True, collate_fn=collate_fn
    )
    return train_loader, val_loader
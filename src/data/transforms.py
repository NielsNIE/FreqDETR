import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_transforms(img_size=640, split="train"):
    if split == "train":
        return A.Compose(
            [
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=(114,114,114)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.2),
                ToTensorV2()
            ],
            bbox_params=A.BboxParams(
                format="coco",              # 注意：COCO 像素坐标 [x,y,w,h]
                label_fields=["class_labels"],
                min_visibility=0.1,
            ),
        )
    else:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=(114,114,114)),
                ToTensorV2()
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["class_labels"],
                min_visibility=0.0,
            ),
        )
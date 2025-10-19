import torch, torch.nn as nn

class FreqDetrHead(nn.Module):
    def __init__(self, num_classes=1, d_model=256):
        super().__init__()
        self.classifier = nn.Linear(d_model, num_classes)  # 二分类/多分类：用 BCE/Focal 或 CE
        self.box_reg = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
            nn.Sigmoid()  # 直接输出 0..1 的 cx,cy,w,h
        )

    def forward(self, dec_out):  # [B,Nq,C]
        logits = self.classifier(dec_out)   # [B,Nq,Cls]
        boxes  = self.box_reg(dec_out)      # [B,Nq,4] (cx,cy,w,h)
        return {"logits": logits, "boxes": boxes}
import torch, torch.nn as nn
from ..backbones.cnn_backbone import ResNet18Backbone
from ..transformer.encoder import TransformerEncoder
from ..transformer.decoder import TransformerDecoder
from ..heads.freqdetr_head import FreqDetrHead

class FreqDETR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        bb = cfg["backbone"]; tf = cfg["transformer"]; hd = cfg["head"]
        self.backbone = ResNet18Backbone(out_channels=bb["out_channels"],
                                         pretrained=bb.get("pretrained", True),
                                         freeze_stages=bb.get("freeze_stages", 0))
        self.enc = TransformerEncoder(d_model=tf["d_model"], nheads=tf["nheads"],
                                      enc_layers=tf["enc_layers"], dim_feedforward=tf["dim_feedforward"],
                                      dropout=tf.get("dropout", 0.1))
        self.dec = TransformerDecoder(d_model=tf["d_model"], nheads=tf["nheads"],
                                      dec_layers=tf["dec_layers"], dim_feedforward=tf["dim_feedforward"],
                                      num_queries=tf["num_queries"], dropout=tf.get("dropout", 0.1))
        self.head = FreqDetrHead(num_classes=hd["num_classes"], d_model=tf["d_model"])

    def forward(self, images):
        f = self.backbone(images)     # [B,256,H/32,W/32]
        mem, hw = self.enc(f)         # [B,HW,C]
        dec = self.dec(mem)           # [B,Nq,C]
        out = self.head(dec)
        return out
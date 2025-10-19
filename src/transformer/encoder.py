import torch, torch.nn as nn, math

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model

    def forward(self, feat):  # [B,C,H,W]
        B,C,H,W = feat.shape
        device = feat.device
        y_embed = torch.linspace(0, 1, H, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.linspace(0, 1, W, device=device).unsqueeze(0).repeat(H, 1)
        pe_y = torch.stack([torch.sin(2*math.pi*y_embed), torch.cos(2*math.pi*y_embed)], dim=0)  # [2,H,W]
        pe_x = torch.stack([torch.sin(2*math.pi*x_embed), torch.cos(2*math.pi*x_embed)], dim=0)
        pe = torch.cat([pe_y, pe_x], dim=0)  # [4,H,W]
        pe = pe.unsqueeze(0).repeat(B,1,1,1)  # [B,4,H,W]
        pe = nn.functional.interpolate(pe, size=(H,W), mode="bilinear", align_corners=False)
        pe = nn.Conv2d(4, C, 1, bias=False).to(device)(pe)  # 简易投影，训练中会固定随机权重
        return feat + pe

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nheads=8, enc_layers=3, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, nheads, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, enc_layers)
        self.proj_in = nn.Identity()
        self.pos = None  # 简化：上面PE示例仅供参考，可直接不用PE

    def forward(self, feat):  # feat: [B,C,H,W]
        B,C,H,W = feat.shape
        x = feat.flatten(2).permute(0,2,1)  # [B,HW,C]
        x = self.proj_in(x)
        x = self.encoder(x)                 # [B,HW,C]
        return x, (H, W)
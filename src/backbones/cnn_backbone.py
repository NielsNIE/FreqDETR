import torch.nn as nn
import torchvision.models as tv

class ResNet18Backbone(nn.Module):
    def __init__(self, out_channels=256, pretrained=True, freeze_stages=0):
        super().__init__()
        m = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT if pretrained else None)
        # 取到 C5 特征图（最后一层卷积输出），输出通道512
        self.stem   = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.conv_out = nn.Conv2d(512, out_channels, 1)
        # 冻结
        self._freeze(freeze_stages)

    def _freeze(self, stages):
        layers = [self.stem, self.layer1, self.layer2, self.layer3]
        for i in range(min(stages, len(layers))):
            for p in layers[i].parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)     # 1/4
        x = self.layer1(x)   # 1/4
        x = self.layer2(x)   # 1/8
        x = self.layer3(x)   # 1/16
        x = self.layer4(x)   # 1/32
        x = self.conv_out(x) # -> 256通道
        return x  # [B,256,H/32,W/32]
import torch
import torch.nn as nn
from torchvision.models import vit_b_16

class ViTCarClassifier(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()

        self.vit = vit_b_16(pretrained=pretrained)
        embed_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.vit(x)
        return self.head(x)

    def freeze_backbone(self):
        for p in self.vit.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.vit.parameters():
            p.requires_grad = True

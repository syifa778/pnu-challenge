import torch
import torch.nn as nn
from torchvision import models

class CNNCarClassifier(nn.Module):
    def __init__(self, backbone="resnet50", num_classes=6, pretrained=True):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet50":
            # Load pretrained ResNet50
            base_model = models.resnet50(pretrained=pretrained)
            # Remove the final FC layer and average pooling
            self.backbone = nn.Sequential(*list(base_model.children())[:-2])
            feat_dim = 2048

        elif backbone == "vgg16":
            # VGG16 features only
            self.backbone = models.vgg16(pretrained=pretrained).features
            feat_dim = 512

        else:
            raise ValueError("Unsupported backbone")

        # Classification head
        self.head = nn.Sequential(
            nn.Conv2d(feat_dim, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)  # Should be (B, C, H, W)
        
        # Pass through classification head
        output = self.head(features)  # Should be (B, num_classes)
        
        return output

    def freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[INFO] Backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[INFO] Backbone unfrozen")
# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

# ===========================
# ConvNeXt Feature Extractor
# ===========================
class ConvNeXtFeature(nn.Module):
    def __init__(self):
        super().__init__()
        # Updated (no deprecation warning)
        self.model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        self.features = self.model.features  # keep only feature extractor

    def forward(self, x):
        x = self.features(x)  # [B, 1024, H, W]
        return x


# ===========================
# Vision Mamba (Simplified Safe Version)
# ===========================
class VisionMambaFeature(nn.Module):
    def __init__(self, input_channels=3, patch_size=16, embed_dim=768):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.conv(x)                    # [B, 768, H', W']
        x = x.flatten(2).transpose(1, 2)    # [B, N, 768]
        x = x.mean(dim=1)                  # [B, 768]
        return x


# ===========================
# TabNet-like MLP
# ===========================
class TabNetFeature(nn.Module):
    def __init__(self, input_dim, output_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# ===========================
# Full Model
# ===========================
class ConvNeXtVisionMambaTabNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.convnext = ConvNeXtFeature()
        self.vision_mamba = VisionMambaFeature()

        # ConvNeXt → 1024, VisionMamba → 768
        self.tabnet = TabNetFeature(input_dim=1024 + 768, output_dim=256)

        self.final_fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # ===== ConvNeXt =====
        c_feat = self.convnext(x)  # [B, 1024, H, W]
        c_feat = F.adaptive_avg_pool2d(c_feat, 1).flatten(1)  # [B, 1024]

        # ===== Vision Mamba =====
        v_feat = self.vision_mamba(x)  # [B, 768]

        # ===== Combine =====
        combined = torch.cat([c_feat, v_feat], dim=1)  # [B, 1792]

        # ===== TabNet =====
        t_feat = self.tabnet(combined)  # [B, 256]

        # ===== Final =====
        out = self.final_fc(t_feat)  # [B, num_classes]

        return out


# ===========================
# TEST (IMPORTANT)
# ===========================
if __name__ == "__main__":
    model = ConvNeXtVisionMambaTabNet(num_classes=4)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)  # should be [2, 4]
import torch
import torch.nn as nn
import timm

# --- Model Components ---
class DynamicKernelDepthwiseConv(nn.Module):
    def __init__(self, in_channels, filters):
        super().__init__()
        self.dw3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.dw5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.relu = nn.ReLU()
        self.pointwise = nn.Conv2d(in_channels * 2, filters, kernel_size=1)

    def forward(self, x):
        x3 = self.relu(self.dw3(x))
        x5 = self.relu(self.dw5(x))
        x = torch.cat([x3, x5], dim=1)
        x = self.relu(self.pointwise(x))
        return x

class CapsuleLikeLayer(nn.Module):
    def __init__(self, in_channels, num_capsules=10, dim_capsule=8):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_capsules * dim_capsule)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        x = x.view(-1, self.num_capsules, self.dim_capsule)
        x = torch.nn.functional.normalize(x, dim=-1)
        return x

class LungCancerModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        from timm import create_model
        self.base = create_model('ghostnet_100', pretrained=True, features_only=True)
        self.dkdc = DynamicKernelDepthwiseConv(160, 256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.caps = CapsuleLikeLayer(256, num_capsules=10, dim_capsule=8)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(10 * 8, num_classes)

    def forward(self, x):
        x = self.base(x)[-1]  # Using the last feature map from GhostNetV2
        x = self.dkdc(x)
        x = self.pool(x)
        x = self.caps(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

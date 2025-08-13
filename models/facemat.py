"""
FaceMat model implementation
Reference: Section 4 of the paper
"""
import torch
import torch.nn as nn
from .rvm import MobileNetV3Large
from .aspp import ASPP

class FaceMatTeacher(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Encoder
        self.encoder = MobileNetV3Large(pretrained=pretrained).features
        self.aspp = ASPP(960, 256)  # 960 is output channels from MobileNetV3
        
        # Alpha prediction head
        self.alpha_head = nn.Sequential(
            UpConv(256, 128),
            UpConv(128, 64),
            UpConv(64, 32),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
        # Uncertainty prediction head
        self.uncertainty_head = nn.Sequential(
            UpConv(256, 128),
            UpConv(128, 64),
            UpConv(64, 32),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Softplus()  # Ensure positive output
        )
        
    def forward(self, x):
        features = self.encoder(x)
        features = self.aspp(features)
        
        alpha = torch.sigmoid(self.alpha_head(features))
        uncertainty = self.uncertainty_head(features) + 1e-6  # Avoid zero
        return alpha, uncertainty

class FaceMatStudent(nn.Module):
    """Student model without uncertainty head"""
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = MobileNetV3Large(pretrained=pretrained).features
        self.aspp = ASPP(960, 256)
        self.alpha_head = nn.Sequential(
            UpConv(256, 128),
            UpConv(128, 64),
            UpConv(64, 32),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
    def forward(self, x):
        features = self.encoder(x)
        features = self.aspp(features)
        return torch.sigmoid(self.alpha_head(features))

class UpConv(nn.Module):
    """Upsampling convolution block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.up(x)
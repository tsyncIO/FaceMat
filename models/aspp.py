"""
Atrous Spatial Pyramid Pooling (ASPP) Module
Reference: 
  - DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs
  - FaceMat paper (Section 4.1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    """
    ASPP module with 4 parallel branches:
      1. 1x1 convolution
      2. 3x3 dilated conv (rate=6)
      3. 3x3 dilated conv (rate=12)
      4. 3x3 dilated conv (rate=18)
      5. Global average pooling
    """
    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18]):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Dilated convolutions
        self.conv3x3_1 = self._make_aspp_conv(in_channels, out_channels, rates[0])
        self.conv3x3_2 = self._make_aspp_conv(in_channels, out_channels, rates[1])
        self.conv3x3_3 = self._make_aspp_conv(in_channels, out_channels, rates[2])
        
        # Image pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Output convolution
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self._init_weights()

    def _make_aspp_conv(self, in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3, 
                padding=dilation, dilation=dilation, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_1(x)
        x3 = self.conv3x3_2(x)
        x4 = self.conv3x3_3(x)
        
        # Global average pooling
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5, size=x4.size()[2:], 
            mode='bilinear', align_corners=False
        )
        
        # Concatenate all branches
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.conv_out(x)

class ASPPConv(nn.Sequential):
    """Single ASPP convolution branch"""
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels, out_channels, 3, 
                padding=dilation, dilation=dilation, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super().__init__(*modules)

class ASPPPooling(nn.Sequential):
    """Image pooling branch for ASPP"""
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        size = x.shape[-2:]
        x = super().forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPPWithSeparableConv(nn.Module):
    """
    Efficient ASPP with depthwise separable convolutions
    Reference: MobileNetV3 and DeepLabv3+
    """
    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18]):
        super().__init__()
        self.conv1x1 = self._make_separable_conv(in_channels, out_channels, 1)
        self.conv3x3_1 = self._make_separable_conv(in_channels, out_channels, rates[0])
        self.conv3x3_2 = self._make_separable_conv(in_channels, out_channels, rates[1])
        self.conv3x3_3 = self._make_separable_conv(in_channels, out_channels, rates[2])
        self.global_avg_pool = ASPPPooling(in_channels, out_channels)
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def _make_separable_conv(self, in_channels, out_channels, dilation):
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(
                in_channels, in_channels, 3, 
                padding=dilation, dilation=dilation, 
                groups=in_channels, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_1(x)
        x3 = self.conv3x3_2(x)
        x4 = self.conv3x3_3(x)
        x5 = self.global_avg_pool(x)
        
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.conv_out(x)
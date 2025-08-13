"""
MobileNetV3 Backbone Implementation
Adapted from PyTorch official MobileNetV3 implementation
Reference: 
  - FaceMat paper (Section 4.1, uses RVM as baseline)
  - RVM: Robust High-Resolution Video Matting with Temporal Guidance
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenetv3
from torchvision.ops import SqueezeExcitation

class ConvBNReLU(nn.Sequential):
    """Convolution-BatchNorm-ReLU block"""
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, 
                 groups=1, dilation=1, norm_layer=nn.BatchNorm2d):
        padding = (kernel_size - 1) // 2 * dilation
        super().__init__(
            nn.Conv2d(
                in_planes, out_planes, kernel_size, stride, 
                padding, dilation=dilation, groups=groups, bias=False
            ),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    """Inverted residual block with SE"""
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1, 
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        
        layers = []
        if expand_ratio != 1:
            # Pointwise
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, 
                                     norm_layer=norm_layer))
        
        # Depthwise
        layers.extend([
            ConvBNReLU(
                hidden_dim, hidden_dim, stride=stride, 
                groups=hidden_dim, dilation=dilation,
                norm_layer=norm_layer
            ),
            SqueezeExcitation(hidden_dim, hidden_dim // 4),
            # Pointwise linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup)
        ])
        
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class MobileNetV3Large(nn.Module):
    """MobileNetV3-Large backbone for RVM"""
    def __init__(self, pretrained=True, dilated=True, output_stride=16):
        super().__init__()
        # Configuration for inverted residual blocks
        # [expansion, out_channels, num_blocks, stride, dilation]
        inverted_residual_setting = [
            # Expansion, out_channels, num_blocks, stride, dilation
            [1, 16, 1, 1, 1],
            [4, 24, 1, 2, 1],
            [3, 24, 1, 1, 1],
            [3, 40, 1, 2, 1],
            [3, 40, 1, 1, 1],
            [3, 40, 1, 1, 1],
            [6, 80, 1, 2, 1],
            [2.5, 80, 1, 1, 1],
            [2.3, 80, 1, 1, 1],
            [2.3, 80, 1, 1, 1],
            [6, 112, 1, 1, 1],
            [6, 112, 1, 1, 1],
            [6, 160, 1, 2, 1],
            [6, 160, 1, 1, 1],
            [6, 160, 1, 1, 1]
        ]
        
        # Adjust dilation rates for output stride
        if dilated and output_stride == 16:
            for i in range(12, len(inverted_residual_setting)):
                inverted_residual_setting[i][3] = 1  # Set stride to 1
                inverted_residual_setting[i][4] = 2  # Set dilation to 2
        
        # Build layers
        features = []
        # First layer
        features.append(ConvBNReLU(3, 16, stride=2, norm_layer=nn.BatchNorm2d))
        
        # Inverted residual blocks
        for t, c, n, s, d in inverted_residual_setting:
            for i in range(n):
                stride = s if i == 0 else 1
                dilation = d if stride == 1 else 1
                features.append(
                    InvertedResidual(
                        features[-1].out_channels if features else 16, 
                        c, stride, expand_ratio=t, dilation=dilation
                    )
                )
        
        # Last layers
        features.append(
            ConvBNReLU(
                inverted_residual_setting[-1][1], 960, 
                kernel_size=1, norm_layer=nn.BatchNorm2d
            )
        )
        
        self.features = nn.Sequential(*features)
        self.out_channels = 960
        
        # Initialize weights
        self._init_weights(pretrained)
    
    def _init_weights(self, pretrained):
        if pretrained:
            # Load pretrained MobileNetV3-Large weights
            pretrained_model = mobilenetv3.mobilenet_v3_large(pretrained=True)
            state_dict = self.state_dict()
            pretrained_state_dict = {}
            
            # Map our state dict to pretrained model
            for (k, v), (k_pt, v_pt) in zip(state_dict.items(), pretrained_model.state_dict().items()):
                if v.shape == v_pt.shape:
                    pretrained_state_dict[k] = v_pt
            
            # Load compatible weights
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Feature extraction
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            # Store feature maps at different resolutions
            if i in [3, 6, 12, 16]:
                features.append(x)
        
        # Return last feature map and intermediate features
        return features[-1]

class RVMDecoder(nn.Module):
    """
    Lightweight decoder for RVM
    Reference: Robust High-Resolution Video Matting with Temporal Guidance
    """
    def __init__(self, in_channels=960, out_channels=1):
        super().__init__()
        self.up_conv1 = self._make_upconv(in_channels, 128)
        self.up_conv2 = self._make_upconv(128, 64)
        self.up_conv3 = self._make_upconv(64, 32)
        self.conv_out = nn.Conv2d(32, out_channels, 1)
        
    def _make_upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.up_conv1(x)
        x = self.up_conv2(x)
        x = self.up_conv3(x)
        return torch.sigmoid(self.conv_out(x))

class RVMTemporalModule(nn.Module):
    """
    Temporal Fusion Module for RVM
    Processes recurrent features for video consistency
    """
    def __init__(self, channels=64):
        super().__init__()
        self.conv_gru = ConvGRU(channels, channels, kernel_size=3)
        
    def forward(self, x, hidden_state=None):
        if hidden_state is None:
            b, c, h, w = x.size()
            hidden_state = torch.zeros(b, c, h, w, device=x.device)
        
        return self.conv_gru(x, hidden_state)

class ConvGRU(nn.Module):
    """Convolutional GRU for temporal processing"""
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv_z = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding
        )
        self.conv_r = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding
        )
        self.conv_h = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding
        )
        
    def forward(self, x, state):
        xh = torch.cat([x, state], dim=1)
        z = torch.sigmoid(self.conv_z(xh))
        r = torch.sigmoid(self.conv_r(xh))
        h_candidate = torch.tanh(self.conv_h(torch.cat([x, r * state], dim=1)))
        new_state = (1 - z) * state + z * h_candidate
        return new_state
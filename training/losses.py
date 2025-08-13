"""
Loss functions for FaceMat training
Reference: Section 4.1 (Equation 2-4)
"""
import torch
import torch.nn.functional as F

def laplacian_pyramid_loss(pred, target, levels=5):
    """Pyramid Laplacian loss for matting"""
    def gauss_kernel(size=5, sigma=1.0):
        kernel = torch.exp(-(torch.arange(size) - size//2)**2 / (2*sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, 1) * kernel.view(1, 1, 1, size)
    
    def downsample(x):
        return F.avg_pool2d(x, kernel_size=2, stride=2)
    
    loss = 0
    for _ in range(levels):
        pred_laplace = pred - F.interpolate(downsample(pred), size=pred.shape[2:], mode='bilinear')
        target_laplace = target - F.interpolate(downsample(target), size=target.shape[2:], mode='bilinear')
        loss += F.l1_loss(pred_laplace, target_laplace)
        pred = downsample(pred)
        target = downsample(target)
    return loss / levels

def nll_loss(pred, target, var, beta=0.5):
    """Negative Log-Likelihood loss with uncertainty"""
    error = (pred - target).abs()
    loss = 0.5 * (error**2) / var + 0.5 * torch.log(var)
    return (loss * (var.detach()**beta)).mean()

def uncertainty_guided_l1(pred, target, uncertainty, w1=2, w2=2):
    """Uncertainty-guided L1 loss (Equation 4)"""
    weights = w1 + w2 * uncertainty
    return (weights * torch.abs(pred - target)).mean()
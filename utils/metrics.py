"""
Matting evaluation metrics implementation
Reference: 
  - Section 5.3 (Experiments) of the FaceMat paper
  - Traditional matting metrics from previous works
"""
import torch
import numpy as np
import scipy.ndimage

def calculate_matting_metrics(pred_alpha, true_alpha, trimap=None, epsilon=1e-6):
    """
    Calculate standard matting metrics between predicted and ground truth alpha mattes
    
    Args:
        pred_alpha: Predicted alpha matte (0-1), tensor of shape (B, 1, H, W)
        true_alpha: Ground truth alpha (0-1), tensor of same shape as pred_alpha
        trimap: Trimap tensor (0=bg, 0.5=unknown, 1=fg) of shape (B, 1, H, W)
        epsilon: Small value to avoid division by zero
        
    Returns:
        Dictionary containing:
        - mse: Mean Squared Error
        - sad: Sum of Absolute Differences
        - grad: Gradient error
        - conn: Connectivity error
    """
    metrics = {}
    B = pred_alpha.shape[0]
    
    # If trimap provided, only evaluate on unknown regions (trimap == 0.5)
    if trimap is not None:
        mask = (trimap == 0.5).float()
    else:
        mask = torch.ones_like(pred_alpha)
    
    # Convert to numpy for some metrics that are easier to implement in numpy
    pred_np = pred_alpha.detach().cpu().numpy()
    true_np = true_alpha.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    
    # Initialize metric accumulators
    mse_total = 0.0
    sad_total = 0.0
    grad_total = 0.0
    conn_total = 0.0
    
    for i in range(B):
        pred = pred_np[i, 0]  # (H, W)
        true = true_np[i, 0]  # (H, W)
        m = mask_np[i, 0]     # (H, W)
        
        # Only evaluate on masked regions
        pred_masked = pred * m
        true_masked = true * m
        
        # Mean Squared Error (MSE)
        mse = np.sum((pred_masked - true_masked)**2) / (np.sum(m) + epsilon)
        mse_total += mse
        
        # Sum of Absolute Differences (SAD)
        sad = np.sum(np.abs(pred_masked - true_masked))
        sad_total += sad
        
        # Gradient Error
        grad = gradient_error(pred_masked, true_masked, m)
        grad_total += grad
        
        # Connectivity Error
        conn = connectivity_error(pred_masked, true_masked, m)
        conn_total += conn
    
    metrics['mse'] = mse_total / B
    metrics['sad'] = sad_total / B
    metrics['grad'] = grad_total / B
    metrics['conn'] = conn_total / B
    
    return metrics

def gradient_error(pred, true, mask, sigma=1.4):
    """
    Gradient error metric
    Computes the differences between gradients of predicted and ground truth alpha
    
    Args:
        pred: Predicted alpha matte (H, W)
        true: Ground truth alpha (H, W)
        mask: Evaluation mask (H, W)
        sigma: Sigma for Gaussian gradient calculation
        
    Returns:
        Gradient error score
    """
    # Calculate gradients
    pred_grad = gaussian_gradient(pred, sigma)
    true_grad = gaussian_gradient(true, sigma)
    
    # Calculate error only in masked regions
    error = np.abs(pred_grad - true_grad) * mask
    return np.sum(error) / (np.sum(mask) + 1e-6)

def gaussian_gradient(img, sigma):
    """
    Compute Gaussian gradients for an image
    """
    # Gaussian filter
    smoothed = scipy.ndimage.gaussian_filter(img, sigma, mode='reflect')
    
    # Sobel operators
    dx = scipy.ndimage.sobel(smoothed, axis=1, mode='reflect')
    dy = scipy.ndimage.sobel(smoothed, axis=0, mode='reflect')
    
    return np.sqrt(dx**2 + dy**2)

def connectivity_error(pred, true, mask, step=0.1):
    """
    Connectivity error metric
    Measures how well the predicted alpha matte preserves connectedness
    
    Args:
        pred: Predicted alpha matte (H, W)
        true: Ground truth alpha (H, W)
        mask: Evaluation mask (H, W)
        step: Threshold step size
        
    Returns:
        Connectivity error score
    """
    error = 0.0
    h, w = pred.shape
    
    for t in np.arange(0, 1.0, step):
        pred_mask = (pred >= t).astype(np.uint8)
        true_mask = (true >= t).astype(np.uint8)
        
        # Label connected components
        pred_labeled = scipy.ndimage.label(pred_mask)[0]
        true_labeled = scipy.ndimage.label(true_mask)[0]
        
        # Find largest connected component in ground truth
        true_counts = np.bincount(true_labeled.ravel())
        if len(true_counts) > 1:  # Exclude background
            true_largest = np.argmax(true_counts[1:]) + 1
            true_largest_mask = (true_labeled == true_largest)
            
            # Find corresponding component in prediction
            overlap = pred_labeled * true_largest_mask
            overlap_counts = np.bincount(overlap.ravel())
            if len(overlap_counts) > 1:
                pred_largest = np.argmax(overlap_counts[1:]) + 1
                pred_largest_mask = (pred_labeled == pred_largest)
                
                # Calculate difference
                diff = np.abs(true_largest_mask.astype(float) - pred_largest_mask.astype(float))
                error += np.sum(diff * mask)
    
    return error / (np.sum(mask) + 1e-6)

def uncertainty_metrics(pred_alpha, true_alpha, uncertainty, trimap=None):
    """
    Calculate metrics specific to uncertainty estimation quality
    
    Args:
        pred_alpha: Predicted alpha matte (0-1), tensor of shape (B, 1, H, W)
        true_alpha: Ground truth alpha (0-1), tensor of same shape
        uncertainty: Predicted uncertainty (B, 1, H, W)
        trimap: Optional trimap tensor
        
    Returns:
        Dictionary containing:
        - uncertainty_auc: Area under ROC curve for error vs uncertainty
        - uncertainty_correlation: Pearson correlation between error and uncertainty
    """
    # Flatten all tensors
    pred_flat = pred_alpha.flatten()
    true_flat = true_alpha.flatten()
    uncert_flat = uncertainty.flatten()
    
    if trimap is not None:
        mask = (trimap == 0.5).float().flatten()
        pred_flat = pred_flat[mask > 0]
        true_flat = true_flat[mask > 0]
        uncert_flat = uncert_flat[mask > 0]
    
    # Calculate absolute errors
    errors = torch.abs(pred_flat - true_flat).cpu().numpy()
    uncert_np = uncert_flat.cpu().numpy()
    
    # Calculate correlation
    correlation = np.corrcoef(errors, uncert_np)[0, 1]
    
    # Calculate AUC for error prediction
    from sklearn.metrics import roc_auc_score
    binary_errors = (errors > 0.1).astype(np.float32)  # Threshold for "large error"
    auc = roc_auc_score(binary_errors, uncert_np)
    
    return {
        'uncertainty_auc': auc,
        'uncertainty_correlation': correlation
    }
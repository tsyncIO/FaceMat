"""
Visualization utilities for FaceMat results
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

def visualize_results(images, pred_alphas, true_alphas=None, trimaps=None, 
                     save_dir=None, prefix='', dpi=100):
    """
    Visualize matting results side-by-side
    
    Args:
        images: Input images tensor (B, 3, H, W)
        pred_alphas: Predicted alpha mattes (B, 1, H, W)
        true_alphas: Ground truth alpha mattes (optional)
        trimaps: Trimaps (optional)
        save_dir: Directory to save visualizations
        prefix: Prefix for saved filenames
        dpi: DPI for saved figures
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Convert tensors to numpy arrays
    images_np = images.cpu().numpy().transpose(0, 2, 3, 1)  # (B, H, W, 3)
    pred_alphas_np = pred_alphas.cpu().numpy().transpose(0, 2, 3, 1)  # (B, H, W, 1)
    
    if true_alphas is not None:
        true_alphas_np = true_alphas.cpu().numpy().transpose(0, 2, 3, 1)
    
    if trimaps is not None:
        trimaps_np = trimaps.cpu().numpy().transpose(0, 2, 3, 1)
    
    for i in range(images_np.shape[0]):
        fig, axes = plt.subplots(1, 4 if true_alphas is not None else 2, 
                                figsize=(15, 5))
        fig.set_dpi(dpi)
        
        # Input image
        img = (images_np[i] * 255).astype(np.uint8)
        axes[0].imshow(img)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Predicted alpha
        pred_alpha = (pred_alphas_np[i, ..., 0] * 255).astype(np.uint8)
        axes[1].imshow(pred_alpha, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title('Predicted Alpha')
        axes[1].axis('off')
        
        if true_alphas is not None:
            # Ground truth alpha
            true_alpha = (true_alphas_np[i, ..., 0] * 255).astype(np.uint8)
            axes[2].imshow(true_alpha, cmap='gray', vmin=0, vmax=255)
            axes[2].set_title('Ground Truth Alpha')
            axes[2].axis('off')
            
            # Error map
            error = np.abs(pred_alphas_np[i] - true_alphas_np[i])
            error = (error * 255).astype(np.uint8)
            axes[3].imshow(error, cmap='hot', vmin=0, vmax=255)
            axes[3].set_title('Error Map')
            axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, f'{prefix}_sample_{i}.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
            plt.close()
        else:
            plt.show()

def visualize_uncertainty(image, alpha, uncertainty, save_path=None):
    """
    Visualize uncertainty estimates alongside predictions
    
    Args:
        image: Input image tensor (3, H, W)
        alpha: Predicted alpha matte (1, H, W)
        uncertainty: Uncertainty map (1, H, W)
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input image
    img = image.cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Predicted alpha
    alpha_np = alpha.squeeze().cpu().numpy()
    axes[1].imshow(alpha_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Predicted Alpha')
    axes[1].axis('off')
    
    # Uncertainty
    uncert_np = uncertainty.squeeze().cpu().numpy()
    im = axes[2].imshow(uncert_np, cmap='viridis')
    axes[2].set_title('Uncertainty')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close()
    else:
        plt.show()

def create_composite(image, alpha, background=None):
    """
    Create composite image using predicted alpha
    
    Args:
        image: Input image tensor (3, H, W)
        alpha: Predicted alpha matte (1, H, W)
        background: Optional background image tensor (3, H, W)
    
    Returns:
        Composite image as PIL Image
    """
    # Convert tensors to numpy
    image_np = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    alpha_np = alpha.cpu().numpy().squeeze()[..., np.newaxis]  # (H, W, 1)
    
    if background is None:
        # Default to checkerboard background
        bg = np.zeros_like(image_np)
        checker = np.indices((image_np.shape[0]//8, image_np.shape[1]//8)).sum(axis=0) % 2
        checker = np.kron(checker, np.ones((8, 8, 1))) * 255
        bg[:checker.shape[0], :checker.shape[1]] = checker[:bg.shape[0], :bg.shape[1]]
    else:
        bg = background.cpu().numpy().transpose(1, 2, 0)
    
    # Composite foreground and background
    composite = image_np * alpha_np + bg * (1 - alpha_np)
    composite = np.clip(composite, 0, 255).astype(np.uint8)
    
    return Image.fromarray(composite)

def save_alpha_as_image(alpha, path, colormap='gray'):
    """
    Save alpha matte as an image file
    
    Args:
        alpha: Alpha matte tensor (1, H, W)
        path: Output path
        colormap: Matplotlib colormap name
    """
    alpha_np = alpha.squeeze().cpu().numpy()
    plt.imsave(path, alpha_np, cmap=colormap, vmin=0, vmax=1)

def plot_training_curve(log_path, save_path=None):
    """
    Plot training curves from log file
    
    Args:
        log_path: Path to training log file
        save_path: Path to save plot
    """
    import pandas as pd
    
    # Read log file
    log = pd.read_csv(log_path)
    
    # Create plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot losses
    if 'train_loss' in log.columns:
        ax1.plot(log['epoch'], log['train_loss'], 'b-', label='Train Loss')
    if 'val_loss' in log.columns:
        ax1.plot(log['epoch'], log['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')
    
    # Plot metrics on secondary axis if available
    if any(m in log.columns for m in ['mse', 'sad', 'grad']):
        ax2 = ax1.twinx()
        
        if 'mse' in log.columns:
            ax2.plot(log['epoch'], log['mse'], 'g--', label='MSE')
        if 'sad' in log.columns:
            ax2.plot(log['epoch'], log['sad'], 'm--', label='SAD')
        if 'grad' in log.columns:
            ax2.plot(log['epoch'], log['grad'], 'c--', label='Grad')
        
        ax2.set_ylabel('Metric Value')
        ax2.legend(loc='upper right')
    
    plt.title('Training Curves')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close()
    else:
        plt.show()
"""
Data transformations for FaceMat training and evaluation
Reference: Section 5.1 (Implementation Details) of the paper
"""
import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter

class MattingTransform:
    """Composition of transforms for matting samples"""
    def __init__(self, size, augment=True, trimap_dilation=15):
        """
        Args:
            size: Target size (height, width)
            augment: Apply data augmentation if True
            trimap_dilation: Kernel size for trimap dilation
        """
        self.size = size
        self.augment = augment
        self.trimap_dilation = trimap_dilation
        
    def __call__(self, sample):
        image = sample['image']
        alpha = sample['alpha']
        trimap = sample.get('trimap', None)
        
        # Resize all components
        image = F.resize(image, self.size, interpolation=Image.BILINEAR)
        alpha = F.resize(alpha, self.size, interpolation=Image.BILINEAR)
        
        if trimap is not None:
            trimap = F.resize(trimap, self.size, interpolation=Image.NEAREST)
        
        # Data augmentation
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = F.hflip(image)
                alpha = F.hflip(alpha)
                if trimap is not None:
                    trimap = F.hflip(trimap)
            
            # Random rotation (-15 to 15 degrees)
            angle = random.uniform(-15, 15)
            image = F.rotate(image, angle, interpolation=Image.BILINEAR)
            alpha = F.rotate(alpha, angle, interpolation=Image.BILINEAR)
            if trimap is not None:
                trimap = F.rotate(trimap, angle, interpolation=Image.NEAREST)
            
            # Color jitter (40% probability)
            if random.random() < 0.4:
                jitter = T.ColorJitter(
                    brightness=0.2,
                    contrast=0.15,
                    saturation=0.1,
                    hue=0.05
                )
                image = jitter(image)
            
            # Gaussian blur (20% probability)
            if random.random() < 0.2:
                sigma = random.uniform(0.1, 1.5)
                image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        # Convert to tensors
        image_tensor = F.to_tensor(image)
        alpha_tensor = F.to_tensor(alpha).squeeze(0)  # Remove channel dim
        
        # Normalize image (ImageNet stats)
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        image_tensor = normalize(image_tensor)
        
        # Process trimap
        if trimap is not None:
            trimap_tensor = self.process_trimap(trimap)
        else:
            trimap_tensor = torch.zeros_like(alpha_tensor)
        
        return {
            'image': image_tensor,
            'alpha': alpha_tensor,
            'trimap': trimap_tensor
        }
    
    def process_trimap(self, trimap):
        """Convert trimap PIL image to tensor and encode as classes"""
        # Convert to numpy array
        trimap_np = np.array(trimap, dtype=np.uint8)
        
        # Handle different trimap formats
        if trimap_np.ndim == 3:  # RGB format
            trimap_np = trimap_np[..., 0]  # Use first channel
        
        # Class encoding: 
        #   0 = background (0-85), 
        #   1 = unknown (86-170), 
        #   2 = foreground (171-255)
        classes = np.zeros_like(trimap_np, dtype=np.int64)
        classes[(trimap_np >= 86) & (trimap_np <= 170)] = 1
        classes[trimap_np > 170] = 2
        
        return torch.from_numpy(classes)

class VideoTransform:
    """Transforms for video consistency during inference"""
    def __init__(self, size, temporal_length=5):
        """
        Args:
            size: Target size (height, width)
            temporal_length: Number of frames to process
        """
        self.size = size
        self.temporal_length = temporal_length
        self.transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
        ])
        
    def __call__(self, frames):
        """
        Process a sequence of video frames
        Args:
            frames: List of PIL Images
        Returns:
            Tensor of shape (T, C, H, W)
        """
        transformed = []
        for frame in frames:
            transformed.append(self.transform(frame))
        return torch.stack(transformed, dim=0)

class RandomOcclusion:
    """Apply random occlusion during training (Section 5.1)"""
    def __init__(self, occ_dir, occlusion_ratio=0.25):
        """
        Args:
            occ_dir: Directory with occlusion images
            occlusion_ratio: Target occlusion ratio
        """
        self.occ_generator = SyntheticOcclusionGenerator(occ_dir)
        self.occlusion_ratio = occlusion_ratio
        
    def __call__(self, sample):
        image = sample['image']
        
        # Apply occlusion with probability based on occlusion ratio
        if random.random() < self.occlusion_ratio:
            image, alpha = self.occ_generator.apply_occlusion(image)
            sample['image'] = image
            sample['alpha'] = alpha
            sample['trimap'] = self._generate_trimap(alpha)
        
        return sample
    
    def _generate_trimap(self, alpha, kernel_size=15):
        """Generate trimap from alpha matte"""
        alpha_np = (alpha * 255).astype(np.uint8) if isinstance(alpha, np.ndarray) else np.array(alpha)
        fg = (alpha_np > 240).astype(np.uint8)
        unknown = (alpha_np >= 20) & (alpha_np <= 240)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        unknown = cv2.dilate(unknown.astype(np.uint8), kernel, iterations=2)
        return Image.fromarray(np.where(fg, 255, np.where(unknown, 128, 0))

class MotionBlur:
    """Apply motion blur to simulate video conditions (Section 5.1)"""
    def __init__(self, kernel_size=15, intensity=0.3):
        """
        Args:
            kernel_size: Size of motion kernel
            intensity: Probability of applying blur
        """
        self.kernel_size = kernel_size
        self.intensity = intensity
        
    def __call__(self, sample):
        if random.random() < self.intensity:
            image = sample['image']
            
            # Create motion blur kernel
            kernel = np.zeros((self.kernel_size, self.kernel_size))
            angle = random.randint(0, 180)
            length = random.randint(3, self.kernel_size)
            center = (self.kernel_size // 2, self.kernel_size // 2)
            
            # Draw line in kernel
            cv2.line(kernel, 
                    (center[0] - length//2, center[1]),
                    (center[0] + length//2, center[1]),
                    1, thickness=1)
            
            # Rotate kernel
            M = cv2.getRotationMatrix2D(center, angle, 1)
            kernel = cv2.warpAffine(kernel, M, (self.kernel_size, self.kernel_size))
            kernel /= kernel.sum()
            
            # Apply blur
            image_np = np.array(image)
            blurred = cv2.filter2D(image_np, -1, kernel)
            sample['image'] = Image.fromarray(blurred)
            
        return sample

class ToDevice:
    """Move sample to specified device"""
    def __init__(self, device='cuda'):
        self.device = device
        
    def __call__(self, sample):
        return {k: v.to(self.device) if torch.is_tensor(v) else v 
                for k, v in sample.items()}
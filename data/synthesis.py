"""
Synthetic data generation for CelebAMat dataset
Reference: Section 3.2 and Figure 5 of the paper
"""
import cv2
import numpy as np
import random
from PIL import Image

class SyntheticOcclusionGenerator:
    def __init__(self, occ_dir, min_scale=0.2, max_scale=0.5):
        """
        Args:
            occ_dir: Directory with RGBA occlusion images
            min_scale: Minimum scale for occlusions (relative to face)
            max_scale: Maximum scale for occlusions
        """
        self.occ_paths = [os.path.join(occ_dir, f) for f in os.listdir(occ_dir)]
        self.min_scale = min_scale
        self.max_scale = max_scale
        
    def apply_occlusion(self, face_img, occ_img=None):
        """
        Apply random occlusion to a face image
        Args:
            face_img: PIL Image (RGB)
            occ_img: Optional PIL Image (RGBA)
        Returns:
            composited: PIL Image (RGB) with occlusion
            alpha: Occlusion alpha matte (H, W) float32
        """
        if not occ_img:
            occ_img = self._load_random_occlusion()
            
        face_w, face_h = face_img.size
        occ_img = self._apply_random_transform(occ_img)
        
        # Random scale
        scale = random.uniform(self.min_scale, self.max_scale)
        occ_w, occ_h = int(face_w * scale), int(face_h * scale)
        occ_img = occ_img.resize((occ_w, occ_h), Image.LANCZOS)
        
        # Random position
        x = random.randint(0, face_w - occ_w)
        y = random.randint(0, face_h - occ_h)
        
        # Composite
        face_arr = np.array(face_img)
        occ_arr = np.array(occ_img)
        alpha = occ_arr[..., 3] / 255.0
        alpha = np.expand_dims(alpha, axis=-1)
        
        for c in range(3):
            face_arr[y:y+occ_h, x:x+occ_w, c] = (
                face_arr[y:y+occ_h, x:x+occ_w, c] * (1 - alpha) + 
                occ_arr[..., c] * alpha
            )
        
        # Generate full alpha matte
        full_alpha = np.zeros((face_h, face_w), dtype=np.float32)
        full_alpha[y:y+occ_h, x:x+occ_w] = alpha.squeeze()
        
        return Image.fromarray(face_arr.astype('uint8')), full_alpha
    
    def _load_random_occlusion(self):
        path = random.choice(self.occ_paths)
        return Image.open(path).convert('RGBA')
    
    def _apply_random_transform(self, img):
        """Apply affine transformation with motion simulation"""
        arr = np.array(img)
        angle = random.uniform(-30, 30)
        dx, dy = random.randint(-20, 20), random.randint(-20, 20)
        scale = random.uniform(0.9, 1.1)
        
        M = cv2.getRotationMatrix2D((arr.shape[1]/2, arr.shape[0]/2), angle, scale)
        M[:, 2] += [dx, dy]
        
        return Image.fromarray(
            cv2.warpAffine(arr, M, (arr.shape[1], arr.shape[0]), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REFLECT
        )
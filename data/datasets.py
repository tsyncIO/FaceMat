"""
CelebAMat dataset implementation
Reference: Section 3.2 and Section 5.1 of the paper
"""
import torch
from torch.utils.data import Dataset

class CelebAMatDataset(Dataset):
    def __init__(self, face_dir, occ_dir, size=10000, transform=None):
        """
        Args:
            face_dir: Directory with face images
            occ_dir: Directory with occlusion images
            size: Number of synthetic samples to generate
            transform: Composition of transforms
        """
        self.face_paths = [os.path.join(face_dir, f) for f in os.listdir(face_dir)]
        self.occ_generator = SyntheticOcclusionGenerator(occ_dir)
        self.transform = transform
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Random face image
        face_path = random.choice(self.face_paths)
        face_img = Image.open(face_path).convert('RGB')
        
        # Apply synthetic occlusion
        img, alpha = self.occ_generator.apply_occlusion(face_img)
        
        # Generate trimap
        trimap = self._generate_trimap(alpha)
        
        sample = {
            'image': img,
            'alpha': alpha,
            'trimap': trimap
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def _generate_trimap(self, alpha, kernel_size=15):
        """Generate trimap from alpha matte (Section 3.1)"""
        fg = (alpha > 0.95).astype(np.uint8)
        unknown = ((alpha >= 0.05) & (alpha <= 0.95)).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        unknown = cv2.dilate(unknown, kernel, iterations=2)
        trimap = fg * 255 + unknown * 128
        return trimap
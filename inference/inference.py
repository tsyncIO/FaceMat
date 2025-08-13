"""
FaceMat inference module
Supports image/video processing and real-time filtering
"""
class FaceMatInference:
    def __init__(self, model_path, device='cuda'):
        self.model = FaceMatStudent()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.device = device
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def predict_alpha(self, image):
        """Predict alpha matte from input image"""
        original_size = image.size
        img_t = self.transform(image.resize((512, 512))).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            alpha = self.model(img_t).cpu().squeeze()
        
        return Image.fromarray((alpha * 255).astype('uint8')).resize(original_size)
    
    def apply_filter(self, image, filter_fn):
        """
        Apply face filter with occlusion handling
        Steps: 
          1. Predict occlusion matte
          2. Complete face (optional)
          3. Apply transformation
          4. Composite with original
        """
        alpha = self.predict_alpha(image)
        clean_face = self.complete_face(image, alpha)  # Optional inpainting
        transformed = filter_fn(clean_face)
        return self.composite(image, transformed, alpha)
    
    def composite(self, bg, fg, alpha):
        """Alpha blending composition"""
        alpha = np.array(alpha) / 255.0
        result = bg * (1 - alpha[..., None]) + fg * alpha[..., None]
        return Image.fromarray(result.astype('uint8'))
    
    def process_video(self, video_path, output_path, filter_fn):
        """Process video frame-by-frame"""
        cap = cv2.VideoCapture(video_path)
        writer = cv2.VideoWriter(output_path, 
                                cv2.VideoWriter_fourcc(*'MP4V'), 
                                30, 
                                (int(cap.get(3)), int(cap.get(4))))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.apply_filter(Image.fromarray(frame_rgb), filter_fn)
            writer.write(cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR))
        
        cap.release()
        writer.release()
"""
Real-time webcam demo for FaceMat with face filters
"""
import argparse
import time
import cv2
import numpy as np
from PIL import Image
from .inference import FaceMatInference
from .filters import create_filter

def main():
    parser = argparse.ArgumentParser(description='FaceMat Webcam Demo')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to trained FaceMat model')
    parser.add_argument('--filter', type=str, default='stylization', 
                        help='Face filter to apply (none, stylization, face_swap, etc.)')
    parser.add_argument('--swap_face', type=str, default=None,
                        help='Path to target face for face swap')
    parser.add_argument('--mask', type=str, default=None,
                        help='Path to mask image for mask filter')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video file path')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target FPS for output video')
    args = parser.parse_args()

    # Initialize FaceMat
    print("Loading FaceMat model...")
    facemat = FaceMatInference(model_path=args.model, device='cuda')
    
    # Initialize filter
    filter_kwargs = {}
    if args.filter == 'face_swap' and args.swap_face:
        filter_kwargs['target_face_path'] = args.swap_face
    elif args.filter == 'mask' and args.mask:
        filter_kwargs['mask_path'] = args.mask
        
    face_filter = create_filter(args.filter, **filter_kwargs)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
    
    # Get webcam resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare video writer if output specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))
    
    # For FPS calculation
    prev_time = time.time()
    fps_counter = 0
    avg_fps = 0
    
    print("Starting webcam demo. Press 'q' to quit...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to PIL Image (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            
            try:
                # Process frame with FaceMat
                result = facemat.apply_filter(pil_frame, face_filter.apply)
            except Exception as e:
                print(f"Processing error: {e}")
                result = pil_frame
            
            # Convert back to OpenCV format (BGR)
            result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            fps_counter = fps_counter * 0.9 + fps * 0.1
            avg_fps = fps_counter if avg_fps == 0 else 0.95 * avg_fps + 0.05 * fps
            
            # Display FPS
            cv2.putText(result_bgr, f"FPS: {avg_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show result
            cv2.imshow("FaceMat Webcam", result_bgr)
            
            # Write to output
            if writer:
                writer.write(result_bgr)
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Demo stopped")

if __name__ == "__main__":
    main()
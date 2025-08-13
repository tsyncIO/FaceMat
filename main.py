#!/usr/bin/env python3
"""
FaceMat: Uncertainty-Guided Face Matting for Occlusion-Aware Face Transformation
Main entry point for training and inference
Reference: MM '25 paper
"""
import argparse
import os
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.datasets import CelebAMatDataset
from data.transforms import MattingTransform
from models.facemat import FaceMatTeacher, FaceMatStudent
from training.train_stage1 import train_stage1
from training.train_stage2 import train_stage2
from inference.inference import FaceMatInference
from utils.metrics import calculate_matting_metrics
from utils.visualize import visualize_results

def parse_args():
    parser = argparse.ArgumentParser(description='FaceMat: Uncertainty-Guided Face Matting')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Training commands
    train_parser = subparsers.add_parser('train', help='Train FaceMat model')
    train_parser.add_argument('--stage', type=int, choices=[1, 2], required=True,
                             help='Training stage (1: teacher, 2: student)')
    train_parser.add_argument('--config', type=str, required=True,
                             help='Path to config file')
    train_parser.add_argument('--resume', type=str, default=None,
                             help='Path to checkpoint to resume from')
    
    # Inference commands
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--input', type=str, required=True,
                             help='Input image/video path')
    infer_parser.add_argument('--output', type=str, required=True,
                             help='Output directory')
    infer_parser.add_argument('--model', type=str, required=True,
                             help='Path to trained model')
    infer_parser.add_argument('--filter', type=str, default='none',
                             help='Filter to apply (none, stylization, etc.)')
    
    # Evaluation commands
    eval_parser = subparsers.add_parser('eval', help='Evaluate model')
    eval_parser.add_argument('--dataset', type=str, required=True,
                            help='Path to evaluation dataset')
    eval_parser.add_argument('--model', type=str, required=True,
                            help='Path to trained model')
    eval_parser.add_argument('--output', type=str, default=None,
                            help='Output directory for visualizations')
    
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_training_data(config):
    transform = MattingTransform(
        size=config['data']['input_size'],
        augment=config['training']['augment']
    )
    
    dataset = CelebAMatDataset(
        face_dir=config['data']['face_dir'],
        occ_dir=config['data']['occ_dir'],
        size=config['data']['dataset_size'],
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    return loader

def train(config, args):
    # Prepare data
    train_loader = prepare_training_data(config)
    
    if args.stage == 1:
        # Stage 1: Train teacher model
        model = FaceMatTeacher(pretrained=config['model']['pretrained'])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        if args.resume:
            model.load_state_dict(torch.load(args.resume))
        
        train_stage1(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epochs=config['training']['epochs'],
            device=config['training']['device'],
            save_dir=config['training']['save_dir'],
            save_interval=config['training']['save_interval']
        )
    else:
        # Stage 2: Train student model with distillation
        teacher = FaceMatTeacher(pretrained=False)
        teacher.load_state_dict(torch.load(config['training']['teacher_ckpt']))
        teacher.eval()
        
        student = FaceMatStudent(pretrained=config['model']['pretrained'])
        
        if args.resume:
            student.load_state_dict(torch.load(args.resume))
        
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        train_stage2(
            teacher=teacher,
            student=student,
            train_loader=train_loader,
            optimizer=optimizer,
            epochs=config['training']['epochs'],
            device=config['training']['device'],
            save_dir=config['training']['save_dir'],
            use_ema=config['training']['use_ema']
        )

def infer(args):
    # Initialize inference pipeline
    facemat = FaceMatInference(model_path=args.model)
    
    # Check input type
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        # Single image processing
        image = Image.open(input_path).convert('RGB')
        alpha = facemat.predict_alpha(image)
        alpha.save(output_path / f'{input_path.stem}_alpha.png')
        
        # Apply filter if specified
        if args.filter != 'none':
            from inference.filters import create_filter
            face_filter = create_filter(args.filter)
            result = facemat.apply_filter(image, face_filter.apply)
            result.save(output_path / f'{input_path.stem}_filtered.png')
    else:
        # Video processing
        facemat.process_video(
            video_path=str(input_path),
            output_path=str(output_path / 'output.mp4'),
            filter_fn=lambda x: x  # Identity function by default
        )

def evaluate(args):
    # Load model
    model = FaceMatStudent()
    model.load_state_dict(torch.load(args.model))
    model.eval()
    
    # Prepare dataset (should have ground truth alpha)
    transform = MattingTransform(size=(512, 512), augment=False)
    dataset = CelebAMatDataset(
        face_dir=args.dataset,
        occ_dir=None,  # Assuming dataset already has occlusions
        transform=transform,
        size=None  # Use all samples
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Evaluation loop
    metrics = {
        'mse': 0,
        'sad': 0,
        'grad': 0,
        'conn': 0
    }
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            images = batch['image'].to('cuda')
            alpha_gt = batch['alpha'].to('cuda')
            trimaps = batch['trimap'].to('cuda')
            
            alpha_pred = model(images)
            
            batch_metrics = calculate_matting_metrics(
                alpha_pred, alpha_gt, trimaps
            )
            
            # Accumulate metrics
            for k in metrics:
                metrics[k] += batch_metrics[k]
            
            # Visualize samples
            if args.output and i < 5:  # Save first 5 batches
                visualize_results(
                    images, alpha_pred, alpha_gt,
                    save_dir=args.output,
                    prefix=f'batch_{i}'
                )
    
    # Average metrics
    num_samples = len(dataset)
    for k in metrics:
        metrics[k] /= num_samples
    
    print("\nEvaluation Results:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"SAD: {metrics['sad']:.4f}")
    print(f"Grad: {metrics['grad']:.4f}")
    print(f"Conn: {metrics['conn']:.4f}")

def main():
    args = parse_args()
    
    if args.command == 'train':
        config = load_config(args.config)
        train(config, args)
    elif args.command == 'infer':
        infer(args)
    elif args.command == 'eval':
        evaluate(args)

if __name__ == '__main__':
    main()
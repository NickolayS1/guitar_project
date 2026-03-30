#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for multimodal guitar transcription.

Analogous to inference_enhanced_baseline.py but for multimodal model.

Usage:
    python training/inference_multimodal.py \
        --checkpoint experiments/multimodal/checkpoints/checkpoint_best.pth \
        --config configs/multimodal_config.yaml \
        --split test
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml

from models.multimodal_cnn import MultimodalCNN
from data_loading.video_dataset import VideoFramesDataset, collate_fn as video_collate_fn
from data_loading.dataloader import create_dataloader
from evaluation.metrics import GuitarTranscriptionMetrics


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate_model(model, dataloader, device, onset_threshold=0.5, use_dummy_video=True):
    """Evaluate model on dataset."""
    model.eval()
    
    metrics_collector = GuitarTranscriptionMetrics(
        midi_min=36,  # C2
        midi_max=108  # C8
    )
    
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        # Handle both dict (real video) and tuple (dummy video) batches
        if isinstance(batch, dict):
            # Real video: dict format
            cqt = batch['cqt'].to(device)
            video = batch['video'].to(device)
            onset_true = batch['onset'].to(device)
            pitch_true = batch['pitch'].to(device)
        else:
            # Dummy video: tuple format (cqt, video, onset, pitch, has_onset)
            cqt = batch[0].to(device)
            video = batch[1].to(device)
            onset_true = batch[2].to(device)
            pitch_true = batch[3].to(device)
        
        cqt = cqt.unsqueeze(1)  # [B, 13, 72] → [B, 1, 13, 72]
        
        # Forward pass
        onset_pred, pitch_pred = model(cqt, video)
        
        # Update metrics
        metrics_collector.update_batch(
            onset_pred=onset_pred,
            pitch_pred=pitch_pred,
            onset_true=onset_true,
            pitch_true=pitch_true,
            mask=(onset_true > 0.5).float(),
            onset_threshold=onset_threshold
        )
        
        n_batches += 1
    
    # Compute final metrics
    metrics = metrics_collector.compute_all()
    
    return metrics, metrics_collector


def main():
    parser = argparse.ArgumentParser(description='Evaluate multimodal guitar transcription model')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/multimodal_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate (train/val/test)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Onset detection threshold')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON (optional)')
    parser.add_argument('--save_predictions', type=str, default=None,
                        help='Path to save predictions PT file (optional)')
    parser.add_argument('--dataset', type=str, default='own_sessions',
                        choices=['guitarset', 'own_sessions'],
                        help='Dataset to use (guitarset or own_sessions)')
    parser.add_argument('--dummy-video', action='store_true',
                        help='Use dummy video (for testing without video data)')
    parser.add_argument('--use-real-video', action='store_true',
                        help='Use real video frames from video.mp4')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device if args.device else 
                         ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}\n")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model
    print(f"Creating model...")
    model = MultimodalCNN(
        n_strings=config['model']['n_strings'],
        dropout=config['model']['dropout'],
        fusion_type=config['model'].get('fusion_type', 'cross_attention'),
        freeze_video_backbone=False
    )
    
    # Load weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"\n{'='*60}")
            print(f"[ERROR] Shape mismatch!")
            print(f"{'='*60}")
            print(f"Checkpoint architecture doesn't match current model.")
            print(f"\nOriginal error: {e}")
            print(f"{'='*60}\n")
            sys.exit(1)
        else:
            raise
    
    print(f"Model created with {model.count_parameters():,} parameters\n")
    
    # Move model to device
    model.to(device)
    
    # Load dataset
    print(f"Loading {args.split} dataset...")
    
    use_real_video = args.use_real_video and not args.dummy_video
    
    if use_real_video:
        # Use video dataset with real video frames
        own_sessions_dir = Path("data/own_sessions")
        
        if not own_sessions_dir.exists():
            print(f"Error: {own_sessions_dir} not found")
            return
        
        dataset = VideoFramesDataset(
            root_dir=str(own_sessions_dir),
            split=args.split,
            split_dir='splits',
            n_context_frames=7,
            fps=25,
            negative_ratio=1.0
        )
    else:
        # Use dummy video dataset (audio-only with random video)
        print("  Using dummy video (random frames)")
        own_sessions_dir = Path("data/own_sessions")
        
        if not own_sessions_dir.exists():
            print(f"Error: {own_sessions_dir} not found")
            return
        
        # Load audio-only dataset and add dummy video
        from data_loading.own_sessions_dataset import OwnSessionsDataset
        
        audio_dataset = OwnSessionsDataset(
            root_dir=str(own_sessions_dir),
            split=args.split,
            split_dir='splits',
            negative_ratio=1.0
        )
        
        # Wrap to add dummy video
        from torch.utils.data import TensorDataset
        
        # Create dummy video for all samples
        video_dummy = torch.randn(len(audio_dataset), 7, 3, 224, 224)
        
        dataset = TensorDataset(
            torch.stack([audio_dataset[i]['cqt'] for i in range(len(audio_dataset))]),
            video_dummy,
            torch.stack([audio_dataset[i]['onset'] for i in range(len(audio_dataset))]),
            torch.stack([audio_dataset[i]['pitch'] for i in range(len(audio_dataset))]),
            torch.stack([audio_dataset[i]['has_onset'] for i in range(len(audio_dataset))])
        )

    if len(dataset) == 0:
        print("\n[ERROR] Dataset is empty!")
        return

    # Create dataloader
    if use_real_video:
        # Use smaller batch size for own_sessions (max 64 due to video memory)
        batch_size = min(config['training']['batch_size'], 64)
        
        dataloader = create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config['training'].get('num_workers', 0),
            prefetch_factor=config['training'].get('prefetch_factor', 2),
            collate_fn=video_collate_fn,
            drop_last=False  # Don't drop last batch for evaluation!
        )
    else:
        # For dummy video (TensorDataset), use torch default_collate
        # Use smaller batch size for own_sessions (max 256)
        batch_size = min(config['training']['batch_size'], 256)
        
        from torch.utils.data import default_collate
        
        dataloader = create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config['training'].get('num_workers', 0),
            prefetch_factor=config['training'].get('prefetch_factor', 2),
            collate_fn=default_collate,  # Use torch default
            drop_last=False  # Don't drop last batch for evaluation!
        )

    print(f"Evaluation samples: {len(dataset)}\n")
    
    # Evaluate
    print(f"Evaluating on {args.split} split...")
    metrics, collector = evaluate_model(
        model,
        dataloader,
        device,
        onset_threshold=args.threshold,
        use_dummy_video=args.dummy_video
    )
    
    # Print summary
    print(collector.summary())
    
    # Print detailed metrics
    print(f"\n{'='*60}")
    print(f"Detailed Metrics")
    print(f"{'='*60}")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    print(f"{'='*60}\n")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'checkpoint': str(args.checkpoint),
            'epoch': checkpoint.get('epoch', 'unknown'),
            'split': args.split,
            'onset_threshold': args.threshold,
            'metrics': metrics
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_path}\n")
    
    # Save predictions for peak picking
    if args.save_predictions:
        save_path = Path(args.save_predictions)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save all predictions and ground truth
        predictions = {
            'onset_pred': [],
            'pitch_pred': [],
            'onset_true': [],
            'pitch_true': []
        }
        
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                cqt = batch['cqt'].to(device)
                cqt = cqt.unsqueeze(1)
                onset_true = batch['onset']
                pitch_true = batch['pitch']
                
                # Dummy video
                video = torch.randn(cqt.shape[0], 7, 3, 224, 224).to(device)
                
                onset_pred, pitch_pred = model(cqt, video)
                
                predictions['onset_pred'].append(onset_pred.cpu())
                predictions['pitch_pred'].append(pitch_pred.cpu())
                predictions['onset_true'].append(onset_true)
                predictions['pitch_true'].append(pitch_true)
        
        # Concatenate all batches
        predictions['onset_pred'] = torch.cat(predictions['onset_pred'], dim=0)
        predictions['pitch_pred'] = torch.cat(predictions['pitch_pred'], dim=0)
        predictions['onset_true'] = torch.cat(predictions['onset_true'], dim=0)
        predictions['pitch_true'] = torch.cat(predictions['pitch_true'], dim=0)
        
        # Save
        torch.save(predictions, save_path)
        print(f"Predictions saved to: {save_path}")
        print(f"  onset_pred: {predictions['onset_pred'].shape}")
        print(f"  pitch_pred: {predictions['pitch_pred'].shape}")
        print(f"  onset_true: {predictions['onset_true'].shape}")
        print(f"  pitch_true: {predictions['pitch_true'].shape}\n")


if __name__ == '__main__':
    main()

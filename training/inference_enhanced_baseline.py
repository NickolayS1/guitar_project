#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for enhanced baseline guitar transcription model.

Usage:
    python training/inference_enhanced_baseline.py --checkpoint experiments/enhanced_baseline/checkpoints/checkpoint_best.pth
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

from models.enhanced_baseline_cnn import EnhancedBaselineCNN
from data_loading.guitarset_frame_dataset import GuitarSetFrameDataset
from data_loading.own_sessions_dataset import OwnSessionsDataset
from data_loading.dataloader import create_dataloader
from evaluation.metrics import GuitarTranscriptionMetrics


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate_model(model, dataloader, device, onset_threshold=0.5):
    """Evaluate model on dataset."""
    model.eval()
    
    metrics_collector = GuitarTranscriptionMetrics(
        midi_min=36,  # C2
        midi_max=108  # C8
    )
    
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        # Move data to device
        cqt = batch['cqt'].to(device)
        cqt = cqt.unsqueeze(1)
        onset_true = batch['onset'].to(device)
        pitch_true = batch['pitch'].to(device)
        
        # Forward pass
        onset_pred, pitch_pred = model(cqt)
        
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
    parser = argparse.ArgumentParser(description='Evaluate enhanced baseline guitar transcription model')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/enhanced_baseline_config.yaml',
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
    parser.add_argument('--dataset', type=str, default='guitarset',
                        choices=['guitarset', 'own_sessions'],
                        help='Dataset to use (guitarset or own_sessions)')

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
    model = EnhancedBaselineCNN(
        n_strings=config['model']['n_strings'],
        dropout=config['model']['enhanced_baseline']['dropout']
    )
    
    # Load weights with error handling for shape mismatch
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"\n{'='*60}")
            print(f"[ERROR] Shape mismatch!")
            print(f"{'='*60}")
            print(f"This happens when checkpoint architecture doesn't match current model.")
            print(f"\nPossible causes:")
            print(f"  1. Config was changed (encoder_channels, head_hidden)")
            print(f"  2. Checkpoint is from old model architecture")
            print(f"\nSolutions:")
            print(f"  1. Re-train model with current config:")
            print(f"     python training/train_enhanced_baseline.py --config configs/enhanced_baseline_config.yaml")
            print(f"  2. Or restore original config parameters")
            print(f"\nOriginal error: {e}")
            print(f"{'='*60}\n")
            sys.exit(1)
        else:
            raise
    
    print(f"Model created with {model.count_parameters():,} parameters")
    
    # Move model to device
    model.to(device)
    
    # Load dataset
    print(f"Loading {args.split} dataset...")
    
    if args.dataset == 'own_sessions':
        # Use own sessions dataset (audio only, ignore video)
        own_sessions_dir = Path("data/own_sessions")
        
        if not own_sessions_dir.exists():
            print(f"Error: {own_sessions_dir} not found")
            return
        
        dataset = OwnSessionsDataset(
            root_dir=str(own_sessions_dir),
            split=args.split,
            split_dir='splits',
            negative_ratio=1.0
        )
    else:
        # Use GuitarSet dataset
        dataset = GuitarSetFrameDataset(
            root_dir=config['data']['root_dir'],
            split=args.split,
            negative_ratio=config['data'].get('negative_ratio', 1.0)
        )

    if len(dataset) == 0:
        print("\n[ERROR] Dataset is empty!")
        return

    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
        prefetch_factor=config['training'].get('prefetch_factor', 2)
    )

    print(f"Evaluation samples: {len(dataset)}\n")
    
    # Evaluate
    print(f"Evaluating on {args.split} split...")
    metrics, collector = evaluate_model(
        model,
        dataloader,
        device,
        onset_threshold=args.threshold
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
                
                onset_pred, pitch_pred = model(cqt)
                
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

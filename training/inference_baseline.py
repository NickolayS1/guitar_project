#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for baseline guitar transcription model.

Usage:
    python training/inference_baseline.py --checkpoint experiments/baseline/checkpoints/checkpoint_best.pth
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

from models.baseline_cnn import BaselineCNN
from data_loading.guitarset_frame_dataset import GuitarSetFrameDataset, create_dataloader
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
        midi_min=40,
        midi_max=103
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
    parser = argparse.ArgumentParser(description='Evaluate baseline guitar transcription model')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/baseline_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate (train/val/test)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Onset detection threshold')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results (optional)')
    
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
    model = BaselineCNN(
        n_strings=config['model']['n_strings'],
        dropout=config['model']['baseline']['dropout']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}\n")
    
    # Move model to device
    model.to(device)
    
    # Load dataset
    print(f"Loading {args.split} dataset...")
    dataset = GuitarSetFrameDataset(
        root_dir=config['data']['root_dir'],
        split=args.split,
        negative_ratio=1.0
    )
    
    if len(dataset) == 0:
        print("\n[ERROR] Dataset is empty!")
        return
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
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


if __name__ == '__main__':
    main()

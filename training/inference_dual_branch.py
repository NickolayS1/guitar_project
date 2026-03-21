#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for dual-branch guitar transcription model.

Usage:
    python training/inference_dual_branch.py --checkpoint experiments/dual_branch/checkpoints/checkpoint_best.pth
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

from models.dual_branch_cnn import DualBranchCNN, get_default_config
from data_loading.guitarset_frame_dataset import GuitarSetFrameDataset
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
    parser = argparse.ArgumentParser(description='Evaluate dual-branch guitar transcription model')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/dual_branch_config.yaml',
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
    
    # Create model with current config parameters
    print(f"Creating model with parameters:")
    print(f"  encoder_channels: {config['model']['dual_branch']['encoder_channels']}")
    print(f"  head_hidden: {config['model']['dual_branch']['head_hidden']}")
    print(f"  dropout: {config['model']['dual_branch']['dropout']}")
    print(f"  se_reduction: {config['model']['dual_branch']['se_reduction']}")
    
    model = DualBranchCNN(
        n_strings=config['model']['n_strings'],
        encoder_channels=config['model']['dual_branch']['encoder_channels'],
        head_hidden=config['model']['dual_branch']['head_hidden'],
        dropout=config['model']['dual_branch']['dropout'],
        se_reduction=config['model']['dual_branch']['se_reduction']
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
            print(f"     python training/train_dual_branch.py --config configs/dual_branch_config.yaml")
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
    
    # Debug: Print sample predictions
    print(f"\n{'='*60}")
    print(f"Debug: Sample predictions (first batch)")
    print(f"{'='*60}")
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        cqt = batch['cqt'].to(device)[:4]
        cqt = cqt.unsqueeze(1)
        onset_true = batch['onset'][:4]
        pitch_true = batch['pitch'][:4]
        
        onset_pred, pitch_pred = model(cqt)
        
        print(f"Onset True:  {onset_true.numpy()}")
        print(f"Onset Pred:  {onset_pred.cpu().numpy()}")
        print(f"Pitch True (norm):  {pitch_true.numpy()}")
        print(f"Pitch Pred (norm):  {pitch_pred.cpu().numpy()}")
        
        # Denormalize
        pitch_true_midi = pitch_true.numpy() * (103 - 40) + 40
        pitch_pred_midi = pitch_pred.cpu().numpy() * (103 - 40) + 40
        print(f"Pitch True (MIDI):  {pitch_true_midi}")
        print(f"Pitch Pred (MIDI):  {pitch_pred_midi}")
    print(f"{'='*60}\n")
    
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

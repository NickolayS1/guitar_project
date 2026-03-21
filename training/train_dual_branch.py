#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for dual-branch guitar transcription model.

Usage:
    python training/train_dual_branch.py --config configs/dual_branch_config.yaml
"""

import argparse
import os
import sys
import random
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from models.dual_branch_cnn import DualBranchCNN
from data_loading.guitarset_frame_dataset import GuitarSetFrameDataset
from data_loading.dataloader import create_dataloader
from training.losses import CombinedLoss
from training.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_epoch(model, dataloader, optimizer, loss_fn, device, grad_clip=None):
    """Train model for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_onset_loss = 0.0
    total_pitch_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        # Move data to device
        cqt = batch['cqt'].to(device)  # [B, 13, 72]
        cqt = cqt.unsqueeze(1)  # [B, 1, 13, 72]
        onset_true = batch['onset'].to(device)
        pitch_true = batch['pitch'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        onset_pred, pitch_pred = model(cqt)
        
        # Compute loss
        losses = loss_fn(onset_pred, pitch_pred, onset_true, pitch_true)
        loss = losses['total']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += losses['total'].item()
        total_onset_loss += losses['onset'].item()
        total_pitch_loss += losses['pitch'].item()
        n_batches += 1
    
    return {
        'train_loss': total_loss / n_batches,
        'train_onset_loss': total_onset_loss / n_batches,
        'train_pitch_loss': total_pitch_loss / n_batches
    }


@torch.no_grad()
def validate_epoch(model, dataloader, loss_fn, device):
    """Validate model for one epoch."""
    model.eval()
    
    total_loss = 0.0
    total_onset_loss = 0.0
    total_pitch_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        # Move data to device
        cqt = batch['cqt'].to(device)
        cqt = cqt.unsqueeze(1)
        onset_true = batch['onset'].to(device)
        pitch_true = batch['pitch'].to(device)
        
        # Forward pass
        onset_pred, pitch_pred = model(cqt)
        
        # Compute loss
        losses = loss_fn(onset_pred, pitch_pred, onset_true, pitch_true)
        
        # Accumulate losses
        total_loss += losses['total'].item()
        total_onset_loss += losses['onset'].item()
        total_pitch_loss += losses['pitch'].item()
        n_batches += 1
    
    return {
        'val_loss': total_loss / n_batches,
        'val_onset_loss': total_onset_loss / n_batches,
        'val_pitch_loss': total_pitch_loss / n_batches
    }


def main():
    parser = argparse.ArgumentParser(description='Train dual-branch guitar transcription model')
    parser.add_argument('--config', '-c', type=str, default='configs/dual_branch_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.device:
        config['experiment']['device'] = args.device
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Set seed
    set_seed(config['experiment']['seed'])
    
    # Set device
    device = torch.device(config['experiment']['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    
    print(f"\n{'='*60}")
    print(f"Guitar Transcription Training - Dual-Branch CNN")
    print(f"{'='*60}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = GuitarSetFrameDataset(
        root_dir=config['data']['root_dir'],
        split='train',
        negative_ratio=1.0
    )
    
    val_dataset = GuitarSetFrameDataset(
        root_dir=config['data']['root_dir'],
        split='val',
        negative_ratio=1.0
    )
    
    if len(train_dataset) == 0:
        print("\n[ERROR] Train dataset is empty!")
        print("Make sure data/guitarset/splits/train.txt exists")
        return
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 0),
        prefetch_factor=config['training'].get('prefetch_factor', 2)
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
        prefetch_factor=config['training'].get('prefetch_factor', 2)
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}\n")
    
    # Create model
    print("Creating model...")
    model = DualBranchCNN(
        n_strings=config['model']['n_strings'],
        encoder_channels=config['model']['dual_branch']['encoder_channels'],
        head_hidden=config['model']['dual_branch']['head_hidden'],
        dropout=config['model']['dual_branch']['dropout'],
        se_reduction=config['model']['dual_branch']['se_reduction']
    )
    
    model = model.to(device)
    
    # Count parameters
    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}\n")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create loss function
    loss_fn = CombinedLoss(
        onset_weight=config['training']['loss']['onset_weight'],
        pitch_weight=config['training']['loss']['pitch_weight'],
        onset_pos_weight=config['training']['loss'].get('onset_pos_weight', 2.0)
    )
    
    # Create callbacks
    checkpoint_dir = Path(config['training']['checkpoint']['save_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(config['training']['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            mode=config['training']['early_stopping']['mode']
        ),
        ModelCheckpoint(
            checkpoint_dir=str(checkpoint_dir),
            mode=config['training']['early_stopping']['mode'],
            save_best_only=config['training']['checkpoint']['save_best'],
            save_last=config['training']['checkpoint']['save_last']
        ),
        LearningRateScheduler(
            optimizer,
            scheduler_type=config['training']['scheduler']['type'],
            **config['training']['scheduler']
        )
    ]
    
    # Training loop
    print(f"Starting training for {config['training']['epochs']} epochs...\n")
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_onset_loss': [],
        'val_onset_loss': [],
        'train_pitch_loss': [],
        'val_pitch_loss': []
    }
    
    use_tqdm = config['training']['logging'].get('use_tqdm', True)
    validation_interval = config['training']['validation']['interval']
    
    for epoch in range(1, config['training']['epochs'] + 1):
        start_time = time.time()
        
        # Train
        if use_tqdm:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{config["training"]["epochs"]}')
            for batch in pbar:
                cqt = batch['cqt'].to(device)
                cqt = cqt.unsqueeze(1)
                onset_true = batch['onset'].to(device)
                pitch_true = batch['pitch'].to(device)
                
                optimizer.zero_grad()
                onset_pred, pitch_pred = model(cqt)
                losses = loss_fn(onset_pred, pitch_pred, onset_true, pitch_true)
                loss = losses['total']
                loss.backward()
                
                if config['training']['grad_clip']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
                
                optimizer.step()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'onset': f'{losses["onset"].item():.4f}',
                    'pitch': f'{losses["pitch"].item():.4f}'
                })
        else:
            print(f'Epoch {epoch}/{config["training"]["epochs"]}')
            train_epoch(model, train_loader, optimizer, loss_fn, device, config['training']['grad_clip'])
        
        # Validate
        if epoch % validation_interval == 0:
            val_metrics = validate_epoch(model, val_loader, loss_fn, device)
            
            # Update history
            for key, value in val_metrics.items():
                if key in history:
                    history[key].append(value)
            
            # Check for improvement
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                if config['training']['checkpoint']['save_best']:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_metrics['val_loss'],
                    }, checkpoint_dir / 'checkpoint_best.pth')
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch} summary:")
        print(f"  Time: {epoch_time:.1f}s")
        if epoch % validation_interval == 0:
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val Onset: {val_metrics['val_onset_loss']:.4f}")
            print(f"  Val Pitch: {val_metrics['val_pitch_loss']:.4f}")
        print()
        
        # Call callbacks
        for callback in callbacks:
            callback.on_epoch_end({
                'epoch': epoch,
                'model': model,
                'optimizer': optimizer,
                'metrics': val_metrics if epoch % validation_interval == 0 else {}
            })
        
        # Check for early stopping
        for callback in callbacks:
            if isinstance(callback, EarlyStopping) and callback.should_stop:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        else:
            continue
        break
    
    # Save final checkpoint
    if config['training']['checkpoint']['save_last']:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_dir / 'checkpoint_last.pth')
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir.absolute()}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

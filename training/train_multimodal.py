#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for multimodal guitar transcription.

Usage:
    python onsets/training/train_multimodal.py \
        --config onsets/configs/multimodal_config.yaml
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

from models.multimodal_cnn import MultimodalCNN
from data_loading.video_dataset import VideoFramesDataset, collate_fn as video_collate_fn
from data_loading.dataloader import create_dataloader
from training.losses import CombinedLoss
from training.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler


def set_seed(seed: int):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_dummy_batch(device: torch.device):
    """Create dummy batch for testing."""
    audio = torch.randn(4, 1, 13, 72).to(device)
    video = torch.randn(4, 7, 3, 224, 224).to(device)
    onset = torch.randint(0, 2, (4, 6)).float().to(device)
    pitch = torch.rand(4, 6).to(device)
    
    return {
        'cqt': audio.squeeze(1),  # [B, 13, 72]
        'onset': onset,
        'pitch': pitch
    }, video


def train_epoch(model, dataloader, optimizer, loss_fn, device, grad_clip=None):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_onset_loss = 0.0
    total_pitch_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        # Handle both dict and tuple batches
        if isinstance(batch, dict):
            audio = batch['cqt'].to(device).unsqueeze(1)
            onset_true = batch['onset'].to(device)
            pitch_true = batch['pitch'].to(device)
        else:  # tuple/list
            audio = batch[0].to(device).unsqueeze(1)
            onset_true = batch[1].to(device)
            pitch_true = batch[2].to(device)
        
        # Dummy video (replace with real video loading)
        video = torch.randn(audio.shape[0], 7, 3, 224, 224).to(device)
        
        # Forward
        optimizer.zero_grad()
        onset_pred, pitch_pred = model(audio, video)
        
        # Loss
        losses = loss_fn(onset_pred, pitch_pred, onset_true, pitch_true)
        loss = losses['total']
        
        # Backward
        loss.backward()
        
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
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
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0.0
    total_onset_loss = 0.0
    total_pitch_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        # Get audio and video from batch
        audio = batch['cqt'].to(device).unsqueeze(1)
        video = batch['video'].to(device)
        onset_true = batch['onset'].to(device)
        pitch_true = batch['pitch'].to(device)
        
        onset_pred, pitch_pred = model(audio, video)
        losses = loss_fn(onset_pred, pitch_pred, onset_true, pitch_true)
        
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
    parser = argparse.ArgumentParser(description='Train multimodal model')
    parser.add_argument('--config', '-c', type=str, default='configs/multimodal_config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--audio-checkpoint', type=str, default=None,
                        help='Path to pretrained audio checkpoint (enhanced_baseline)')
    parser.add_argument('--freeze-audio', action='store_true',
                        help='Freeze audio branch during training')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    # Override
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
    
    # Device
    device = torch.device(config['experiment']['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    
    print(f"\n{'='*60}")
    print(f"Multimodal Guitar Transcription Training")
    print(f"{'='*60}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Create datasets
    print("Loading datasets...")
    
    # Use video dataset for multimodal training
    own_sessions_dir = Path("data/own_sessions")
    
    if own_sessions_dir.exists():
        train_dataset = VideoFramesDataset(
            root_dir=str(own_sessions_dir),
            split='train',
            split_dir='splits',
            n_context_frames=7,
            fps=25,
            negative_ratio=config['training'].get('negative_ratio', 3.0)
        )
        
        val_dataset = VideoFramesDataset(
            root_dir=str(own_sessions_dir),
            split='val',
            split_dir='splits',
            n_context_frames=7,
            fps=25,
            negative_ratio=config['training'].get('negative_ratio', 3.0)
        )
        
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
    else:
        # Fallback to dummy dataset
        print("  Warning: own_sessions not found, using dummy dataset")
        from torch.utils.data import TensorDataset
        audio_dummy = torch.randn(200, 13, 72)
        video_dummy = torch.randn(200, 7, 3, 224, 224)
        onset_dummy = torch.randint(0, 2, (200, 6)).float()
        pitch_dummy = torch.rand(200, 6)
        
        train_dataset = TensorDataset(audio_dummy[:160], video_dummy[:160], onset_dummy[:160], pitch_dummy[:160])
        val_dataset = TensorDataset(audio_dummy[160:], video_dummy[160:], onset_dummy[160:], pitch_dummy[160:])
        print(f"  Train: {len(train_dataset)} samples (dummy)")
        print(f"  Val: {len(val_dataset)} samples (dummy)")
    
    # Create dataloaders with video collate_fn
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 0),
        prefetch_factor=config['training'].get('prefetch_factor', 2),
        collate_fn=video_collate_fn
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
        prefetch_factor=config['training'].get('prefetch_factor', 2),
        collate_fn=video_collate_fn
    )
    
    print()
    
    # Create model
    print("Creating model...")
    model = MultimodalCNN(
        n_strings=config['model']['n_strings'],
        dropout=config['model']['dropout'],
        fusion_type=config['model'].get('fusion_type', 'cross_attention'),
        freeze_video_backbone=True  # Video backbone frozen by default
    )
    
    # Load pretrained audio weights if specified
    if args.audio_checkpoint:
        model.load_audio_weights(args.audio_checkpoint)
        # Don't freeze audio - allow fine-tuning
        if not args.freeze_audio:
            model.unfreeze_audio_branch()
            print("Audio branch UNFROZEN (fine-tuning enabled)")
    else:
        # Optionally freeze audio branch
        if args.freeze_audio:
            model.freeze_audio_branch()
            print("Audio branch frozen")
    
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}\n")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    scheduler_config = config['training']['scheduler']
    if scheduler_config['type'] == 'one_cycle':
        # Dummy dataloader for steps_per_epoch
        steps_per_epoch = 100  # Will be updated with real dataloader
        total_steps = steps_per_epoch * scheduler_config['epochs']
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_config['max_lr'],
            total_steps=total_steps,
            pct_start=scheduler_config.get('pct_start', 0.3),
            anneal_strategy=scheduler_config.get('anneal_strategy', 'cos'),
            cycle_momentum=False
        )
        use_scheduler = True
    else:
        scheduler = None
        use_scheduler = False
    
    # Loss
    loss_fn = CombinedLoss(
        onset_weight=config['training']['loss']['onset_weight'],
        pitch_weight=config['training']['loss']['pitch_weight'],
        onset_pos_weight=config['training']['loss'].get('onset_pos_weight', 2.0)
    )
    
    # Checkpointing
    checkpoint_dir = Path(config['training']['checkpoint']['save_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"Starting training for {config['training']['epochs']} epochs...\n")
    
    best_val_loss = float('inf')
    use_tqdm = config['training']['logging'].get('use_tqdm', True)
    
    for epoch in range(1, config['training']['epochs'] + 1):
        start_time = time.time()
        
        # Train
        if use_tqdm:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{config["training"]["epochs"]}')
            for batch in pbar:
                # Get audio and video from batch
                audio = batch['cqt'].to(device).unsqueeze(1)  # [B, 13, 72] → [B, 1, 13, 72]
                video = batch['video'].to(device)  # [B, 7, 3, 224, 224]
                onset_true = batch['onset'].to(device)
                pitch_true = batch['pitch'].to(device)
                
                optimizer.zero_grad()
                onset_pred, pitch_pred = model(audio, video)
                losses = loss_fn(onset_pred, pitch_pred, onset_true, pitch_true)
                loss = losses['total']
                loss.backward()
                
                if config['training']['grad_clip']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
                
                optimizer.step()
                
                if use_scheduler:
                    scheduler.step()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'onset': f'{losses["onset"].item():.4f}',
                    'pitch': f'{losses["pitch"].item():.4f}'
                })
        
        # Validate
        if epoch % config['training']['validation']['interval'] == 0:
            val_metrics = validate_epoch(model, val_loader, loss_fn, device)
            
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                
                if config['training']['checkpoint']['save_best']:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_metrics['val_loss'],
                    }, checkpoint_dir / 'checkpoint_best.pth')
        
        # Print summary
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch} summary:")
        print(f"  Time: {epoch_time:.1f}s")
        if epoch % config['training']['validation']['interval'] == 0:
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print()
    
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
    print(f"Checkpoints: {checkpoint_dir.absolute()}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

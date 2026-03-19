# -*- coding: utf-8 -*-
"""
Training callbacks for guitar transcription.
"""

import torch
from pathlib import Path


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
        
        if mode == 'min':
            self.is_better = lambda new, best: new < best - min_delta
        else:
            self.is_better = lambda new, best: new > best + min_delta
    
    def on_epoch_end(self, state: dict):
        current = state['metrics'].get('val_loss')
        if current is None:
            return
        
        if self.best_value is None or self.is_better(current, self.best_value):
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


class ModelCheckpoint:
    """Model checkpoint callback."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        mode: str = 'min',
        save_best_only: bool = True,
        save_last: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.best_value = None
        
        self.is_better = lambda x, y: x < y if mode == 'min' else x > y
    
    def on_epoch_end(self, state: dict):
        current = state['metrics'].get('val_loss')
        
        # Save last checkpoint
        if self.save_last:
            checkpoint_path = self.checkpoint_dir / 'checkpoint_last.pth'
            torch.save({
                'epoch': state['epoch'],
                'model_state_dict': state['model'].state_dict(),
                'optimizer_state_dict': state['optimizer'].state_dict(),
                'metrics': state['metrics'],
            }, checkpoint_path)
        
        # Save best checkpoint
        if self.save_best_only and current is not None:
            if self.best_value is None or self.is_better(current, self.best_value):
                self.best_value = current
                checkpoint_path = self.checkpoint_dir / 'checkpoint_best.pth'
                torch.save({
                    'epoch': state['epoch'],
                    'model_state_dict': state['model'].state_dict(),
                    'optimizer_state_dict': state['optimizer'].state_dict(),
                    'val_loss': current,
                }, checkpoint_path)


class LearningRateScheduler:
    """Learning rate scheduler callback."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = 'reduce_on_plateau',
        **kwargs
    ):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 5),
                min_lr=kwargs.get('min_lr', 1e-6),
                # verbose=True
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        else:
            self.scheduler = None
    
    def on_epoch_end(self, state: dict):
        if self.scheduler_type == 'reduce_on_plateau':
            current = state['metrics'].get('val_loss')
            if current is not None:
                self.scheduler.step(current)
        elif self.scheduler is not None:
            self.scheduler.step()

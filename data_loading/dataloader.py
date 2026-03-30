# -*- coding: utf-8 -*-
"""
DataLoader utilities for guitar transcription.
"""

from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import collate_fn as default_collate_fn


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
    collate_fn: Optional[callable] = None,
    prefetch_factor: int = 2
) -> DataLoader:
    """
    Create a DataLoader for guitar transcription dataset.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        collate_fn: Custom collate function (uses default if None)
        prefetch_factor: Number of batches loaded in advance by each worker
    
    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn or default_collate_fn,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )


def create_train_val_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2
) -> Dict[str, DataLoader]:
    """
    Create train and validation DataLoaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Number of batches loaded in advance
    
    Returns:
        Dictionary with 'train' and 'val' DataLoaders
    """
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=prefetch_factor
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        prefetch_factor=prefetch_factor
    )
    
    return {
        'train': train_loader,
        'val': val_loader
    }

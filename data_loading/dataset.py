# -*- coding: utf-8 -*-
"""
Dataset utilities for guitar transcription.
"""

import torch


def collate_fn(batch):
    """
    Collate function for DataLoader.
    
    Batches samples with same CQT window size.
    """
    return {
        'cqt': torch.stack([item['cqt'] for item in batch]),  # [B, total_frames, n_bins]
        'onset': torch.stack([item['onset'] for item in batch]),  # [B, 6]
        'pitch': torch.stack([item['pitch'] for item in batch]),  # [B, 6]
        'has_onset': torch.tensor([item['has_onset'] for item in batch]),  # [B]
        'time_sec': [item['time_sec'] for item in batch]
    }

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
    # Handle both dict formats (with or without time_sec)
    sample_keys = batch[0].keys()
    
    # Stack tensors properly
    result = {
        'cqt': torch.stack([item['cqt'] for item in batch]),  # [B, total_frames, n_bins]
        'onset': torch.stack([item['onset'] for item in batch]),  # [B, 6]
        'pitch': torch.stack([item['pitch'] for item in batch]),  # [B, 6]
    }
    
    # Handle has_onset - could be bool or tensor
    has_onset_list = []
    for item in batch:
        val = item.get('has_onset', True)
        if isinstance(val, bool):
            has_onset_list.append(1 if val else 0)
        elif isinstance(val, torch.Tensor):
            has_onset_list.append(val.item())
        else:
            has_onset_list.append(int(val))
    result['has_onset'] = torch.tensor(has_onset_list)  # [B]
    
    # Add time_sec if present
    if 'time_sec' in sample_keys:
        result['time_sec'] = [item['time_sec'] for item in batch]
    
    return result

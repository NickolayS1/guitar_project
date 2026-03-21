# -*- coding: utf-8 -*-
"""
Loss functions for guitar transcription.

Combined loss = BCE(onset) + MSE(pitch | active strings)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """
    Combined loss for guitar transcription.
    
    Total Loss = λ_onset * WeightedBCE(onset) + λ_pitch * MSE(pitch | active)
    """
    
    def __init__(self, onset_weight: float = 1.0, pitch_weight: float = 1.0, onset_pos_weight: float = 2.0):
        super().__init__()
        self.onset_weight = onset_weight
        self.pitch_weight = pitch_weight
        self.onset_pos_weight = onset_pos_weight  # Weight for positive class
    
    def forward(
        self,
        onset_pred: torch.Tensor,
        pitch_pred: torch.Tensor,
        onset_true: torch.Tensor,
        pitch_true: torch.Tensor,
        mask: torch.Tensor = None
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            onset_pred: [B, 6] - predicted onset probabilities
            pitch_pred: [B, 6] - predicted normalized MIDI
            onset_true: [B, 6] - ground truth onsets
            pitch_true: [B, 6] - ground truth normalized MIDI
            mask: [B, 6] - active strings mask (uses onset_true if None)
        
        Returns:
            Dictionary with total, onset, pitch losses
        """
        if mask is None:
            mask = onset_true
        
        # Weighted BCE for onset detection (handle class imbalance)
        # pos_weight > 1 increases penalty for missing positive onsets
        loss_onset = F.binary_cross_entropy(
            onset_pred, 
            onset_true, 
            reduction='none'
        )
        
        # Apply class weighting: positive examples get onset_pos_weight
        weights = onset_true * (self.onset_pos_weight - 1) + 1
        loss_onset = (loss_onset * weights).mean()
        
        # MSE for pitch (ONLY for active strings!)
        # Zero out loss for inactive strings
        pitch_diff = (pitch_pred - pitch_true) ** 2
        masked_pitch_loss = pitch_diff * mask  # Zero where mask=0
        
        # Normalize by number of active strings
        n_active = mask.sum() + 1e-8
        loss_pitch = masked_pitch_loss.sum() / n_active
        
        # Combined loss
        total_loss = self.onset_weight * loss_onset + self.pitch_weight * loss_pitch
        
        return {
            'total': total_loss,
            'onset': loss_onset,
            'pitch': loss_pitch
        }

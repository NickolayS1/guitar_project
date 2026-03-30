# -*- coding: utf-8 -*-
"""
Evaluation metrics for guitar transcription.
"""

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error, mean_squared_error


class GuitarTranscriptionMetrics:
    """
    Metrics collector for guitar transcription.
    
    Computes:
    - Onset: Precision, Recall, F1 (per string)
    - Pitch: MAE, RMSE (per string, only for correct onsets)
    - Combined score
    """
    
    def __init__(self, midi_min: int = 36, midi_max: int = 108, onset_tolerance_ms: float = 50):
        self.midi_min = midi_min  # C2
        self.midi_max = midi_max  # C8
        self.onset_tolerance_ms = onset_tolerance_ms
        
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.all_pred_onsets = []
        self.all_true_onsets = []
        self.all_pred_midi = []
        self.all_true_midi = []
        self.all_masks = []
    
    def update_batch(
        self,
        onset_pred: torch.Tensor,
        pitch_pred: torch.Tensor,
        onset_true: torch.Tensor,
        pitch_true: torch.Tensor,
        mask: torch.Tensor,
        onset_threshold: float = 0.5
    ):
        """
        Update metrics from batch predictions.
        """
        B = onset_pred.shape[0]

        # Convert to numpy
        onset_pred_np = onset_pred.cpu().numpy()
        onset_true_np = onset_true.cpu().numpy()
        pitch_pred_np = pitch_pred.cpu().numpy()
        pitch_true_np = pitch_true.cpu().numpy()

        for b in range(B):
            # Get sample (handle both 1D and 2D)
            if onset_true_np.ndim == 1:
                onset_true_sample = onset_true_np[b]
                pitch_true_sample = pitch_true_np[b]
                pitch_pred_sample = pitch_pred_np[b]
                onset_pred_sample = onset_pred_np[b]
            else:
                onset_true_sample = onset_true_np[b]
                pitch_true_sample = pitch_true_np[b]
                pitch_pred_sample = pitch_pred_np[b]
                onset_pred_sample = onset_pred_np[b]
            
            # Binarize predictions
            onset_binary = (onset_pred_sample > onset_threshold).astype(float)

            # Store for onset metrics
            self.all_pred_onsets.append(onset_binary)
            self.all_true_onsets.append(onset_true_sample)

            # Denormalize pitch
            pred_midi = self._denormalize_pitch(pitch_pred_sample)
            true_midi = self._denormalize_pitch(pitch_true_sample)
            
            # For pitch metrics: use onset_true as mask
            # Ignore NaN values in true_midi (negative samples)
            active_mask = (onset_true_sample > 0.5) & (~np.isnan(true_midi))
            
            self.all_pred_midi.append(pred_midi * active_mask)
            self.all_true_midi.append(true_midi * active_mask)
            self.all_masks.append(active_mask)
    
    def _denormalize_pitch(self, normalized: np.ndarray) -> np.ndarray:
        """Denormalize pitch from [0, 1] to MIDI."""
        return normalized * (self.midi_max - self.midi_min) + self.midi_min
    
    def compute_all(self) -> dict:
        """Compute all metrics."""
        metrics = {}

        # Convert to arrays - ensure 2D
        pred_onsets = np.array(self.all_pred_onsets)
        true_onsets = np.array(self.all_true_onsets)
        pred_midi = np.array(self.all_pred_midi)
        true_midi = np.array(self.all_true_midi)
        masks = np.array(self.all_masks)
        
        # Ensure 2D arrays
        if pred_onsets.ndim == 1:
            pred_onsets = pred_onsets.reshape(-1, 6)
        if true_onsets.ndim == 1:
            true_onsets = true_onsets.reshape(-1, 6)
        if pred_midi.ndim == 1:
            pred_midi = pred_midi.reshape(-1, 6)
        if true_midi.ndim == 1:
            true_midi = true_midi.reshape(-1, 6)
        if masks.ndim == 1:
            masks = masks.reshape(-1, 6)

        # Onset metrics (per string)
        onset_metrics = {}
        for s in range(6):
            # Check if there are any positive samples for this string
            if np.sum(true_onsets[:, s]) == 0:
                # No positive samples - skip this string
                onset_metrics[f'onset_precision_s{s}'] = 0.0
                onset_metrics[f'onset_recall_s{s}'] = 0.0
                onset_metrics[f'onset_f1_s{s}'] = 0.0
            else:
                p, r, f1, _ = precision_recall_fscore_support(
                    true_onsets[:, s],
                    pred_onsets[:, s],
                    average='binary',
                    zero_division=0
                )
                onset_metrics[f'onset_precision_s{s}'] = p
                onset_metrics[f'onset_recall_s{s}'] = r
                onset_metrics[f'onset_f1_s{s}'] = f1

        # Mean across strings
        onset_metrics['onset_precision'] = np.mean([onset_metrics[f'onset_precision_s{s}'] for s in range(6)])
        onset_metrics['onset_recall'] = np.mean([onset_metrics[f'onset_recall_s{s}'] for s in range(6)])
        onset_metrics['onset_f1'] = np.mean([onset_metrics[f'onset_f1_s{s}'] for s in range(6)])

        metrics.update(onset_metrics)
        
        # Pitch metrics (only for correct onsets)
        pitch_metrics = {}
        for s in range(6):
            active_mask = masks[:, s] > 0.5
            if active_mask.sum() > 0:
                mae = mean_absolute_error(true_midi[active_mask, s], pred_midi[active_mask, s])
                rmse = np.sqrt(mean_squared_error(true_midi[active_mask, s], pred_midi[active_mask, s]))
            else:
                mae = 0.0
                rmse = 0.0
            
            pitch_metrics[f'pitch_mae_s{s}'] = mae
            pitch_metrics[f'pitch_rmse_s{s}'] = rmse
        
        # Mean across strings
        pitch_metrics['pitch_mae'] = np.mean([pitch_metrics[f'pitch_mae_s{s}'] for s in range(6)])
        pitch_metrics['pitch_rmse'] = np.mean([pitch_metrics[f'pitch_rmse_s{s}'] for s in range(6)])
        
        metrics.update(pitch_metrics)

        # Combined score (clamped to [0, 1])
        # Prevent negative values when pitch_mae > 12
        pitch_penalty = min(metrics['pitch_mae'] / 12, 1.0)
        metrics['combined_score'] = metrics['onset_f1'] * (1 - pitch_penalty)

        return metrics
    
    def summary(self) -> str:
        """Return formatted summary string."""
        metrics = self.compute_all()
        
        lines = [
            "=" * 60,
            "Guitar Transcription Metrics Summary",
            "=" * 60,
            f"Onset Detection:",
            f"  Precision: {metrics.get('onset_precision', 0):.3f}",
            f"  Recall:    {metrics.get('onset_recall', 0):.3f}",
            f"  F1-Score:  {metrics.get('onset_f1', 0):.3f}",
            "",
            f"Pitch Estimation:",
            f"  MAE:  {metrics.get('pitch_mae', 0):.2f} semitones",
            f"  RMSE: {metrics.get('pitch_rmse', 0):.2f} semitones",
            "",
            f"Combined Score: {metrics.get('combined_score', 0):.3f}",
            "=" * 60
        ]
        
        return "\n".join(lines)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Peak Picking and Evaluation for Guitar Transcription.

This module implements:
1. Peak picking for onset detection (local maximum + threshold)
2. MIDI pitch rounding to nearest integer
3. Window-based evaluation metrics (not frame-exact matching)

Usage:
    # Evaluate with peak picking
    python evaluation/peak_picking.py \
        --predictions predictions.pt \
        --ground_truth ground_truth.pt \
        --threshold 0.5 \
        --window_size 3
"""

import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch


def round_midi(pitch_pred: np.ndarray) -> np.ndarray:
    """Round MIDI pitch predictions to nearest integer."""
    return np.round(pitch_pred).astype(int)


def peak_picking_1d(
    onset_pred: np.ndarray,
    threshold: float = 0.5,
    window_size: int = 3
) -> np.ndarray:
    """
    1D peak picking for onset detection.
    
    A frame is considered an onset if:
    1. Its probability > threshold
    2. Its probability > all neighbors in window
    """
    T = len(onset_pred)
    onset_binary = np.zeros(T, dtype=float)
    half_window = window_size // 2
    
    for t in range(T):
        if onset_pred[t] <= threshold:
            continue
        
        start = max(0, t - half_window)
        end = min(T, t + half_window + 1)
        
        is_peak = True
        for t_neighbor in range(start, end):
            if t_neighbor != t and onset_pred[t_neighbor] > onset_pred[t]:
                is_peak = False
                break
        
        if is_peak:
            onset_binary[t] = 1.0
    
    return onset_binary


def peak_picking(
    onset_pred: np.ndarray,
    pitch_pred: np.ndarray,
    threshold: float = 0.5,
    window_size: int = 3,
    round_midi_values: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Peak picking for multi-string guitar transcription."""
    T, n_strings = onset_pred.shape
    
    onset_binary = np.zeros((T, n_strings), dtype=float)
    pitch_rounded = pitch_pred.copy()  # Don't round yet! Keep normalized [0, 1]
    
    for s in range(n_strings):
        onset_binary[:, s] = peak_picking_1d(
            onset_pred[:, s],
            threshold=threshold,
            window_size=window_size
        )
        
        # Don't round MIDI here - round after denormalization!
        # if round_midi_values:
        #     pitch_rounded[:, s] = round_midi(pitch_pred[:, s])
    
    return onset_binary, pitch_rounded


def evaluate_with_peak_picking(
    onset_pred: np.ndarray,
    pitch_pred: np.ndarray,
    onset_true: np.ndarray,
    pitch_true: np.ndarray,
    threshold: float = 0.5,
    window_size: int = 3,
    onset_tolerance_ms: float = 50,
    midi_min: int = 36,
    midi_max: int = 108,
    hop_length: int = 512,
    sr: int = 22050
) -> Dict[str, float]:
    """
    Evaluate predictions with peak picking using window-based metrics.
    """
    # Apply peak picking
    onset_binary, pitch_rounded = peak_picking(
        onset_pred,
        pitch_pred,
        threshold=threshold,
        window_size=window_size,
        round_midi_values=True
    )

    # Denormalize pitch THEN round to nearest integer
    # pitch_pred is normalized [0, 1], denormalize to MIDI [36, 108], then round
    pitch_midi = round_midi(pitch_rounded * (midi_max - midi_min) + midi_min)
    
    # CRITICAL FIX: For inactive strings (onset_true=0), pitch_true=0 which denormalizes to midi_min (36)
    # This creates huge errors when comparing to predictions. Set to NaN to ignore.
    pitch_true_midi = np.full_like(pitch_true, np.nan, dtype=float)
    active_mask = onset_true > 0.5
    pitch_true_midi[active_mask] = round_midi(pitch_true[active_mask] * (midi_max - midi_min) + midi_min)
    
    # Compute metrics
    metrics = {}
    frame_duration_ms = hop_length / sr * 1000
    tolerance_frames = int(np.round(onset_tolerance_ms / frame_duration_ms))
    T, n_strings = onset_true.shape
    
    # Count true onsets per string
    true_onset_events = {s: [] for s in range(6)}
    for s in range(6):
        active_frames = np.where(onset_true[:, s] > 0.5)[0]
        if len(active_frames) == 0:
            continue
        
        events = []
        current_event = [active_frames[0]]
        for i in range(1, len(active_frames)):
            if active_frames[i] - active_frames[i-1] <= 2:
                current_event.append(active_frames[i])
            else:
                events.append(current_event)
                current_event = [active_frames[i]]
        events.append(current_event)
        
        for event in events:
            center_frame = event[len(event)//2]
            start = max(0, center_frame - 2)
            end = min(T, center_frame + 2)
            # Use nanmean to ignore NaN values (inactive strings)
            midi_val = int(np.round(np.nanmean(pitch_true_midi[start:end, s])))
            true_onset_events[s].append({
                'center': center_frame,
                'start': center_frame - tolerance_frames,
                'end': center_frame + tolerance_frames,
                'midi': midi_val
            })
    
    # Count detected onsets per string
    detected_onset_events = {s: [] for s in range(6)}
    for s in range(6):
        pred_frames = np.where(onset_binary[:, s] > 0.5)[0]
        if len(pred_frames) == 0:
            continue
        
        events = []
        current_event = [pred_frames[0]]
        for i in range(1, len(pred_frames)):
            if pred_frames[i] - pred_frames[i-1] <= 2:
                current_event.append(pred_frames[i])
            else:
                events.append(current_event)
                current_event = [pred_frames[i]]
        events.append(current_event)
        
        for event in events:
            center_frame = event[len(event)//2]
            start = max(0, center_frame - 2)
            end = min(T, center_frame + 2)
            midi_val = int(np.round(np.mean(pitch_midi[start:end, s])))
            detected_onset_events[s].append({
                'center': center_frame,
                'midi': midi_val
            })
    
    # Match detected onsets to true onsets (per string)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    pitch_errors = []
    
    for s in range(6):
        true_events = true_onset_events[s]
        detected_events = detected_onset_events[s]
        
        matched_true = set()
        matched_detected = set()
        
        # Match detected to true
        for i, det in enumerate(detected_events):
            for j, true in enumerate(true_events):
                if j in matched_true:
                    continue
                
                if true['start'] <= det['center'] <= true['end']:
                    matched_true.add(j)
                    matched_detected.add(i)
                    total_tp += 1
                    
                    # Compare MIDI values
                    midi_error = abs(det['midi'] - true['midi'])
                    pitch_errors.append(midi_error)
                    break
        
        total_fp += len(detected_events) - len(matched_detected)
        total_fn += len(true_events) - len(matched_true)
        
        tp = len(matched_detected)
        fp = len(detected_events) - len(matched_detected)
        fn = len(true_events) - len(matched_true)
        
        precision_s = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_s = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_s = 2 * precision_s * recall_s / (precision_s + recall_s) if (precision_s + recall_s) > 0 else 0
        
        metrics[f'onset_precision_s{s}'] = precision_s
        metrics[f'onset_recall_s{s}'] = recall_s
        metrics[f'onset_f1_s{s}'] = f1_s
    
    # Overall onset metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics['onset_precision'] = precision
    metrics['onset_recall'] = recall
    metrics['onset_f1'] = f1
    
    # Pitch metrics (only for matched onsets)
    if len(pitch_errors) > 0:
        metrics['pitch_mae'] = np.mean(pitch_errors)
        metrics['pitch_rmse'] = np.sqrt(np.mean(np.square(pitch_errors)))
    else:
        metrics['pitch_mae'] = 0.0
        metrics['pitch_rmse'] = 0.0
    
    # Per-string pitch MAE (only for matched onsets)
    for s in range(6):
        string_errors = []
        true_events = true_onset_events[s]
        detected_events = detected_onset_events[s]
        
        for i, det in enumerate(detected_events):
            for j, true in enumerate(true_events):
                if true['start'] <= det['center'] <= true['end']:
                    midi_error = abs(det['midi'] - true['midi'])
                    string_errors.append(midi_error)
                    break
        
        metrics[f'pitch_mae_s{s}'] = np.mean(string_errors) if len(string_errors) > 0 else 0.0
    
    # Combined score (clamped to [0, 1])
    pitch_penalty = min(metrics['pitch_mae'] / 12, 1.0)
    metrics['combined_score'] = metrics['onset_f1'] * (1 - pitch_penalty)
    
    return metrics


def find_best_threshold(
    onset_pred: np.ndarray,
    pitch_pred: np.ndarray,
    onset_true: np.ndarray,
    pitch_true: np.ndarray,
    threshold_range: List[float] = None,
    window_size: int = 3,
    midi_min: int = 36,
    midi_max: int = 108
) -> Tuple[float, Dict[str, float]]:
    """Find best peak picking threshold via grid search."""
    if threshold_range is None:
        threshold_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    best_threshold = 0.5
    best_metrics = {}
    best_combined = 0.0
    
    print(f"Searching for best threshold in {threshold_range}...")
    
    for threshold in threshold_range:
        metrics = evaluate_with_peak_picking(
            onset_pred, pitch_pred, onset_true, pitch_true,
            threshold=threshold, window_size=window_size,
            midi_min=midi_min, midi_max=midi_max
        )
        
        print(f"  Threshold {threshold:.1f}: F1={metrics['onset_f1']:.3f}, "
              f"MAE={metrics['pitch_mae']:.2f}, Combined={metrics['combined_score']:.3f}")
        
        if metrics['combined_score'] > best_combined:
            best_combined = metrics['combined_score']
            best_threshold = threshold
            best_metrics = metrics
    
    print(f"\nBest threshold: {best_threshold:.1f}")
    print(f"Best combined score: {best_combined:.3f}")
    
    return best_threshold, best_metrics


def main():
    parser = argparse.ArgumentParser(description='Peak Picking Evaluation')
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--ground_truth', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--window_size', type=int, default=3)
    parser.add_argument('--find_best', action='store_true')
    parser.add_argument('--midi_min', type=int, default=36)
    parser.add_argument('--midi_max', type=int, default=108)
    parser.add_argument('--tolerance_ms', type=float, default=50)
    
    args = parser.parse_args()
    
    print(f"Loading predictions from {args.predictions}...")
    predictions = torch.load(args.predictions)
    onset_pred = predictions['onset_pred'].numpy()
    pitch_pred = predictions['pitch_pred'].numpy()
    
    print(f"Loading ground truth from {args.ground_truth}...")
    ground_truth = torch.load(args.ground_truth)
    onset_true = ground_truth['onset_true'].numpy()
    pitch_true = ground_truth['pitch_true'].numpy()
    
    print(f"Data shapes: onset={onset_pred.shape}, pitch={pitch_pred.shape}")
    
    if args.find_best:
        best_threshold, best_metrics = find_best_threshold(
            onset_pred, pitch_pred, onset_true, pitch_true,
            window_size=args.window_size,
            midi_min=args.midi_min, midi_max=args.midi_max
        )
        
        print(f"\n{'='*60}")
        print(f"Best Threshold: {best_threshold:.1f}")
        print(f"{'='*60}")
        for key, value in best_metrics.items():
            print(f"  {key}: {value:.4f}")
    else:
        metrics = evaluate_with_peak_picking(
            onset_pred, pitch_pred, onset_true, pitch_true,
            threshold=args.threshold, window_size=args.window_size,
            midi_min=args.midi_min, midi_max=args.midi_max,
            onset_tolerance_ms=args.tolerance_ms
        )
        
        print(f"\n{'='*60}")
        print(f"Peak Picking Evaluation (threshold={args.threshold:.1f})")
        print(f"{'='*60}")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize model predictions on a test sample.

Creates a detailed comparison between Baseline and Dual-Branch models
showing onset predictions, pitch predictions, and ground truth.

Usage:
    python experiments/visualize_predictions.py \
        --audio data/guitarset/audio_mono-mic/00_BN1-129-Eb_comp_mic.wav \
        --annotations data/guitarset/csv_annotations/00_BN1-129-Eb_comp.csv \
        --baseline_checkpoint experiments/baseline/checkpoints/checkpoint_best.pth \
        --dual_branch_checkpoint experiments/dual_branch/checkpoints/checkpoint_best.pth \
        --output experiments/visualizations/comparison.png
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from models.baseline_cnn import BaselineCNN
from models.dual_branch_cnn import DualBranchCNN
from data_loading.guitarset_frame_dataset import AudioConfig


def load_audio_and_annotations(audio_path: str, annotations_path: str):
    """Load audio and parse annotations."""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=AudioConfig.sr)
    
    # Parse annotations
    import csv
    notes = []
    with open(annotations_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            notes.append({
                'time': float(row['time_sec']),
                'string': int(row['string']),
                'fret': int(row['fret']),
                'midi': float(row['midi'])
            })
    
    return audio, sr, notes


def extract_cqt_windows(audio, notes, context_frames=6):
    """
    Extract CQT windows centered on each note.
    
    Returns list of (cqt_window, note_info) tuples.
    """
    # Compute full CQT
    cqt = np.abs(librosa.cqt(
        audio,
        sr=AudioConfig.sr,
        hop_length=AudioConfig.hop_length,
        n_bins=AudioConfig.n_bins,
        bins_per_octave=AudioConfig.bins_per_octave,
        fmin=AudioConfig.fmin
    ))
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
    
    # Extract windows
    windows = []
    for note in notes:
        center_frame = int(note['time'] * AudioConfig.sr / AudioConfig.hop_length)
        
        # Extract window with padding
        start_frame = center_frame - context_frames
        end_frame = center_frame + context_frames + 1
        
        if start_frame < 0:
            pad_left = -start_frame
            cqt_padded = np.pad(cqt_db, ((0, 0), (pad_left, 0)), mode='reflect')
            start_frame = 0
        else:
            cqt_padded = cqt_db
            pad_left = 0
        
        if end_frame > cqt_padded.shape[1]:
            pad_right = end_frame - cqt_padded.shape[1]
            cqt_padded = np.pad(cqt_padded, ((0, 0), (0, pad_right)), mode='reflect')
        
        window = cqt_padded[:, start_frame:end_frame].T  # [13, 72]
        
        # Normalize
        window = (window - window.mean()) / (window.std() + 1e-8)
        window = np.nan_to_num(window, nan=0.0)
        
        windows.append({
            'cqt': window,
            'note': note,
            'time': note['time']
        })
    
    return windows


def load_model(checkpoint_path: str, model_type: str, device: torch.device):
    """Load model from checkpoint."""
    if model_type == 'baseline':
        model = BaselineCNN(
            encoder_channels=[32, 64, 96],
            head_hidden=48,
            dropout=0.3
        )
    else:  # dual_branch
        model = DualBranchCNN(
            encoder_channels=[32, 64, 96],
            head_hidden=64,
            dropout=0.3,
            se_reduction=8
        )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def predict_on_windows(model, windows, device, threshold=0.5):
    """Run predictions on all windows."""
    predictions = []
    
    with torch.no_grad():
        for w in windows:
            cqt = torch.FloatTensor(w['cqt']).unsqueeze(0).unsqueeze(0).to(device)
            onset_pred, pitch_pred = model(cqt)
            
            onset_pred = onset_pred.cpu().numpy()[0]
            pitch_pred = pitch_pred.cpu().numpy()[0]
            
            # Denormalize pitch
            pitch_midi = pitch_pred * (103 - 40) + 40
            
            predictions.append({
                'time': w['time'],
                'onset': onset_pred,
                'pitch': pitch_midi,
                'onset_binary': (onset_pred > threshold).astype(float)
            })
    
    return predictions


def visualize_comparison(windows, baseline_preds, dual_branch_preds, output_path: str):
    """Create detailed visualization comparing models."""
    
    # Select subset for visualization (first 20 notes)
    n_show = min(20, len(windows))
    windows = windows[:n_show]
    baseline_preds = baseline_preds[:n_show]
    dual_branch_preds = dual_branch_preds[:n_show]
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    # ============================================
    # 1. Ground Truth
    # ============================================
    ax = axes[0]
    ax.set_title('Ground Truth', fontsize=14, fontweight='bold')
    
    gt_matrix = np.zeros((n_show, 6))
    pitch_matrix = np.zeros((n_show, 6))
    
    for i, w in enumerate(windows):
        string = w['note']['string']
        gt_matrix[i, string] = 1.0
        pitch_matrix[i, string] = w['note']['midi']
    
    # Onset heatmap
    cmap = ListedColormap(['white', 'red'])
    ax.imshow(gt_matrix, aspect='auto', cmap=cmap)
    ax.set_xlabel('String')
    ax.set_ylabel('Note #')
    ax.set_xticks(range(6))
    ax.set_xticklabels(['E4', 'B3', 'G3', 'D3', 'A2', 'E2'])
    ax.set_yticks(range(n_show))
    ax.set_yticklabels([f"{i+1}\n{windows[i]['time']:.2f}s" for i in range(n_show)])
    
    # Add pitch values
    for i in range(n_show):
        for s in range(6):
            if gt_matrix[i, s] > 0:
                ax.text(s, i, f'{pitch_matrix[i, s]:.0f}', ha='center', va='center', 
                       fontsize=9, color='black', fontweight='bold')
    
    # ============================================
    # 2. Baseline Predictions
    # ============================================
    ax = axes[1]
    ax.set_title('Baseline CNN Predictions', fontsize=14, fontweight='bold')
    
    baseline_onset = np.array([p['onset'] for p in baseline_preds])
    baseline_pitch = np.array([p['pitch'] for p in baseline_preds])
    
    im = ax.imshow(baseline_onset.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xlabel('Note #')
    ax.set_ylabel('String')
    ax.set_xticks(range(n_show))
    ax.set_xticklabels([f"{i+1}" for i in range(n_show)], rotation=90)
    ax.set_yticks(range(6))
    ax.set_yticklabels(['E4', 'B3', 'G3', 'D3', 'A2', 'E2'])
    plt.colorbar(im, ax=ax, label='Onset Probability')
    
    # ============================================
    # 3. Dual-Branch Predictions
    # ============================================
    ax = axes[2]
    ax.set_title('Dual-Branch CNN Predictions', fontsize=14, fontweight='bold')
    
    dual_branch_onset = np.array([p['onset'] for p in dual_branch_preds])
    dual_branch_pitch = np.array([p['pitch'] for p in dual_branch_preds])
    
    im = ax.imshow(dual_branch_onset.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xlabel('Note #')
    ax.set_ylabel('String')
    ax.set_xticks(range(n_show))
    ax.set_xticklabels([f"{i+1}" for i in range(n_show)], rotation=90)
    ax.set_yticks(range(6))
    ax.set_yticklabels(['E4', 'B3', 'G3', 'D3', 'A2', 'E2'])
    plt.colorbar(im, ax=ax, label='Onset Probability')
    
    # ============================================
    # 4. Comparison Metrics
    # ============================================
    ax = axes[3]
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    
    # Compute metrics per note
    baseline_correct = 0
    dual_branch_correct = 0
    
    for i in range(n_show):
        gt_string = np.argmax(gt_matrix[i])
        
        baseline_max_string = np.argmax(baseline_onset[i])
        dual_branch_max_string = np.argmax(dual_branch_onset[i])
        
        if baseline_max_string == gt_string and baseline_onset[i, gt_string] > 0.5:
            baseline_correct += 1
        
        if dual_branch_max_string == gt_string and dual_branch_onset[i, gt_string] > 0.5:
            dual_branch_correct += 1
    
    # Plot comparison
    metrics = ['Correct\nDetections', 'Total\nNotes']
    values_baseline = [baseline_correct, n_show]
    values_dual = [dual_branch_correct, n_show]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, values_baseline, width, label='Baseline', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, values_dual, width, label='Dual-Branch', color='darkorange', alpha=0.8)
    
    ax.set_ylabel('Count')
    ax.set_title('Detection Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--annotations', type=str, required=True, help='Path to annotations CSV')
    parser.add_argument('--baseline_checkpoint', type=str, required=True, help='Baseline checkpoint')
    parser.add_argument('--dual_branch_checkpoint', type=str, required=True, help='Dual-Branch checkpoint')
    parser.add_argument('--output', type=str, default='experiments/visualizations/comparison.png', help='Output path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Onset threshold')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading audio and annotations...")
    audio, sr, notes = load_audio_and_annotations(args.audio, args.annotations)
    print(f"Loaded {len(notes)} notes from {args.audio}\n")
    
    # Extract windows
    print("Extracting CQT windows...")
    windows = extract_cqt_windows(audio, notes)
    print(f"Extracted {len(windows)} windows\n")
    
    # Load models
    print("Loading Baseline model...")
    baseline_model = load_model(args.baseline_checkpoint, 'baseline', device)
    print(f"  Parameters: {baseline_model.count_parameters():,}\n")
    
    print("Loading Dual-Branch model...")
    dual_branch_model = load_model(args.dual_branch_checkpoint, 'dual_branch', device)
    print(f"  Parameters: {dual_branch_model.count_parameters():,}\n")
    
    # Run predictions
    print("Running Baseline predictions...")
    baseline_preds = predict_on_windows(baseline_model, windows, device, args.threshold)
    
    print("Running Dual-Branch predictions...")
    dual_branch_preds = predict_on_windows(dual_branch_model, windows, device, args.threshold)
    
    # Visualize
    print("\nCreating visualization...")
    visualize_comparison(windows, baseline_preds, dual_branch_preds, args.output)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    # Count correct detections
    gt_strings = [w['note']['string'] for w in windows]
    baseline_correct = sum(1 for i, s in enumerate(gt_strings) 
                          if np.argmax(baseline_preds[i]['onset']) == s 
                          and baseline_preds[i]['onset'][s] > args.threshold)
    dual_correct = sum(1 for i, s in enumerate(gt_strings) 
                      if np.argmax(dual_branch_preds[i]['onset']) == s 
                      and dual_branch_preds[i]['onset'][s] > args.threshold)
    
    print(f"Total notes: {len(windows)}")
    print(f"Baseline correct: {baseline_correct} ({baseline_correct/len(windows)*100:.1f}%)")
    print(f"Dual-Branch correct: {dual_correct} ({dual_correct/len(windows)*100:.1f}%)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

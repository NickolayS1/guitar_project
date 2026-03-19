#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize GuitarSet dataset samples with Gaussian-smoothed labels.

Shows:
1. Full CQT spectrogram with note annotations
2. Multiple prediction windows with Gaussian onset probabilities
3. Detailed view of prediction frame and neighbors with probabilities

Usage:
    python visualize_dataset.py
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from guitarset_frame_dataset import AudioConfig, GuitarSetFrameDataset


# ============================================================================
# Visualization Configuration
# ============================================================================

class VizConfig:
    """Visualization configuration."""
    n_samples_to_show = 8       # Number of samples to visualize
    fig_width = 16              # Figure width in inches
    fig_height = 12             # Figure height in inches
    dpi = 150                   # DPI for figures
    cmap = 'magma'              # Colormap for spectrograms


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_full_cqt_with_annotations(
    audio_path: Path,
    annotations_path: Path,
    save_path: Path = None
):
    """Plot full CQT spectrogram with note annotations."""
    print(f"\n{'='*70}")
    print(f"Loading audio: {audio_path.name}")
    print(f"{'='*70}")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=AudioConfig.sr)
    duration = len(audio) / sr
    print(f"Duration: {duration:.2f} sec")
    
    # Compute CQT
    print("Computing CQT...")
    cqt = np.abs(librosa.cqt(
        audio,
        sr=AudioConfig.sr,
        hop_length=AudioConfig.hop_length,
        n_bins=AudioConfig.n_bins,
        bins_per_octave=AudioConfig.bins_per_octave,
        fmin=AudioConfig.fmin
    ))
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
    
    # Load annotations
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
    
    print(f"Found {len(notes)} notes")
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(VizConfig.fig_width, 8), dpi=VizConfig.dpi)
    
    # Plot CQT
    img = librosa.display.specshow(
        cqt_db,
        sr=AudioConfig.sr,
        hop_length=AudioConfig.hop_length,
        x_axis='time',
        y_axis='cqt_note',
        cmap=VizConfig.cmap,
        ax=axes[0]
    )
    axes[0].set_title(f'CQT Spectrogram - {audio_path.stem}')
    plt.colorbar(img, ax=axes[0], format='%+2.0f dB')
    
    # Plot note annotations
    string_colors = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#8B00FF']
    for i, n in enumerate(notes):
        color = string_colors[n['string']] if 0 <= n['string'] < 6 else '#FFFFFF'
        axes[0].axvline(x=n['time'], color=color, alpha=0.3, linewidth=1)
    
    # Note density
    note_density = np.zeros(int(duration * 10))
    for n in notes:
        idx = int(n['time'] * 10)
        if idx < len(note_density):
            note_density[idx] += 1
    
    axes[1].fill_between(
        np.arange(len(note_density)) / 10,
        note_density,
        alpha=0.7,
        color='blue'
    )
    axes[1].set_xlabel('Time (sec)')
    axes[1].set_ylabel('Notes per 100ms')
    axes[1].set_title('Note Density')
    axes[1].set_xlim(0, duration)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VizConfig.dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    return audio, cqt_db, notes


def plot_prediction_windows(
    dataset: GuitarSetFrameDataset,
    sample_indices: List[int],
    audio_duration: float,
    save_dir: Path = None
):
    """
    Plot prediction windows with Gaussian-smoothed onset probabilities.
    
    Shows:
    1. Full CQT window with prediction frame highlighted
    2. Onset probabilities for all strings
    3. Gaussian probabilities for center frame and neighbors
    """
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    for idx in sample_indices:
        if idx >= len(dataset):
            continue
        
        sample = dataset[idx]
        
        # Get sample info
        has_onset = sample['has_onset']
        time_sec = sample['time_sec']
        onset = sample['onset'].numpy()
        pitch_norm = sample['pitch'].numpy()
        cqt_norm = sample['cqt'].numpy()  # [13, 84]
        
        # Basic info
        n_frames = cqt_norm.shape[0]
        pred_frame = n_frames // 2  # Center frame (index 6)
        
        print(f"\n{'='*70}")
        print(f"Sample {idx}: Time={time_sec:.3f}s, Has_onset={has_onset}")
        print(f"  Onset probs: {onset}")
        
        # Compute neighbor probabilities using Gaussian formula
        frame_duration_ms = AudioConfig.frame_duration_ms
        sigma_ms = AudioConfig.gaussian_sigma_ms
        
        # Find active strings
        active_strings = np.where(onset > AudioConfig.onset_threshold)[0]
        
        # Create figure
        fig, axes = plt.subplots(4, 1, figsize=(VizConfig.fig_width, 14), dpi=VizConfig.dpi)
        
        # === 1. CQT Spectrogram ===
        time_offsets = np.arange(n_frames) * frame_duration_ms / 1000
        time_offsets = time_offsets - AudioConfig.context_window_ms / 1000
        
        img0 = librosa.display.specshow(
            cqt_norm.T,
            sr=AudioConfig.sr,
            hop_length=AudioConfig.hop_length,
            x_axis=None,
            y_axis='cqt_note',
            cmap=VizConfig.cmap,
            ax=axes[0]
        )
        
        # Mark prediction frame (single frame, not window!)
        axes[0].axvspan(
            pred_frame - 0.5,
            pred_frame + 0.5,
            color='yellow', alpha=0.5,
            label=f'Prediction frame ({frame_duration_ms:.1f}ms)'
        )
        axes[0].axvline(x=pred_frame, color='red', linestyle='--', linewidth=2,
                       label=f'Center (t={time_sec:.3f}s)')
        
        # Mark context boundaries
        axes[0].axvline(x=0, color='blue', linestyle=':', alpha=0.5, label='Context start')
        axes[0].axvline(x=n_frames-1, color='green', linestyle=':', alpha=0.5, label='Context end')
        
        # X-axis labels
        axes[0].set_xticks([0, pred_frame, n_frames-1])
        axes[0].set_xticklabels([
            f'{time_offsets[0]:.2f}s',
            f'{time_offsets[pred_frame]:.2f}s (center)',
            f'{time_offsets[-1]:.2f}s'
        ])
        
        axes[0].set_title(f'CQT Input [{n_frames} frames × 84 bins] | Context: ±{AudioConfig.context_window_ms}ms')
        axes[0].legend(loc='upper right', fontsize=8)
        plt.colorbar(img0, ax=axes[0], format='%+2.1f')
        
        # === 2. Onset Probabilities (Gaussian-smoothed) ===
        string_names = ['E4 (1st)', 'B3 (2nd)', 'G3 (3rd)', 'D3 (4th)', 'A2 (5th)', 'E2 (6th)']
        string_colors = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#8B00FF']
        y_pos = np.arange(6)
        
        bars = axes[1].barh(y_pos, onset, color=[string_colors[i] for i in range(6)], alpha=0.7)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(string_names)
        axes[1].set_xlim(0, 1.1)
        axes[1].set_xlabel('Onset Probability (Gaussian-smoothed)')
        axes[1].set_title(f'Onset Probabilities at Prediction Frame (σ={sigma_ms}ms, threshold={AudioConfig.onset_threshold})')
        axes[1].grid(axis='x', alpha=0.3)
        axes[1].axvline(x=AudioConfig.onset_threshold, color='red', linestyle='--', 
                       label=f'Threshold ({AudioConfig.onset_threshold})')
        
        # Add probability values on bars
        for i, (bar, prob) in enumerate(zip(bars, onset)):
            if prob > 0.1:
                axes[1].text(prob + 0.02, i, f'{prob:.2f}', va='center', fontsize=10, fontweight='bold')
        
        # === 3. Gaussian Probabilities for Neighboring Frames ===
        if len(active_strings) > 0:
            # Show probabilities for center frame and neighbors
            neighbor_offsets = [-1, 0, +1]  # Previous, center, next
            neighbor_times = [f'-{frame_duration_ms:.1f}ms', '0ms (center)', f'+{frame_duration_ms:.1f}ms']
            
            x_pos = np.arange(len(neighbor_offsets))
            width = 0.25
            
            for i, string in enumerate(active_strings[:3]):  # Show up to 3 active strings
                probs = []
                for offset in neighbor_offsets:
                    # Compute Gaussian probability at offset frames
                    dist_ms = abs(offset) * frame_duration_ms
                    if offset == 0:
                        prob = onset[string]
                    else:
                        # Approximate: prob at distance d from center
                        prob = onset[string] * np.exp(-(dist_ms**2) / (2 * sigma_ms**2))
                    probs.append(prob)
                
                axes[2].bar(x_pos + i*width, probs, width, label=f'String {string} ({string_names[string]})', alpha=0.8)
                
                # Add value labels
                for j, prob in enumerate(probs):
                    if prob > 0.05:
                        axes[2].text(j + i*width, prob + 0.02, f'{prob:.2f}', ha='center', va='bottom', fontsize=8)
            
            axes[2].set_xticks(x_pos + width)
            axes[2].set_xticklabels(neighbor_times)
            axes[2].set_xlabel('Frame offset from prediction frame')
            axes[2].set_ylabel('Onset Probability')
            axes[2].set_title(f'Gaussian-Smoothed Onset Probabilities (σ={sigma_ms}ms)')
            axes[2].legend(loc='upper right', fontsize=8)
            axes[2].grid(axis='y', alpha=0.3)
            axes[2].set_ylim(0, 1.1)
        else:
            axes[2].text(0.5, 0.5, 'No active strings (negative sample)', 
                        ha='center', va='center', fontsize=14)
            axes[2].set_xlim(0, 1)
            axes[2].set_ylim(0, 1)
            axes[2].set_title('Gaussian-Smoothed Onset Probabilities')
        
        # === 4. Pitch/Fret Information ===
        pitch_midi = pitch_norm * (AudioConfig.midi_max - AudioConfig.midi_min) + AudioConfig.midi_min
        pitch_midi = np.round(pitch_midi).astype(int)
        
        tuning_midi = [64, 59, 55, 50, 45, 40]
        frets = np.zeros(6, dtype=int)
        for s in range(6):
            if onset[s] > 0.5 and pitch_midi[s] > 0:
                frets[s] = pitch_midi[s] - tuning_midi[s]
            else:
                frets[s] = -1
        
        if has_onset:
            info_text = f"Active strings:\n"
            for s in range(6):
                if onset[s] > 0.5:
                    info_text += f"  String {s} ({string_names[s]}): MIDI={pitch_midi[s]}, Fret={frets[s]}, Prob={onset[s]:.2f}\n"
        else:
            info_text = "No active strings (negative sample)"
        
        axes[3].text(0.1, 0.5, info_text, fontsize=11, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[3].axis('off')
        axes[3].set_title('Pitch and Fret Information')
        
        # Add summary text
        summary = (f'Sample {idx} | Time: {time_sec:.3f}s | '
                  f'Input: [{n_frames}, {AudioConfig.n_bins}] | '
                  f'Context: ±{AudioConfig.context_window_ms}ms | '
                  f'Prediction: {frame_duration_ms:.1f}ms (1 frame)')
        fig.suptitle(summary, y=0.995, fontsize=10, family='monospace')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = save_dir / f'sample_{idx:04d}.png'
            plt.savefig(save_path, dpi=VizConfig.dpi, bbox_inches='tight')
            print(f"  Saved: {save_path}")
            plt.close()
        else:
            plt.show()


def plot_dataset_statistics(dataset: GuitarSetFrameDataset, save_path: Path = None):
    """Plot dataset statistics."""
    print(f"\n{'='*70}")
    print(f"Dataset Statistics")
    print(f"{'='*70}")
    
    n_positive = sum(1 for s in dataset.samples if s['has_onset'])
    n_negative = len(dataset) - n_positive
    
    all_pitches = []
    for s in dataset.samples:
        if s['has_onset']:
            pitch_norm = s['pitch']
            for i, o in enumerate(s['onset']):
                if o > 0.5 and pitch_norm[i] > 0:
                    midi = pitch_norm[i] * (AudioConfig.midi_max - AudioConfig.midi_min) + AudioConfig.midi_min
                    all_pitches.append(midi)
    
    all_pitches = np.array(all_pitches)
    
    string_counts = np.zeros(6)
    for s in dataset.samples:
        if s['has_onset']:
            for i, o in enumerate(s['onset']):
                if o > 0.5:
                    string_counts[i] += 1
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=VizConfig.dpi)
    
    # 1. Positive vs Negative
    axes[0, 0].pie([n_positive, n_negative], 
                   labels=[f'Positive\n{n_positive}', f'Negative\n{n_negative}'],
                   autopct='%1.1f%%',
                   colors=['green', 'gray'])
    axes[0, 0].set_title('Sample Distribution')
    
    # 2. String distribution
    string_names = ['E4', 'B3', 'G3', 'D3', 'A2', 'E2']
    axes[0, 1].bar(range(6), string_counts, color='blue', alpha=0.7)
    axes[0, 1].set_xticks(range(6))
    axes[0, 1].set_xticklabels(string_names)
    axes[0, 1].set_xlabel('String')
    axes[0, 1].set_ylabel('Number of Notes')
    axes[0, 1].set_title('Note Distribution by String')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Pitch distribution
    if len(all_pitches) > 0:
        axes[1, 0].hist(all_pitches, bins=40, color='purple', alpha=0.7)
        axes[1, 0].set_xlabel('MIDI Pitch')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Pitch Distribution')
        axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Onset probability distribution
    onset_probs = []
    for s in dataset.samples:
        if s['has_onset']:
            for o in s['onset']:
                if o > 0.1:
                    onset_probs.append(o)
    
    if len(onset_probs) > 0:
        axes[1, 1].hist(onset_probs, bins=20, color='orange', alpha=0.7)
        axes[1, 1].set_xlabel('Onset Probability')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Gaussian-Smoothed Onset Probability Distribution')
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].axvline(x=AudioConfig.onset_threshold, color='red', linestyle='--',
                          label=f'Threshold ({AudioConfig.onset_threshold})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VizConfig.dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    # Print statistics
    print(f"Total samples: {len(dataset)}")
    print(f"Positive: {n_positive} ({n_positive/len(dataset)*100:.1f}%)")
    print(f"Negative: {n_negative} ({n_negative/len(dataset)*100:.1f}%)")
    print(f"\nString distribution:")
    for i, (name, count) in enumerate(zip(string_names, string_counts)):
        print(f"  String {i} ({name}): {int(count)} notes")
    
    if len(all_pitches) > 0:
        print(f"\nPitch range: {all_pitches.min():.1f} - {all_pitches.max():.1f} MIDI")
        print(f"Pitch mean: {all_pitches.mean():.1f} MIDI")
    else:
        print("\nNo pitches found")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main visualization function."""
    print("="*70)
    print("GuitarSet Dataset Visualization (Gaussian-Smoothed Labels)")
    print("="*70)
    
    # Configuration
    root_dir = Path('data/guitarset')
    split = 'train'
    save_dir = Path('experiments/visualizations')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    print(f"\nLoading dataset from {root_dir}...")
    dataset = GuitarSetFrameDataset(
        root_dir=str(root_dir),
        split=split,
        negative_ratio=1.0,
        seed=42
    )
    
    if len(dataset) == 0:
        print("Dataset is empty!")
        return
    
    # 1. Plot dataset statistics
    plot_dataset_statistics(
        dataset,
        save_path=save_dir / 'dataset_statistics.png'
    )
    
    # 2. Plot full CQT for first file
    first_filename = dataset.filenames[0]
    audio_path = root_dir / 'audio_mono-mic' / first_filename.replace('.csv', '_mic.wav')
    annotations_path = root_dir / 'csv_annotations_v3' / first_filename
    
    duration = 0.0
    if audio_path.exists() and annotations_path.exists():
        audio, cqt_db, notes = plot_full_cqt_with_annotations(
            audio_path,
            annotations_path,
            save_path=save_dir / 'full_cqt_example.png'
        )
        duration = len(audio) / AudioConfig.sr
    
    # 3. Plot prediction windows
    # Select diverse samples:
    sample_indices = []
    
    # Find samples with chords (multiple strings active)
    chord_indices = []
    for i, s in enumerate(dataset.samples):
        if s['has_onset']:
            n_active = np.sum(s['onset'] > 0.5)
            if n_active >= 2:
                chord_indices.append(i)
                if len(chord_indices) >= 2:
                    break
    
    sample_indices.extend(chord_indices)
    print(f"Found {len(chord_indices)} chord samples")
    
    # Find samples with high onset probability (single note)
    high_onset_indices = [i for i, s in enumerate(dataset.samples) 
                          if s['has_onset'] and np.max(s['onset']) > 0.8 
                          and i not in sample_indices][:3]
    sample_indices.extend(high_onset_indices)
    
    # Find negative samples (not at the very beginning to avoid padding)
    min_time = AudioConfig.context_window_ms / 1000.0
    no_onset_indices = [i for i, s in enumerate(dataset.samples) 
                        if not s['has_onset'] and s['center_time'] > min_time][:2]
    sample_indices.extend(no_onset_indices)
    
    print(f"\nVisualizing {len(sample_indices)} samples...")
    plot_prediction_windows(
        dataset,
        sample_indices,
        audio_duration=duration,
        save_dir=save_dir
    )
    
    print(f"\n{'='*70}")
    print(f"Visualization complete!")
    print(f"Saved to: {save_dir.absolute()}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

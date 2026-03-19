#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GuitarSet Dataset for frame-level guitar transcription.

Creates CQT windows centered on prediction frames with ±150ms context.
Uses single-frame prediction with Gaussian-smoothed labels.

Configuration:
    - Sample rate: 22050 Hz
    - Hop length: 512 samples (23.2 ms per frame)
    - Prediction window: 23.2 ms (1 frame, center)
    - Context window: ±150 ms (~6 frames each side)
    - Total CQT input: 13 frames × 72 bins (C2-C8)
    - fmin: 65.4 Hz (C2, below lowest guitar note E2=82.4Hz)
    - Gaussian smoothing: σ=15ms, threshold=0.1
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import librosa


# ============================================================================
# Configuration
# ============================================================================

class AudioConfig:
    """Audio processing configuration."""
    sr = 22050              # Sample rate
    hop_length = 512        # Hop length in samples (23.2 ms)
    n_bins = 72             # Number of CQT bins (6 octaves: C2 to C8)
    bins_per_octave = 12    # Bins per octave
    fmin = 65.4             # Minimum frequency (C2, below lowest guitar note E2=82.4Hz)
    
    # Time windows in milliseconds
    prediction_window_ms = hop_length / sr * 1000  # Single frame (~23.2 ms)
    context_window_ms = 150     # Context on each side (±150 ms)
    
    # Label smoothing with Gaussian
    # σ = 15ms gives smooth transition over ~45ms (3σ)
    gaussian_sigma_ms = 15.0
    onset_threshold = 0.1       # Below this → onset = 0
    
    # Derived values
    frame_duration_ms = hop_length / sr * 1000  # ~23.2 ms
    prediction_frames = 1       # Single center frame
    context_frames = int(np.round(context_window_ms / frame_duration_ms))  # ~6 frames
    total_frames = 2 * context_frames + prediction_frames  # ~13 frames
    
    # MIDI range for pitch normalization
    midi_min = 40   # E2 (lowest guitar string)
    midi_max = 103  # C6 (highest practical guitar note)


# ============================================================================
# Dataset
# ============================================================================

class GuitarSetFrameDataset(Dataset):
    """
    Frame-level GuitarSet dataset for guitar transcription.
    
    For each note onset in the annotations:
    1. Extract CQT window with ±500ms context
    2. Create labels for the prediction window (50ms)
    
    For negative sampling:
    1. Sample random frames between notes
    2. Label as silence (no onsets)
    """
    
    def __init__(
        self,
        root_dir: str = 'data/guitarset',
        split: str = 'train',
        negative_ratio: float = 1.0,
        seed: int = 42
    ):
        """
        Args:
            root_dir: Root directory for GuitarSet
            split: 'train', 'val', or 'test'
            negative_ratio: Ratio of negative samples to positive samples
            seed: Random seed for negative sampling
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.negative_ratio = negative_ratio
        self.rng = np.random.default_rng(seed)
        
        # Load split list
        self.split_file = self.root_dir / 'splits' / f'{split}.txt'
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        
        with open(self.split_file, 'r') as f:
            self.filenames = [line.strip() for line in f if line.strip()]
        
        # Prepare samples
        self.samples = self._prepare_samples()
        
        print(f"GuitarSet {split}: {len(self.samples)} samples")
        print(f"  Positive: {sum(1 for s in self.samples if s['has_onset'])}")
        print(f"  Negative: {sum(1 for s in self.samples if not s['has_onset'])}")
    
    def _prepare_samples(self) -> List[Dict]:
        """Prepare all samples from the dataset."""
        samples = []
        
        for filename in self.filenames:
            csv_path = self.root_dir / 'csv_annotations' / filename
            # Audio file naming: 00_BN1-129-Eb_comp.csv → 00_BN1-129-Eb_comp_mic.wav
            audio_name = filename.replace('.csv', '_mic.wav')
            audio_path = self.root_dir / 'audio_mono-mic' / audio_name
            
            if not csv_path.exists() or not audio_path.exists():
                print(f"Warning: Missing files for {filename}")
                print(f"  CSV: {csv_path.exists()}")
                print(f"  Audio: {audio_path.exists()}")
                continue
            
            # Load annotations (just the raw notes, not processed)
            notes_raw = []
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    notes_raw.append({
                        'time_sec': float(row['time_sec']),
                        'string': int(row['string']),
                        'fret': int(row['fret']),
                        'midi': float(row['midi'])
                    })
            
            if len(notes_raw) == 0:
                continue

            # Get audio duration
            duration = librosa.get_duration(path=audio_path)

            # Create positive samples (one per note, centered on the note)
            for note in notes_raw:
                onset = np.zeros(6)
                pitch = np.zeros(6)
                string = note['string']
                if 0 <= string < 6:
                    onset[string] = 1.0  # Centered on note → Gaussian = 1.0
                    pitch[string] = note['midi']
                
                samples.append({
                    'audio_path': str(audio_path),
                    'center_time': note['time_sec'],
                    'onset': onset,
                    'pitch': pitch,
                    'has_onset': True,
                    'notes_raw': notes_raw  # Store for Gaussian smoothing in __getitem__
                })

            # Create negative samples (random frames between notes)
            n_negative = int(len(notes_raw) * self.negative_ratio)
            negative_times = self._sample_negative_times(notes_raw, duration, n_negative)

            for time in negative_times:
                # Compute Gaussian-smoothed onset for negative samples
                onset, pitch = self._compute_gaussian_onset(time, notes_raw)
                
                samples.append({
                    'audio_path': str(audio_path),
                    'center_time': time,
                    'onset': onset,
                    'pitch': pitch,
                    'has_onset': onset.sum() > 0,
                    'notes_raw': notes_raw
                })
        
        return samples
    
    def _load_annotations(self, csv_path: Path) -> List[Dict]:
        """
        Load annotations from CSV and create frame-level labels with Gaussian smoothing.
        
        For each frame in the audio, computes onset probability based on distance
        to nearest note onset using Gaussian smoothing.
        """
        # Load all notes
        notes = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                notes.append({
                    'time_sec': float(row['time_sec']),
                    'string': int(row['string']),
                    'fret': int(row['fret']),
                    'midi': float(row['midi'])
                })
        
        if len(notes) == 0:
            return []
        
        # Sort notes by time
        notes_sorted = sorted(notes, key=lambda x: x['time_sec'])
        
        # Create frame-level labels with Gaussian smoothing
        # We'll create samples at each note time (positive) and some negative samples
        samples = []
        
        # Gaussian parameters
        sigma_ms = AudioConfig.gaussian_sigma_ms
        threshold = AudioConfig.onset_threshold
        
        # For each note, create a sample centered on the note
        for note in notes_sorted:
            # Compute Gaussian-smoothed onset for all strings
            onset = np.zeros(6)
            pitch = np.zeros(6)
            
            # For the string this note is on, compute onset probability
            string = note['string']
            if 0 <= string < 6:
                # Distance from center frame to note onset is 0 (we're centered on the note)
                # So Gaussian = 1.0
                onset[string] = 1.0
                pitch[string] = note['midi']
            
            samples.append({
                'time_sec': note['time_sec'],
                'onset': onset,
                'pitch': pitch,
                'has_onset': True
            })
        
        return samples
    
    def _create_window_label(self, notes: List[Dict], center_time: float) -> Dict:
        """
        Create label for a prediction window.
        
        Args:
            notes: List of notes that start in this window
            center_time: Center time of the window
        
        Returns:
            Dictionary with onset and pitch arrays
        """
        onset = np.zeros(6)
        pitch = np.zeros(6)
        
        for note in notes:
            string = note['string']
            if 0 <= string < 6:
                onset[string] = 1.0
                pitch[string] = note['midi']
        
        return {
            'time_sec': center_time,
            'onset': onset,
            'pitch': pitch
        }
    
    def _sample_negative_times(
        self,
        notes: List[Dict],
        duration: float,
        n_samples: int
    ) -> np.ndarray:
        """
        Sample random times between notes for negative samples.
        
        Args:
            notes: List of note annotations
            duration: Audio duration in seconds
            n_samples: Number of negative samples to generate
        
        Returns:
            Array of times for negative samples
        """
        if n_samples == 0:
            return np.array([])
        
        # Create list of safe regions (between notes)
        safe_regions = []
        
        # Add buffer around notes (don't sample too close to onsets)
        buffer = AudioConfig.context_window_ms / 1000.0  # 170ms buffer
        
        prev_end = buffer
        for note in sorted(notes, key=lambda x: x['time_sec']):
            note_start = note['time_sec'] - buffer
            note_end = note['time_sec'] + buffer
            
            if note_start > prev_end:
                safe_regions.append((prev_end, note_start))
            
            prev_end = note_end
        
        # Add region after last note
        if prev_end < duration - buffer:
            safe_regions.append((prev_end, duration - buffer))
        
        if len(safe_regions) == 0:
            return np.array([])
        
        # Sample uniformly from safe regions
        times = []
        for _ in range(n_samples):
            # Choose random region (weighted by length)
            region_lengths = [end - start for start, end in safe_regions]
            region_idx = self.rng.choice(len(safe_regions), p=np.array(region_lengths) / sum(region_lengths))
            
            # Sample uniformly from chosen region
            start, end = safe_regions[region_idx]
            time = self.rng.uniform(start, end)
            times.append(time)
        
        return np.array(times)

    def _compute_gaussian_onset(
        self,
        center_time: float,
        notes: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Gaussian-smoothed onset probabilities for all strings.
        
        For each string, finds the nearest note onset and computes:
            onset_prob = exp(-d² / (2σ²))
        where d = distance from center frame to note onset.
        
        Args:
            center_time: Center time of the prediction frame
            notes: List of all notes in the audio
        
        Returns:
            Tuple of (onset [6], pitch [6])
        """
        onset = np.zeros(6)
        pitch = np.zeros(6)
        
        sigma_ms = AudioConfig.gaussian_sigma_ms
        threshold = AudioConfig.onset_threshold
        sigma_sec = sigma_ms / 1000.0
        
        # For each string, find nearest note onset
        for string in range(6):
            string_notes = [n for n in notes if n['string'] == string]
            
            if len(string_notes) == 0:
                continue
            
            # Find nearest note to center_time
            distances = [abs(n['time_sec'] - center_time) for n in string_notes]
            nearest_idx = np.argmin(distances)
            nearest_dist = distances[nearest_idx]
            
            # Compute Gaussian probability
            # σ = 15ms → at d=23ms (1 frame): prob = exp(-23²/(2×15²)) ≈ 0.47
            # at d=46ms (2 frames): prob = exp(-46²/(2×15²)) ≈ 0.05
            prob = np.exp(-(nearest_dist ** 2) / (2 * sigma_sec ** 2))
            
            # Apply threshold
            if prob > threshold:
                onset[string] = prob
                pitch[string] = string_notes[nearest_idx]['midi']
        
        return onset, pitch

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load audio
        audio, _ = librosa.load(sample['audio_path'], sr=AudioConfig.sr)

        # Compute CQT
        cqt = np.abs(librosa.cqt(
            audio,
            sr=AudioConfig.sr,
            hop_length=AudioConfig.hop_length,
            n_bins=AudioConfig.n_bins,
            bins_per_octave=AudioConfig.bins_per_octave,
            fmin=AudioConfig.fmin
        ))

        # Convert to log scale
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)

        # Extract window centered on prediction frame
        center_frame = int(sample['center_time'] * AudioConfig.sr / AudioConfig.hop_length)
        cqt_window = self._extract_cqt_window(cqt_db, center_frame)

        # Normalize
        cqt_window = (cqt_window - cqt_window.mean()) / (cqt_window.std() + 1e-8)

        # Handle NaN/Inf
        cqt_window = np.nan_to_num(cqt_window, nan=0.0, posinf=0.0, neginf=0.0)

        # Prepare labels (already Gaussian-smoothed in _prepare_samples)
        onset = torch.FloatTensor(sample['onset'])
        pitch = torch.FloatTensor(self._normalize_pitch(sample['pitch']))

        return {
            'cqt': torch.FloatTensor(cqt_window),  # [total_frames, n_bins]
            'onset': onset,  # [6] - Gaussian smoothed probabilities
            'pitch': pitch,  # [6]
            'has_onset': sample['has_onset'],
            'time_sec': sample['center_time']
        }
    
    def _extract_cqt_window(
        self,
        cqt: np.ndarray,
        center_frame: int
    ) -> np.ndarray:
        """
        Extract CQT window with context.
        
        Args:
            cqt: CQT spectrogram [n_bins, time]
            center_frame: Center frame index
        
        Returns:
            CQT window [total_frames, n_bins]
        """
        start_frame = center_frame - AudioConfig.context_frames
        end_frame = center_frame + AudioConfig.context_frames + 1
        
        # Pad if necessary to ensure fixed size
        pad_left = max(0, -start_frame)
        pad_right = max(0, end_frame - cqt.shape[1])
        
        if pad_left > 0 or pad_right > 0:
            cqt = np.pad(cqt, ((0, 0), (pad_left, pad_right)), mode='reflect')
            start_frame = max(0, start_frame)
            end_frame = start_frame + AudioConfig.total_frames
        
        # Extract and transpose to [time, freq]
        window = cqt[:, start_frame:end_frame].T
        
        # Ensure fixed size
        if window.shape[0] != AudioConfig.total_frames:
            # Pad or truncate to exact size
            if window.shape[0] < AudioConfig.total_frames:
                pad = AudioConfig.total_frames - window.shape[0]
                window = np.pad(window, ((0, pad), (0, 0)), mode='reflect')
            else:
                window = window[:AudioConfig.total_frames, :]
        
        return window
    
    def _normalize_pitch(self, pitch: np.ndarray) -> np.ndarray:
        """Normalize MIDI pitch to [0, 1] range."""
        active = pitch > 0
        if active.any():
            pitch[active] = (pitch[active] - AudioConfig.midi_min) / (AudioConfig.midi_max - AudioConfig.midi_min)
        return pitch
    
    def _denormalize_pitch(self, pitch: np.ndarray) -> np.ndarray:
        """Denormalize pitch from [0, 1] to MIDI."""
        return pitch * (AudioConfig.midi_max - AudioConfig.midi_min) + AudioConfig.midi_min


# ============================================================================
# DataLoader utilities
# ============================================================================

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
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


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """Create DataLoader for GuitarSet dataset."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


# ============================================================================
# Testing
# ============================================================================

def test_dataset():
    """Test the dataset creation."""
    print("Testing GuitarSetFrameDataset...")
    print(f"Configuration:")
    print(f"  Frame duration: {AudioConfig.frame_duration_ms:.2f} ms")
    print(f"  Prediction window: {AudioConfig.prediction_window_ms} ms ({AudioConfig.prediction_frames} frames)")
    print(f"  Context window: ±{AudioConfig.context_window_ms} ms ({AudioConfig.context_frames} frames)")
    print(f"  Total frames: {AudioConfig.total_frames}")
    print()
    
    # Test with train split
    try:
        dataset = GuitarSetFrameDataset(
            root_dir='data/guitarset',
            split='train',
            negative_ratio=1.0
        )
        
        if len(dataset) == 0:
            print("Dataset is empty! Check if data/guitarset/splits_v3/train.txt exists.")
            return
        
        # Get a sample
        sample = dataset[0]
        
        print(f"\nSample shapes:")
        print(f"  CQT: {sample['cqt'].shape}")
        print(f"  Onset: {sample['onset'].shape}")
        print(f"  Pitch: {sample['pitch'].shape}")
        
        print(f"\nSample labels:")
        print(f"  Has onset: {sample['has_onset']}")
        print(f"  Time: {sample['time_sec']:.3f} sec")
        print(f"  Onset: {sample['onset']}")
        print(f"  Pitch (normalized): {sample['pitch']}")
        
        # Denormalize pitch for display
        pitch_midi = sample['pitch'].numpy() * (AudioConfig.midi_max - AudioConfig.midi_min) + AudioConfig.midi_min
        print(f"  Pitch (MIDI): {pitch_midi.round().astype(int)}")
        
        # Test DataLoader
        print(f"\nTesting DataLoader...")
        loader = create_dataloader(dataset, batch_size=4, shuffle=False, num_workers=0)
        
        for batch in loader:
            print(f"Batch shapes:")
            print(f"  CQT: {batch['cqt'].shape}")
            print(f"  Onset: {batch['onset'].shape}")
            print(f"  Pitch: {batch['pitch'].shape}")
            print(f"  Has onset: {batch['has_onset'].shape}")
            break
        
        print("\n✓ Dataset test passed!")
        print("\nTip: Run data_loading/visualize_dataset.py for visual inspection")
        
    except FileNotFoundError as e:
        print(f"✗ Dataset test failed: {e}")
        print("Make sure GuitarSet is properly converted and split.")


if __name__ == '__main__':
    test_dataset()

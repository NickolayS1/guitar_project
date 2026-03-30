# -*- coding: utf-8 -*-
"""
Dataset for own sessions (analogous to GuitarSetFrameDataset).

Reads:
- video.mp4 (extract frames)
- audio.wav (compute CQT)
- labels_enriched.csv (annotations)

Usage:
    dataset = OwnSessionsDataset(
        root_dir='data/own_sessions',
        split='train',
        split_dir='splits'
    )
"""

import csv
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


class AudioConfig:
    """Audio processing configuration."""
    sr = 22050
    hop_length = 512
    n_bins = 72  # Match Enhanced Baseline (72 bins)
    bins_per_octave = 12
    fmin = 65.4  # C2
    midi_min = 36  # C2
    midi_max = 108  # C8
    context_window_ms = 150
    prediction_frames = 1


class OwnSessionsDataset(Dataset):
    """
    Frame-level dataset for own sessions.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        split_dir: str = 'splits',
        negative_ratio: float = 1.0
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.negative_ratio = negative_ratio
        
        # Load split file
        splits_path = self.root_dir / split_dir / f"{split}.txt"
        if not splits_path.exists():
            raise ValueError(f"Split file not found: {splits_path}")
        
        with open(splits_path, 'r') as f:
            session_names = [line.strip() for line in f if line.strip()]
        
        print(f"Loading {split} sessions: {len(session_names)} sessions")
        
        # Prepare samples
        self.samples = []
        for session_name in session_names:
            session_dir = self.root_dir / session_name
            self._prepare_session(session_dir)
        
        print(f"  Total samples: {len(self.samples)}")
    
    def _prepare_session(self, session_dir: Path):
        """Prepare samples from single session."""
        # Check files
        audio_path = session_dir / "audio.wav"
        labels_path = session_dir / "labels_enriched.csv"
        
        if not audio_path.exists():
            print(f"  Warning: Missing audio: {audio_path}")
            return
        
        if not labels_path.exists():
            print(f"  Warning: Missing labels: {labels_path}")
            return
        
        # Load audio and compute CQT
        audio, _ = librosa.load(str(audio_path), sr=AudioConfig.sr)
        
        cqt = np.abs(librosa.cqt(
            audio,
            sr=AudioConfig.sr,
            hop_length=AudioConfig.hop_length,
            n_bins=AudioConfig.n_bins,
            bins_per_octave=AudioConfig.bins_per_octave,
            fmin=AudioConfig.fmin
        ))
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
        
        # Load labels
        notes = []
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                notes.append({
                    'time_sec': float(row['time_sec']),
                    'string': int(row['string']),
                    'fret': int(row.get('fret', 0)),
                    'midi': float(row.get('midi', 0))
                })
        
        # Create samples for each note
        frame_duration_ms = AudioConfig.hop_length / AudioConfig.sr * 1000
        context_frames = int(np.round(AudioConfig.context_window_ms / frame_duration_ms))
        total_frames = 2 * context_frames + AudioConfig.prediction_frames
        
        for note in notes:
            center_frame = int(note['time_sec'] * AudioConfig.sr / AudioConfig.hop_length)
            
            # Extract CQT window
            start_frame = center_frame - context_frames
            end_frame = center_frame + context_frames + 1
            
            if start_frame < 0 or end_frame > cqt_db.shape[1]:
                pad_left = max(0, -start_frame)
                pad_right = max(0, end_frame - cqt_db.shape[1])
                cqt_padded = np.pad(cqt_db, ((0, 0), (pad_left, pad_right)), mode='reflect')
                start_frame = 0 if start_frame < 0 else start_frame
                end_frame = cqt_padded.shape[1] if end_frame > cqt_padded.shape[1] else end_frame
            else:
                cqt_padded = cqt_db
            
            cqt_window = cqt_padded[:, start_frame:end_frame].T  # [total_frames, n_bins]
            
            # Normalize
            cqt_window = (cqt_window - cqt_window.mean()) / (cqt_window.std() + 1e-8)
            cqt_window = np.nan_to_num(cqt_window, nan=0.0)
            
            # Create labels
            onset = np.zeros(6, dtype=np.float32)
            onset[note['string']] = 1.0
            
            pitch = np.zeros(6, dtype=np.float32)
            pitch[note['string']] = (note['midi'] - AudioConfig.midi_min) / (AudioConfig.midi_max - AudioConfig.midi_min)
            pitch = np.clip(pitch, 0, 1)
            
            self.samples.append({
                'cqt': cqt_window,
                'onset': onset,
                'pitch': pitch,
                'has_onset': 1
            })
        
        # Add negative samples (no onset)
        if self.negative_ratio > 0:
            n_negative = int(len(notes) * self.negative_ratio)
            
            # Sample random frames without onsets
            total_frames_audio = cqt_db.shape[1]
            for _ in range(n_negative):
                # Random center frame
                center_frame = np.random.randint(context_frames, total_frames_audio - context_frames)
                
                # Extract window
                start_frame = center_frame - context_frames
                end_frame = center_frame + context_frames + 1
                cqt_window = cqt_db[:, start_frame:end_frame].T
                
                # Normalize
                cqt_window = (cqt_window - cqt_window.mean()) / (cqt_window.std() + 1e-8)
                cqt_window = np.nan_to_num(cqt_window, nan=0.0)
                
                # No onset
                onset = np.zeros(6, dtype=np.float32)
                pitch = np.full(6, np.nan, dtype=np.float32)  # NaN for negative samples
                
                self.samples.append({
                    'cqt': cqt_window,
                    'onset': onset,
                    'pitch': pitch,
                    'has_onset': 0
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        return {
            'cqt': torch.FloatTensor(sample['cqt']),  # [total_frames, n_bins]
            'onset': torch.FloatTensor(sample['onset']),  # [6]
            'pitch': torch.FloatTensor(sample['pitch']),  # [6]
            'has_onset': torch.tensor(sample['has_onset'])  # scalar
        }
    
    def __len__(self) -> int:
        return len(self.samples)


def test_dataset():
    """Test the dataset."""
    dataset = OwnSessionsDataset(
        root_dir='data/own_sessions',
        split='train',
        split_dir='splits'
    )
    
    print(f"\nDataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"  cqt: {sample['cqt'].shape}")
        print(f"  onset: {sample['onset'].shape}")
        print(f"  pitch: {sample['pitch'].shape}")


if __name__ == '__main__':
    test_dataset()

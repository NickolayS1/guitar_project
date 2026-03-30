# -*- coding: utf-8 -*-
"""
Video dataset for multimodal guitar transcription.

Loads video frames and synchronizes with audio labels.

Usage:
    dataset = VideoFramesDataset(
        root_dir='data/own_sessions',
        split='train',
        split_dir='splits',
        n_context_frames=7,
        fps=25
    )
"""

import csv
import os
from pathlib import Path
from typing import Dict, List

import cv2
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoConfig:
    """Video processing configuration."""
    fps = 25
    frame_width = 224
    frame_height = 224
    n_context_frames = 7  # Number of frames before current time


class AudioConfig:
    """Audio processing configuration."""
    sr = 22050
    hop_length = 512
    n_bins = 72
    bins_per_octave = 12
    fmin = 65.4
    midi_min = 36
    midi_max = 108
    context_window_ms = 150
    prediction_frames = 1


class VideoFramesDataset(Dataset):
    """
    Multimodal dataset with video frames and audio.
    
    For each note onset, loads:
    - CQT audio window [13, 72]
    - Video frames [7, 3, 224, 224]
    - Labels: onset [6], pitch [6]
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        split_dir: str = 'splits',
        n_context_frames: int = 7,
        fps: int = 25,
        negative_ratio: float = 1.0
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.n_context_frames = n_context_frames
        self.fps = fps
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
        # Check files - ONLY use video_cropped.mp4
        audio_path = session_dir / "audio.wav"
        video_path = session_dir / "video_cropped.mp4"
        labels_path = session_dir / "labels_enriched.csv"
        
        if not audio_path.exists():
            print(f"  Warning: Missing audio: {audio_path}")
            return
        
        # ONLY use video_cropped.mp4
        if not video_path.exists():
            print(f"  Warning: Missing video_cropped.mp4 (skipping session)")
            return
        
        if not labels_path.exists():
            print(f"  Warning: Missing labels: {labels_path}")
            return
        
        print(f"  Using video: {video_path.name}")
        
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
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"    Video: {video_path.name}, {video_fps:.1f} FPS, {total_video_frames} frames")
        
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
            # Audio center frame
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
            
            cqt_window = cqt_padded[:, start_frame:end_frame].T
            
            # Normalize CQT
            cqt_window = (cqt_window - cqt_window.mean()) / (cqt_window.std() + 1e-8)
            cqt_window = np.nan_to_num(cqt_window, nan=0.0)
            
            # Video frames: get frames before note onset
            video_frame_start = int(note['time_sec'] * video_fps) - self.n_context_frames
            video_frame_start = max(0, video_frame_start)
            
            frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_start)
            
            for _ in range(self.n_context_frames):
                ret, frame = cap.read()
                if not ret:
                    # Pad with last frame if video ended
                    if len(frames) > 0:
                        frames.append(frames[-1])
                    else:
                        frames.append(np.zeros((VideoConfig.frame_height, VideoConfig.frame_width, 3), dtype=np.uint8))
                else:
                    # Resize to 224x224
                    frame = cv2.resize(frame, (VideoConfig.frame_width, VideoConfig.frame_height))
                    # BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            
            # Stack frames [7, 224, 224, 3] → [7, 3, 224, 224]
            video_frames = np.stack(frames, axis=0)
            video_frames = np.transpose(video_frames, (0, 3, 1, 2))
            video_frames = video_frames.astype(np.float32) / 255.0
            
            # Create labels
            onset = np.zeros(6, dtype=np.float32)
            onset[note['string']] = 1.0
            
            pitch = np.zeros(6, dtype=np.float32)
            pitch[note['string']] = (note['midi'] - AudioConfig.midi_min) / (AudioConfig.midi_max - AudioConfig.midi_min)
            pitch = np.clip(pitch, 0, 1)
            
            self.samples.append({
                'cqt': cqt_window,
                'video': video_frames,
                'onset': onset,
                'pitch': pitch,
                'has_onset': 1
            })
        
        cap.release()
        
        # Add negative samples
        if self.negative_ratio > 0:
            n_negative = int(len(notes) * self.negative_ratio)
            total_frames_audio = cqt_db.shape[1]
            
            for _ in range(n_negative):
                center_frame = np.random.randint(context_frames, total_frames_audio - context_frames)
                
                # CQT window
                start_frame = center_frame - context_frames
                end_frame = center_frame + context_frames + 1
                cqt_window = cqt_db[:, start_frame:end_frame].T
                cqt_window = (cqt_window - cqt_window.mean()) / (cqt_window.std() + 1e-8)
                cqt_window = np.nan_to_num(cqt_window, nan=0.0)
                
                # Video frames (random)
                video_frame_start = np.random.randint(0, max(1, total_video_frames - self.n_context_frames))
                frames = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_start)
                
                for _ in range(self.n_context_frames):
                    ret, frame = cap.read()
                    if not ret:
                        if len(frames) > 0:
                            frames.append(frames[-1])
                        else:
                            frames.append(np.zeros((VideoConfig.frame_height, VideoConfig.frame_width, 3), dtype=np.uint8))
                    else:
                        frame = cv2.resize(frame, (VideoConfig.frame_width, VideoConfig.frame_height))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                
                video_frames = np.stack(frames, axis=0)
                video_frames = np.transpose(video_frames, (0, 3, 1, 2))
                video_frames = video_frames.astype(np.float32) / 255.0
                
                # No onset
                onset = np.zeros(6, dtype=np.float32)
                pitch = np.zeros(6, dtype=np.float32)
                
                self.samples.append({
                    'cqt': cqt_window,
                    'video': video_frames,
                    'onset': onset,
                    'pitch': pitch,
                    'has_onset': 0
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        return {
            'cqt': torch.FloatTensor(sample['cqt']),
            'video': torch.FloatTensor(sample['video']),
            'onset': torch.FloatTensor(sample['onset']),
            'pitch': torch.FloatTensor(sample['pitch']),
            'has_onset': torch.tensor(sample['has_onset'])
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        'cqt': torch.stack([item['cqt'] for item in batch]),
        'video': torch.stack([item['video'] for item in batch]),
        'onset': torch.stack([item['onset'] for item in batch]),
        'pitch': torch.stack([item['pitch'] for item in batch]),
        'has_onset': torch.stack([item['has_onset'] for item in batch])
    }


def test_dataset():
    """Test the dataset."""
    dataset = VideoFramesDataset(
        root_dir='data/own_sessions',
        split='train',
        split_dir='splits'
    )
    
    print(f"\nDataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"  cqt: {sample['cqt'].shape}")
        print(f"  video: {sample['video'].shape}")
        print(f"  onset: {sample['onset'].shape}")
        print(f"  pitch: {sample['pitch'].shape}")


if __name__ == '__main__':
    test_dataset()

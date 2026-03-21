#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-compute CQT spectrograms for GuitarSet dataset.

This script pre-computes CQT spectrograms and saves them to disk,
so they don't need to be computed on-the-fly during training.

Usage:
    python precompute_cqt.py --input data/guitarset --output data/guitarset/cqt_cache
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import librosa
import torch
from tqdm import tqdm


class AudioConfig:
    """Audio processing configuration."""
    sr = 22050
    hop_length = 512
    n_bins = 72
    bins_per_octave = 12
    fmin = 65.4  # C2


def compute_cqt(audio_path: Path, cache_dir: Path) -> Path:
    """
    Compute CQT for audio file and save to cache.
    
    Args:
        audio_path: Path to audio file
        cache_dir: Directory to save CQT cache
    
    Returns:
        Path to saved CQT file
    """
    # Create cache path
    cache_path = cache_dir / audio_path.with_suffix('.npy').name
    
    # Check if already cached
    if cache_path.exists():
        return cache_path
    
    # Load audio
    audio, _ = librosa.load(str(audio_path), sr=AudioConfig.sr)
    
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
    
    # Save to disk
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), cqt_db)
    
    return cache_path


def precompute_dataset(
    root_dir: str,
    output_dir: str,
    split: str = 'all'
):
    """
    Pre-compute CQT for entire dataset.
    
    Args:
        root_dir: Root directory for GuitarSet
        output_dir: Directory to save CQT cache
        split: Which split to process ('train', 'val', 'test', 'all')
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)
    
    # Get list of audio files
    audio_dir = root_path / 'audio_mono-mic'
    audio_files = list(audio_dir.glob('*.wav'))
    
    if len(audio_files) == 0:
        print(f"No audio files found in {audio_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Pre-computing CQT to {output_path}...")
    print()
    
    # Pre-compute all files
    cache_paths = []
    for audio_file in tqdm(audio_files, desc="Computing CQT"):
        cache_path = compute_cqt(audio_file, output_path)
        cache_paths.append(cache_path)
    
    print(f"\n{'='*60}")
    print(f"Pre-computation complete!")
    print(f"Cached {len(cache_paths)} files to {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Pre-compute CQT for GuitarSet')
    parser.add_argument('--input', '-i', type=str, default='data/guitarset',
                        help='Input directory with GuitarSet')
    parser.add_argument('--output', '-o', type=str, default='data/guitarset/cqt_cache',
                        help='Output directory for CQT cache')
    parser.add_argument('--split', type=str, default='all',
                        choices=['train', 'val', 'test', 'all'],
                        help='Which split to process')
    
    args = parser.parse_args()
    
    precompute_dataset(args.input, args.output, args.split)


if __name__ == '__main__':
    main()

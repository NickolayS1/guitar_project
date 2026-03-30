#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare own sessions for multimodal training.

1. Create train/val/test splits
2. Verify all files exist
3. Print dataset statistics

Usage:
    python scripts/prepare_multimodal_data.py \
        --input data/own_sessions \
        --output data/own_sessions/splits
"""

import argparse
from pathlib import Path

import yaml


def check_session_files(session_dir: Path) -> dict:
    """Check if all required files exist."""
    files = {
        'audio': (session_dir / "audio.wav").exists(),
        'video': (session_dir / "processed_224x224.mp4").exists(),
        'labels': (session_dir / "labels_enriched.csv").exists()
    }
    
    # Also check for video.mp4
    if not files['video']:
        files['video'] = (session_dir / "video.mp4").exists()
    
    return files


def create_splits(input_dir: str, output_dir: str, seed: int = 42):
    """Create train/val/test splits."""
    import random
    random.seed(seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all sessions
    session_dirs = [d for d in input_path.iterdir() if d.is_dir() and d.name != 'splits']
    
    if len(session_dirs) == 0:
        print(f"No sessions found in {input_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Found {len(session_dirs)} sessions")
    print(f"{'='*60}\n")
    
    # Check files
    print("Checking session files...")
    valid_sessions = []
    for session_dir in session_dirs:
        files = check_session_files(session_dir)
        
        if files['audio'] and files['labels']:
            valid_sessions.append(session_dir)
            
            status = "✓" if files['video'] else "⚠"
            print(f"  {status} {session_dir.name}")
            if not files['video']:
                print(f"      Warning: No video found (will use dummy video)")
        else:
            print(f"  ✗ {session_dir.name} - MISSING FILES")
            if not files['audio']:
                print(f"      Missing: audio.wav")
            if not files['labels']:
                print(f"      Missing: labels_enriched.csv")
    
    print(f"\nValid sessions: {len(valid_sessions)}")
    
    if len(valid_sessions) == 0:
        print("ERROR: No valid sessions found!")
        return
    
    # Shuffle
    random.shuffle(valid_sessions)
    
    # Split (70/15/15)
    n_train = int(len(valid_sessions) * 0.70)
    n_val = int(len(valid_sessions) * 0.15)
    n_test = len(valid_sessions) - n_train - n_val
    
    train_sessions = valid_sessions[:n_train]
    val_sessions = valid_sessions[n_train:n_train + n_val]
    test_sessions = valid_sessions[n_train + n_val:]
    
    print(f"\nSplits:")
    print(f"  Train: {len(train_sessions)} sessions")
    print(f"  Val: {len(val_sessions)} sessions")
    print(f"  Test: {len(test_sessions)} sessions")
    
    # Save splits
    for split_name, sessions in [('train', train_sessions), ('val', val_sessions), ('test', test_sessions)]:
        split_file = output_path / f"{split_name}.txt"
        
        with open(split_file, 'w') as f:
            for session_dir in sessions:
                f.write(f"{session_dir.name}\n")
        
        print(f"  Saved: {split_file}")
    
    # Create metadata
    metadata = {
        'total_sessions': len(valid_sessions),
        'train_sessions': len(train_sessions),
        'val_sessions': len(val_sessions),
        'test_sessions': len(test_sessions),
        'sessions': {
            'train': [s.name for s in train_sessions],
            'val': [s.name for s in val_sessions],
            'test': [s.name for s in test_sessions]
        }
    }
    
    metadata_path = output_path / "metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    print(f"  Saved: {metadata_path}")
    
    print(f"\n{'='*60}")
    print(f"Dataset preparation complete!")
    print(f"{'='*60}\n")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Prepare multimodal dataset')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory (data/own_sessions)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for splits')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    create_splits(args.input, args.output, args.seed)


if __name__ == '__main__':
    main()

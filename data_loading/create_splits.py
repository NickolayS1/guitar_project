#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create train/val/test splits for own sessions.

Creates:
- train.txt (70%)
- val.txt (15%)
- test.txt (15%)

Usage:
    python onsets/data/create_splits.py \
        --input data/own_sessions \
        --output data/own_sessions/splits
"""

import argparse
import random
from pathlib import Path


def create_splits(input_dir: str, output_dir: str, seed: int = 42):
    """Create train/val/test splits."""
    random.seed(seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all sessions (exclude splits directory)
    session_dirs = [d for d in input_path.iterdir() if d.is_dir() and d.name != 'splits']
    
    if len(session_dirs) == 0:
        print(f"No sessions found in {input_dir}")
        return
    
    print(f"Found {len(session_dirs)} sessions")
    
    # Shuffle
    random.shuffle(session_dirs)
    
    # Split
    n_train = int(len(session_dirs) * 0.70)
    n_val = int(len(session_dirs) * 0.15)
    n_test = len(session_dirs) - n_train - n_val
    
    train_dirs = session_dirs[:n_train]
    val_dirs = session_dirs[n_train:n_train + n_val]
    test_dirs = session_dirs[n_train + n_val:]
    
    print(f"\nSplits:")
    print(f"  Train: {len(train_dirs)} sessions")
    print(f"  Val: {len(val_dirs)} sessions")
    print(f"  Test: {len(test_dirs)} sessions")
    
    # Save splits
    for split_name, dirs in [('train', train_dirs), ('val', val_dirs), ('test', test_dirs)]:
        split_file = output_path / f"{split_name}.txt"
        
        with open(split_file, 'w') as f:
            for session_dir in dirs:
                # Write session name (relative path)
                f.write(f"{session_dir.name}\n")
        
        print(f"  Saved: {split_file}")
    
    print(f"\n{'='*60}")
    print(f"Splits created successfully!")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Create train/val/test splits')
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

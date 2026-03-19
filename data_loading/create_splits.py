#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create train/val/test splits for GuitarSet CSV annotations.

Splits:
    - train: 70%
    - val: 15%
    - test: 15%

Usage:
    python create_splits.py --input data/guitarset/csv_annotations --output data/guitarset/splits
"""

import argparse
import shutil
from pathlib import Path
import random


def create_splits(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
):
    """
    Split CSV files into train/val/test sets.
    
    Args:
        input_dir: Directory with all CSV files
        output_dir: Directory to create splits
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all CSV files
    csv_files = sorted(list(input_path.glob('*.csv')))
    
    if len(csv_files) == 0:
        print(f"No CSV files found in {input_dir}")
        return
    
    # Shuffle with seed
    random.seed(seed)
    random.shuffle(csv_files)
    
    # Calculate split sizes
    n_total = len(csv_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    # Split
    train_files = csv_files[:n_train]
    val_files = csv_files[n_train:n_train + n_val]
    test_files = csv_files[n_train + n_val:]
    
    # Create output directories
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    test_dir = output_path / 'test'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    print(f"Copying files to splits...")
    
    for f in train_files:
        shutil.copy(f, train_dir / f.name)
    print(f"  Train: {len(train_files)} files → {train_dir}")
    
    for f in val_files:
        shutil.copy(f, val_dir / f.name)
    print(f"  Val:   {len(val_files)} files → {val_dir}")
    
    for f in test_files:
        shutil.copy(f, test_dir / f.name)
    print(f"  Test:  {len(test_files)} files → {test_dir}")
    
    # Save split lists
    with open(output_path / 'train.txt', 'w') as f:
        f.write('\n'.join([x.name for x in train_files]))
    
    with open(output_path / 'val.txt', 'w') as f:
        f.write('\n'.join([x.name for x in val_files]))
    
    with open(output_path / 'test.txt', 'w') as f:
        f.write('\n'.join([x.name for x in test_files]))
    
    print(f"\nSplit lists saved to {output_path}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Split Summary")
    print(f"{'='*50}")
    print(f"Total files:  {n_total}")
    print(f"Train:        {len(train_files)} ({len(train_files)/n_total*100:.1f}%)")
    print(f"Val:          {len(val_files)} ({len(val_files)/n_total*100:.1f}%)")
    print(f"Test:         {len(test_files)} ({len(test_files)/n_total*100:.1f}%)")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description='Create train/val/test splits for GuitarSet'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/guitarset/csv_annotations',
        help='Input directory with CSV files'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/guitarset/splits',
        help='Output directory for splits'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.70,
        help='Training set ratio'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    print("="*50)
    print("Creating GuitarSet Train/Val/Test Splits")
    print("="*50 + "\n")
    
    create_splits(
        args.input,
        args.output,
        args.train_ratio,
        args.val_ratio,
        args.seed
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()

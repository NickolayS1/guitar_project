#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video preprocessing: Downscale video (preserving aspect ratio).

Usage:
    python onsets/video_preprocess.py \
        --session data/own_sessions/session_001 \
        --height 480
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def downscale_video(input_path: str, output_path: str, target_height: int = 480):
    """
    Downscale video (preserving aspect ratio).
    
    NO crop, NO rotation - just resize!
    """
    print(f"Processing: {input_path}")
    
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Get properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Original: {orig_width}x{orig_height} @ {fps}fps")
    
    # Compute downscaled dimensions (PRESERVE ASPECT RATIO)
    scale = target_height / orig_height
    down_width = int(orig_width * scale)
    down_height = target_height
    
    print(f"  Downscaled: {down_width}x{down_height} (aspect ratio preserved)")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (down_width, down_height))
    
    print(f"  Processing frames...")
    
    for _ in tqdm(range(total_frames), desc="Frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Downscale ONLY (preserving aspect ratio)
        frame = cv2.resize(frame, (down_width, down_height), interpolation=cv2.INTER_AREA)
        out.write(frame)
    
    cap.release()
    out.release()
    
    print(f"  Saved: {output_path}")


def process_session(session_path: str, target_height: int = 480):
    """Process video for a single session."""
    session_dir = Path(session_path)
    
    if not session_dir.exists():
        raise ValueError(f"Session directory not found: {session_dir}")
    
    # Find input video
    video_path = session_dir / "video.mp4"
    if not video_path.exists():
        raise ValueError(f"video.mp4 not found in {session_dir}")
    
    # Create output directory
    output_dir = session_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "processed_224x224.mp4"
    
    # Process - DOWNSCALE ONLY
    downscale_video(str(video_path), str(output_path), target_height)
    
    print("\n" + "="*60)
    print("Downscale complete!")
    print("="*60)
    print(f"\nOutput: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Downscale video (preserve aspect ratio)')
    parser.add_argument('--session', type=str, required=True, 
                        help='Session directory')
    parser.add_argument('--height', type=int, default=480, 
                        help='Target height (default: 480)')
    
    args = parser.parse_args()
    
    process_session(args.session, args.height)


if __name__ == '__main__':
    main()

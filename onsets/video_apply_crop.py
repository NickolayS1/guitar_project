#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply crop configuration to video.

Usage:
    python onsets/video_apply_crop.py \
        --session data/own_sessions/session_001
"""

import argparse
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm


def load_crop_config(session_path: str) -> dict:
    """Load crop configuration from session directory."""
    session_dir = Path(session_path)
    config_path = session_dir / "crop_config.yaml"
    
    if not config_path.exists():
        raise ValueError(f"Crop config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config['video_crop']


def apply_crop_to_video(input_path: str, output_path: str, crop_config: dict):
    """Apply crop, rotation, and resize to video."""
    x = crop_config['x']
    y = crop_config['y']
    w = crop_config['width']
    h = crop_config['height']
    rotation = crop_config['rotation_angle']
    target_w = crop_config['target_width']
    target_h = crop_config['target_height']
    
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Get actual video size
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Video size: {video_w}x{video_h}")
    print(f"  Crop: {w}x{h} at ({x}, {y})")
    print(f"  Rotation: {rotation}°")
    print(f"  Output: {target_w}x{target_h}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))
    
    for _ in tqdm(range(total_frames), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Step 1: Rotate FIRST (before crop!)
        if abs(rotation) > 0.1:
            fh, fw = frame.shape[:2]
            center = (fw / 2, fh / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
            frame = cv2.warpAffine(frame, rot_matrix, (fw, fh),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        
        # Step 2: Crop AFTER rotation
        frame = frame[y:y+h, x:x+w]
        
        # Step 3: Resize to target
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Write
        out.write(frame)
    
    cap.release()
    out.release()


def process_session(session_path: str):
    """Process single session."""
    session_dir = Path(session_path)
    
    # Load crop config
    crop_config = load_crop_config(str(session_dir))
    
    # Find input video - use processed_224x224.mp4
    input_path = session_dir / "processed_224x224.mp4"
    
    if not input_path.exists():
        raise ValueError(f"processed_224x224.mp4 not found in {session_dir}")
    
    # Output path
    output_path = session_dir / "video_cropped.mp4"
    
    # Apply crop
    apply_crop_to_video(str(input_path), str(output_path), crop_config)
    
    print(f"  Saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Apply crop to video')
    parser.add_argument('--session', type=str, required=True,
                        help='Session directory')
    
    args = parser.parse_args()
    
    process_session(args.session)


if __name__ == '__main__':
    main()

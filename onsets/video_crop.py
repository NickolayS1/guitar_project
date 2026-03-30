#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive crop region selection with sliders.

Usage:
    python onsets/video_crop.py \
        --session data/own_sessions/session_001
"""

import argparse
import time
from pathlib import Path

import cv2
import yaml


# Global variables
crop_x = 100
crop_y = 100
crop_w = 200
crop_h = 200
rotation_angle = 0.0
frame = None
window_name = 'Crop Region (adjust sliders, press Z to save)'


def rotate_frame(frame, angle):
    """Rotate frame by angle degrees."""
    h, w = frame.shape[:2]
    center = (w / 2, h / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(frame, rot_matrix, (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)


def update_display():
    """Update display with current crop and rotation."""
    global frame, crop_x, crop_y, crop_w, crop_h, rotation_angle
    
    if frame is None:
        return
    
    # Step 1: Rotate the frame FIRST
    if abs(rotation_angle) > 0.1:
        display_frame = rotate_frame(frame, rotation_angle)
    else:
        display_frame = frame.copy()
    
    # Get rotated frame dimensions
    h, w = display_frame.shape[:2]
    
    # Step 2: Clamp crop region to rotated frame bounds
    crop_x_clamped = max(0, min(crop_x, w - 1))
    crop_y_clamped = max(0, min(crop_y, h - 1))
    crop_w_clamped = max(10, min(crop_w, w - crop_x_clamped))
    crop_h_clamped = max(10, min(crop_h, h - crop_y_clamped))
    
    # Step 3: Draw crop rectangle on rotated frame
    cv2.rectangle(display_frame, 
                  (crop_x_clamped, crop_y_clamped), 
                  (crop_x_clamped + crop_w_clamped, crop_y_clamped + crop_h_clamped), 
                  (0, 255, 0), 2)
    
    # Add labels
    cv2.putText(display_frame, f'X: {crop_x_clamped}', 
               (max(0, crop_x_clamped - 50), max(20, crop_y_clamped - 10)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(display_frame, f'{crop_w_clamped}x{crop_h_clamped}', 
               (max(0, crop_x_clamped - 50), crop_y_clamped + crop_h_clamped + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(display_frame, f'Rotation: {rotation_angle:.1f}°', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Add instructions
    cv2.putText(display_frame, 'Z=Save, ESC=Quit', (10, h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow(window_name, display_frame)


def on_crop_x_change(val):
    global crop_x
    crop_x = val
    update_display()


def on_crop_y_change(val):
    global crop_y
    crop_y = val
    update_display()


def on_crop_w_change(val):
    global crop_w
    crop_w = val
    update_display()


def on_crop_h_change(val):
    global crop_h
    crop_h = val
    update_display()


def on_rotation_change(val):
    global rotation_angle
    rotation_angle = val - 50  # Center at 50, range -50 to +50
    update_display()


def select_crop_region(video_file: str):
    """Open video with sliders for crop selection."""
    global frame, crop_x, crop_y, crop_w, crop_h, rotation_angle
    
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_file}")
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Cannot read frame")
    
    height, width = frame.shape[:2]
    print(f"Original video size: {width}x{height}")
    
    # Initialize crop region (center quarter of frame)
    crop_x = width // 4
    crop_y = height // 4
    crop_w = width // 2
    crop_h = height // 2
    
    # Create window FIRST (before trackbars)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, width), min(720, height))
    
    # Show initial frame
    update_display()
    cv2.waitKey(100)
    time.sleep(0.1)
    
    # Create trackbars
    cv2.createTrackbar('X', window_name, crop_x, width, on_crop_x_change)
    cv2.createTrackbar('Y', window_name, crop_y, height, on_crop_y_change)
    cv2.createTrackbar('Width', window_name, crop_w, width, on_crop_w_change)
    cv2.createTrackbar('Height', window_name, crop_h, height, on_crop_h_change)
    cv2.createTrackbar('Rotation', window_name, 50, 100, on_rotation_change)
    
    print("\nInstructions:")
    print("  - Use sliders to adjust crop region")
    print("  - Green rectangle shows selected area on ROTATED frame")
    print("  - Press 'Z' to save, 'ESC' to quit")
    
    while True:
        update_display()
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('z') or key == ord('Z'):
            print(f"\nSaving configuration:")
            print(f"  Rotation: {rotation_angle}° (applied FIRST)")
            print(f"  Crop: {crop_w}x{crop_h} at ({crop_x}, {crop_y}) (applied AFTER rotation)")
            break
        
        elif key == 27:  # ESC
            print("Cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return None
    
    cap.release()
    cv2.destroyAllWindows()
    
    return {
        'x': int(crop_x),
        'y': int(crop_y),
        'width': int(crop_w),
        'height': int(crop_h),
        'rotation_angle': float(rotation_angle)
    }


def save_crop_config(crop_region: dict, session_path: str):
    """Save crop region to session directory."""
    session_dir = Path(session_path)
    
    config = {
        'video_crop': {
            'enabled': True,
            'x': crop_region['x'],
            'y': crop_region['y'],
            'width': crop_region['width'],
            'height': crop_region['height'],
            'rotation_angle': crop_region['rotation_angle'],
            'target_width': 224,
            'target_height': 224
        }
    }
    
    # Save to session directory
    output_path = session_dir / "crop_config.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\nSaved crop config to: {output_path}")
    print(f"\nConfig contents:")
    print(f"  Rotation: {crop_region['rotation_angle']}° (applied FIRST)")
    print(f"  Crop: {crop_region['width']}x{crop_region['height']} at ({crop_region['x']}, {crop_region['y']}) (applied AFTER rotation)")


def main():
    parser = argparse.ArgumentParser(description='Select crop region with sliders')
    parser.add_argument('--session', type=str, required=True, 
                        help='Session directory')
    
    args = parser.parse_args()
    
    session_dir = Path(args.session)
    
    # Find video - use processed_224x224.mp4
    video_path = session_dir / "processed_224x224.mp4"
    
    if not video_path.exists():
        print(f"Error: {video_path} not found")
        print("Run video_preprocess.py first!")
        return
    
    # Select region
    crop_region = select_crop_region(str(video_path))
    
    if crop_region is not None:
        # Save config
        save_crop_config(crop_region, args.session)
        
        print("\n" + "="*60)
        print("Crop region saved!")
        print("="*60)


if __name__ == '__main__':
    main()

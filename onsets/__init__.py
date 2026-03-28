# -*- coding: utf-8 -*-
"""
Onsets annotation tools.

Tools for video preprocessing, cropping, and dataset preparation.

Usage:
    # 1. Downscale video
    python onsets/video_preprocess.py \
        --session data/own_sessions/session_001 \
        --height 480
    
    # 2. Select crop region
    python onsets/video_crop.py \
        --session data/own_sessions/session_001
    
    # 3. Apply crop
    python onsets/video_apply_crop.py \
        --session data/own_sessions/session_001
    
    # 4. Prepare dataset
    python onsets/prepare_dataset.py \
        --input data/own_sessions \
        --output data/own_sessions_processed
"""

__all__ = [
    'video_preprocess',
    'video_crop',
    'video_apply_crop',
    'prepare_dataset',
    'create_splits'
]

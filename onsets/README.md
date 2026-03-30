# Onsets — Multimodal Dataset Preparation

Tools for preparing **multimodal dataset** (audio + video) for custom guitar sessions.

---

## 📋 Overview

### **Part 1: Guitar Pro → Labels**

| File | Purpose |
|------|---------|
| `generate_labels.py` | Convert `.gp5` + `video.mp4` → `labels.csv` + `audio.wav` |
| `enrich_labels.py` | Add MIDI, timing, string information |
| `update_labels.py` | Manual label correction (optional) |

### **Part 2: Video Preprocessing**

| File | Purpose |
|------|---------|
| `video_preprocess.py` | Downscale video to 480p (preserve aspect ratio) |
| `video_crop.py` | Interactive crop region selection (sliders) |
| `video_apply_crop.py` | Apply crop + rotation → 224×224 @ 25 FPS |

### **Part 3: Splits**

| File | Purpose |
|------|---------|
| `create_splits.py` | Create train/val/test splits (70/15/15) |
| `prepare_dataset.py` | Prepare dataset for training (optional) |

---

## 🚀 Quick Start

### **Requirements**

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `pyguitarpro` — read `.gp5` tablatures
- `moviepy` — extract audio from video
- `opencv-python` (with GUI!) — video processing
- `librosa` — audio processing
- `pyyaml` — configs
- `tqdm` — progress bars
- `pandas` — CSV handling

---

## 📁 Part 1: Guitar Pro → Labels

### **Input Data**

```
onsets/sessions/session_name/
├── video.mp4          # Performance video
└── piece.gp5          # GuitarPro tablature
```

### **Step 1: Generate Labels**

```bash
python onsets/generate_labels.py \
    --session onsets/sessions/session_name
```

**What it does:**
1. Extracts audio from `video.mp4` → `audio.wav`
2. Parses `piece.gp5` → notes, timing, strings
3. Creates `labels.csv` (intermediate format)
4. Creates `labels.txt` (for Audacity import)

**Output:**
```
onsets/sessions/session_name/
├── video.mp4
├── piece.gp5
├── audio.wav          ← Created
├── labels.csv         ← Created
└── labels.txt         ← Created (for Audacity)
```

---

### **Step 2: Enrich Labels**

```bash
python onsets/enrich_labels.py \
    --session onsets/sessions/session_name
```

**What it does:**
1. Reads `labels.csv`
2. Computes MIDI notes (36-108)
3. Adds time in seconds
4. Determines strings (0-5)
5. Creates `labels_enriched.csv`

**Output:**
```
onsets/sessions/session_name/
├── labels.csv
└── labels_enriched.csv  ← Created
```

**Format `labels_enriched.csv`:**
```csv
time_sec,midi,string,fret,hammer,pull_off,harmonic,grace,slide,tied,tuning
1.813,76,0,12,0,0,0,0,0,0,"E4,B3,G3,D3,A2,E2"
2.154,75,0,11,0,0,0,0,0,0,"E4,B3,G3,D3,A2,E2"
```

---

### **Step 3: Manual Correction (Optional)**

```bash
python onsets/update_labels.py \
    --session onsets/sessions/session_name
```

**What it does:**
- Opens interactive interface for label correction
- Allows fixing automatic labeling errors

---

## 📁 Part 2: Video Preprocessing

### **Input Data**

```
data/own_sessions/session_name/
├── video.mp4              # Original video (or from onsets/sessions/)
├── audio.wav              # ✓ Required
└── labels_enriched.csv    # ✓ Required
```

### **Step 4: Downscale Video**

```bash
python onsets/video_preprocess.py \
    --session data/own_sessions/session_name \
    --height 480
```

**Output:** `processed_224x224.mp4` (aspect ratio preserved)

---

### **Step 5: Select Crop Region**

```bash
python onsets/video_crop.py \
    --session data/own_sessions/session_name
```

**Interface:**
- Opens window with sliders (X, Y, Width, Height, Rotation)
- Green rectangle shows selected region
- **Z** — save, **ESC** — cancel

**Output:** `crop_config.yaml`

---

### **Step 6: Apply Crop**

```bash
python onsets/video_apply_crop.py \
    --session data/own_sessions/session_name
```

**What it does:**
1. Rotates video by angle from `crop_config.yaml`
2. Crops crop region
3. Resizes to 224×224
4. **Converts to 25 FPS** (important for multimodal model!)

**Output:** `video_cropped.mp4` (224×224, 25 FPS)

---

## 📁 Part 3: Splits

### **Step 7: Create Splits**

```bash
python onsets/create_splits.py \
    --input data/own_sessions \
    --output data/own_sessions/splits
```

**Output:**
- `splits/train.txt` (70% sessions)
- `splits/val.txt` (15% sessions)
- `splits/test.txt` (15% sessions)
- `splits/metadata.yaml`

---

## 🎯 Full Pipeline

```bash
# ============================================
# PART 1: Guitar Pro → Labels
# ============================================

# For each session with GuitarPro:
for session in onsets/sessions/*/; do
    # 1. Generate labels
    python onsets/generate_labels.py --session $session
    
    # 2. Enrich labels
    python onsets/enrich_labels.py --session $session
    
    # 3. Copy to data/own_sessions
    cp -r $session data/own_sessions/
done

# ============================================
# PART 2: Video Preprocessing
# ============================================

# For each session in data/own_sessions:
for session in data/own_sessions/*/; do
    # 4. Downscale
    python onsets/video_preprocess.py --session $session --height 480
    
    # 5. Crop
    python onsets/video_crop.py --session $session
    
    # 6. Apply crop
    python onsets/video_apply_crop.py --session $session
done

# ============================================
# PART 3: Splits
# ============================================

# 7. Create splits
python onsets/create_splits.py \
    --input data/own_sessions \
    --output data/own_sessions/splits
```

---

## 🎓 Training Usage

After data preparation:

```bash
# Train multimodal model
python training/train_multimodal.py \
    --config configs/multimodal_config.yaml \
    --audio-checkpoint experiments/enhanced_baseline/checkpoints/checkpoint_best_enhanced_baseline_v1.pth \
    --freeze-audio \
    --epochs 60
```

**Dataset:** `data_loading.video_dataset.VideoFramesDataset`
- Reads `video_cropped.mp4`
- Reads `audio.wav` → CQT (72 bins)
- Synchronizes video (25 FPS) with audio (22050 Hz)

---

## ⚠️ Troubleshooting

### **OpenCV window not opening**

```
cv2.error: The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support.
```

**Solution:**
```bash
pip uninstall opencv-python -y
pip install opencv-python  # Version with GUI
```

### **moviepy cannot extract audio**

```
OSError: Could not find a suitable audio format
```

**Solution:** Install ffmpeg:
```bash
# Windows: download from https://ffmpeg.org/download.html
# Linux: sudo apt install ffmpeg
# macOS: brew install ffmpeg
```

### **Video not 25 FPS**

`video_apply_crop.py` automatically converts to 25 FPS. Verify:

```bash
python -c "
import cv2
cap = cv2.VideoCapture('data/own_sessions/session/video_cropped.mp4')
print(f'FPS: {cap.get(cv2.CAP_PROP_FPS)}')
"
```

### **Missing `video_cropped.mp4`**

Ensure you completed all steps:
1. `video_preprocess.py` → `processed_224x224.mp4`
2. `video_crop.py` → `crop_config.yaml`
3. `video_apply_crop.py` → `video_cropped.mp4`

---

## 📊 Dataset Statistics

After preparation:

```bash
python -c "
from data_loading.video_dataset import VideoFramesDataset
dataset = VideoFramesDataset(
    root_dir='data/own_sessions',
    split='train',
    split_dir='splits'
)
print(f'Train samples: {len(dataset)}')
"
```

**Expected:** ~50-100 samples per session (with negative_ratio=3.0 → ~200-400 total)

---

## 🔗 Related Files

- **Training:** `training/train_multimodal.py`
- **Inference:** `training/inference_multimodal.py`
- **Dataset:** `data_loading/video_dataset.py`
- **Config:** `configs/multimodal_config.yaml`

# Multimodal Guitar Transcription

Audio-Video model for guitar transcription using GuitarSet and custom sessions.

---

## 🚀 Quick Start

### **1. Installation**

```bash
pip install -r requirements.txt
```

### **2. Data Preparation**

#### **Option A: GuitarSet (ready dataset)**

Download GuitarSet and place in `data/guitarset/`

#### **Option B: Custom Sessions**

```bash
# See detailed instructions in onsets/README.md

# 1. GuitarPro → labels
python onsets/generate_labels.py --session onsets/sessions/session_name
python onsets/enrich_labels.py --session onsets/sessions/session_name

# 2. Video preprocessing
python onsets/video_preprocess.py --session data/own_sessions/session_name
python onsets/video_crop.py --session data/own_sessions/session_name
python onsets/video_apply_crop.py --session data/own_sessions/session_name

# 3. Splits
python onsets/create_splits.py \
    --input data/own_sessions \
    --output data/own_sessions/splits
```

### **3. Training**

#### **Audio-only (Enhanced Baseline)**

```bash
python training/train_enhanced_baseline.py \
    --config configs/enhanced_baseline_config.yaml \
    --epochs 50
```

#### **Multimodal (Audio + Video)**

```bash
python training/train_multimodal.py \
    --config configs/multimodal_config.yaml \
    --audio-checkpoint experiments/enhanced_baseline/checkpoints/checkpoint_best_enhanced_baseline_v1.pth \
    --freeze-audio \
    --epochs 60
```

### **4. Inference**

#### **Audio-only**

```bash
python training/inference_enhanced_baseline.py \
    --checkpoint experiments/enhanced_baseline/checkpoints/checkpoint_best_enhanced_baseline_v1.pth \
    --split test \
    --dataset guitarset
```

#### **Multimodal**

```bash
# With real video
python training/inference_multimodal.py \
    --checkpoint experiments/multimodal/checkpoints/checkpoint_best.pth \
    --split test \
    --use-real-video

# With dummy video (for testing)
python training/inference_multimodal.py \
    --checkpoint experiments/multimodal/checkpoints/checkpoint_best.pth \
    --split test \
    --dummy-video
```

### **5. Peak Picking**

```bash
python evaluation/peak_picking.py \
    --predictions predictions.pt \
    --ground_truth ground_truth.pt \
    --find_best
```

---

## 📁 Project Structure

```
guitar_project/
├── README.md                        ← This file
├── requirements.txt                 ← Dependencies
│
├── onsets/                          ← Dataset preparation
│   ├── README.md                    ← Preparation instructions
│   ├── requirements.txt             ← onsets dependencies
│   ├── generate_labels.py           ← GuitarPro → labels.csv
│   ├── enrich_labels.py             ← labels.csv → labels_enriched.csv
│   ├── update_labels.py             ← Manual correction
│   ├── video_preprocess.py          ← Video downscale
│   ├── video_crop.py                ← Crop selection
│   ├── video_apply_crop.py          ← Apply crop
│   ├── create_splits.py             ← Create splits
│   └── prepare_dataset.py           ← Dataset utility
│
├── data/
│   ├── guitarset/                   ← GuitarSet dataset
│   └── own_sessions/                ← Custom sessions
│       └── splits/                  ← Train/val/test
│
├── data_loading/                    ← PyTorch datasets
│   ├── video_dataset.py             ← Multimodal dataset
│   ├── own_sessions_dataset.py      ← Audio-only dataset
│   ├── guitarset_frame_dataset.py   ← GuitarSet dataset
│   ├── dataloader.py                ← DataLoader utilities
│   └── dataset.py                   ← collate_fn
│
├── models/                          ← Models
│   ├── enhanced_baseline_cnn.py     ← Audio-only (196K params)
│   ├── multimodal_cnn.py            ← Audio + Video (6.8M params)
│   ├── video_branch.py              ← ConvNeXt + TCN
│   └── cross_attention.py           ← Fusion layer
│
├── training/                        ← Training and inference
│   ├── train_multimodal.py          ← Multimodal training
│   ├── inference_multimodal.py      ← Multimodal inference
│   └── inference_enhanced_baseline.py ← Audio-only inference
│
├── evaluation/                      ← Metrics
│   ├── metrics.py                   ← Onset F1, Pitch MAE
│   └── peak_picking.py              ← Peak picking for onsets
│
├── configs/                         ← Configurations
│   ├── enhanced_baseline_config.yaml
│   └── multimodal_config.yaml
│
└── experiments/                     ← Checkpoints
    ├── enhanced_baseline/
    └── multimodal/
```

---

## 📊 Expected Results

| Model | Dataset | Onset F1 | Pitch MAE |
|-------|---------|----------|-----------|
| Enhanced Baseline | GuitarSet | 0.42-0.45 | 1.6-1.8 |
| Enhanced Baseline | Own Sessions | 0.30-0.35 | 2.3-2.5 |
| Multimodal (real video) | Own Sessions | 0.35-0.42 | 2.0-2.5 |
| Multimodal (dummy video) | Own Sessions | 0.00* | 5.0+* |

*Requires fine-tuning with `--freeze-audio`

---

## ⚙️ Configuration

### **Enhanced Baseline**

`configs/enhanced_baseline_config.yaml`:
```yaml
training:
  batch_size: 512
  epochs: 50
  learning_rate: 0.001
  negative_ratio: 3.0
```

### **Multimodal**

`configs/multimodal_config.yaml`:
```yaml
training:
  batch_size: 8  # Smaller due to video
  epochs: 60
  learning_rate: 0.001
  negative_ratio: 3.0
```

---

## 🔗 Documentation

- **onsets/README.md** — Multimodal dataset preparation

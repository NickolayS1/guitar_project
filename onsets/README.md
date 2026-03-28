# Onsets Annotation Tools

Инструменты для аннотации и подготовки датасета.

## Быстрый старт

### 1. Downscale видео

```bash
python onsets/video_preprocess.py \
    --session data/own_sessions/session_001 \
    --height 480
```

**Результат:** `processed_224x224.mp4` (сохраняя aspect ratio)

---

### 2. Выбрать crop region

```bash
python onsets/video_crop.py \
    --session data/own_sessions/session_001
```

**Интерфейс:**
- Ползунки: X, Y, Width, Height, Rotation
- Зелёный прямоугольник показывает выбранную область
- **Z** — сохранить, **ESC** — отменить

**Результат:** `crop_config.yaml`

---

### 3. Применить crop

```bash
python onsets/video_apply_crop.py \
    --session data/own_sessions/session_001
```

**Результат:** `video_cropped.mp4` (224×224)

---

### 4. Создать сплиты

```bash
python onsets/create_splits.py \
    --input data/own_sessions \
    --output data/own_sessions/splits
```

**Результат:**
- `splits/train.txt` (70%)
- `splits/val.txt` (15%)
- `splits/test.txt` (15%)

---

## Полный пайплайн

```bash
# Для каждой сессии:
python onsets/video_preprocess.py --session data/own_sessions/session_001
python onsets/video_crop.py --session data/own_sessions/session_001
python onsets/video_apply_crop.py --session data/own_sessions/session_001

# Создать сплиты:
python onsets/create_splits.py \
    --input data/own_sessions \
    --output data/own_sessions/splits

# Обучить модель:
python training/train_multimodal.py --config configs/multimodal_config.yaml
```

---

## Структура сессии

```
data/own_sessions/session_001/
├── video.mp4              # Исходное видео
├── audio.wav              # Аудио
├── labels_enriched.csv    # Разметка
├── processed_224x224.mp4  # Downscaled (step 1)
├── crop_config.yaml       # Crop config (step 2)
└── video_cropped.mp4     # Cropped (step 3)
```

---

## Примечания

1. **Aspect ratio** сохраняется при downscale
2. **Порядок операций:**
   - **Step 1:** Rotation (поворот всего кадра)
   - **Step 2:** Crop (обрезка по окну на повёрнутом кадре)
   - **Step 3:** Resize до 224×224
3. **Координаты crop** указываются относительно ПОВЁРНУТОГО изображения
4. **Final size:** 224×224 для мультимодальной модели

### Пример:

```
Original: 852×480
Rotate:   852×480 (повернули на -10°)
Crop:     224×224 at (300, 100) на повёрнутом кадре
Resize:   224×224 (final)
```

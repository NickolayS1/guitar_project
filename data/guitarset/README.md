# GuitarSet Dataset - Подготовка данных

## 📊 Статистика конвертации

| Метрика | Значение |
|---------|----------|
| Всего JAMS файлов | 360 |
| Всего нот | 62,476 |
| Train файлы | 251 (70%) |
| Val файлы | 54 (15%) |
| Test файлы | 55 (15%) |

## 📁 Структура данных

```
data/guitarset/
├── audio_mono-mic/          # Аудио файлы (оригинальные)
│   ├── 00_BN1-129-Eb_comp_mono-mic.wav
│   └── ...
├── annotation/              # JAMS аннотации (оригинальные)
│   ├── 00_BN1-129-Eb_comp.jams
│   └── ...
├── csv_annotations/         # CSV аннотации (сконвертированные) ✨
│   ├── 00_BN1-129-Eb_comp.csv
│   └── ...
└── splits/                  # Train/Val/Test сплиты ✨
    ├── train/
    ├── val/
    ├── test/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## 🔄 Формат CSV

Конвертированные файлы имеют формат, совместимый с вашими `labels_enriched.csv`:

```csv
time_sec,midi,string,fret,hammer,pull_off,harmonic,grace,slide,tied,tuning
0.048816,51.04,2,1,0,0,0,0,0,0,"E2,A2,D3,G3,B3,E4"
0.049791,65.05,5,1,0,0,0,0,0,0,"E2,A2,D3,G3,B3,E4"
```

**Поля:**
- `time_sec`: время онсета (секунды)
- `midi`: MIDI значение ноты
- `string`: индекс струны (0 = низкая E, 5 = высокая E)
- `fret`: номер лада
- `tuning`: строй гитары

## 🚀 Использование

### 1. Конвертация (если нужно заново)

```bash
python data_loading/convert_guitarset.py \
    --input data/guitarset/annotation \
    --output data/guitarset/csv_annotations
```

### 2. Создание сплитов (если нужно заново)

```bash
python data_loading/create_splits.py \
    --input data/guitarset/csv_annotations \
    --output data/guitarset/splits
```

### 3. Загрузка в PyTorch

```python
from data_loading.guitarset_dataset import GuitarSetDataset

# Train dataset
train_dataset = GuitarSetDataset(
    root_dir='data/guitarset',
    split='train',
    sr=44100,
    n_frames=101
)

# Validation dataset
val_dataset = GuitarSetDataset(
    root_dir='data/guitarset',
    split='val',
    sr=44100,
    n_frames=101
)

# Создание DataLoader
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

### 4. Пример итерации

```python
for batch in train_loader:
    audio = batch['audio']      # [B, 1, 101, 84] - CQT спектрограмма
    onset = batch['onset']      # [B, 6] - бинарные онсеты
    pitch = batch['pitch']      # [B, 6] - нормализованный MIDI
    mask = batch['mask']        # [B, 6] - маска активных струн
    
    # Обучение модели
    onset_pred, pitch_pred = model(audio)
    loss = loss_fn(onset_pred, pitch_pred, onset, pitch, mask)
```

## 📝 Примечания

### Струны

GuitarSet использует индексацию:
- `string=0` → низкая E (E2, 6-я струна)
- `string=5` → высокая E (E4, 1-я струна)

Это соответствует вашей существующей разметке.

### MIDI значения

MIDI значения в GuitarSet могут быть дробными (например, 51.04), что отражает небольшую расстройку гитары. При обучении модели можно:
1. Округлять до целых (для classification)
2. Использовать как есть (для regression)

### Альтернативные строи

Большинство записей GuitarSet используют стандартный строй (E2,A2,D3,G3,B3,E4). Некоторые записи могут использовать альтернативные строи — информация о строе сохраняется в поле `tuning`.

## 🔧 Устранение проблем

### Ошибка: "Audio not found"

Некоторые аудио файлы могут иметь отличное название от аннотаций. Проверьте наличие файлов в `data/guitarset/audio_mono-mic/`.

### Ошибка: "No CSV files found"

Запустите конвертер заново:
```bash
python data_loading/convert_guitarset.py
```

### Пустой датасет

Убедитесь, что сплиты созданы правильно:
```bash
ls data/guitarset/splits/train/
```

## 📊 Статистика по сплитам

Для просмотра статистики:

```python
from collections import Counter

def dataset_stats(dataset):
    n_notes = len(dataset)
    n_onsets = sum(sample['onset'].sum().item() for sample in dataset)
    
    print(f"Total samples: {n_notes}")
    print(f"Total onsets: {n_onsets}")
    print(f"Average onsets per sample: {n_onsets/n_notes:.2f}")

train_dataset = GuitarSetDataset(root_dir='data/guitarset', split='train')
dataset_stats(train_dataset)
```

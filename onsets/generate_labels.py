#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guitar Label Pipeline — генерация меток для Audacity.

Вход:
    - video.mov или video.mp4
    - piece.gp5 (GuitarPro табулатура)

Выход:
    - video.mp4 (конвертация из mov при необходимости)
    - audio.wav (извлечённое аудио)
    - labels.csv (промежуточный формат)
    - labels.txt (метки для импорта в Audacity)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import csv
import shutil
from pathlib import Path
from typing import List, Tuple

import guitarpro
import librosa
import numpy as np
import pandas as pd
from moviepy import AudioFileClip


def extract_audio(video_path: Path, audio_path: Path, target_sr: int = 48000) -> Tuple[np.ndarray, int]:
    """
    Извлечь аудио из видео файла с помощью moviepy.
    """
    print(f"[1/6] Извлечение аудио из {video_path.name}...")
    
    clip = AudioFileClip(str(video_path))
    audio = clip.to_soundarray()
    audio = audio.astype(np.float32)
    
    # Конвертация в mono
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    
    # Ресемплинг
    src_sr = clip.fps
    if src_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=src_sr, target_sr=target_sr)
    
    clip.close()
    
    import soundfile as sf
    sf.write(str(audio_path), audio, target_sr)
    
    return audio, target_sr


def parse_guitarpro(gp_path: Path) -> List[Tuple[float, int, int]]:
    """
    Распарсить GuitarPro файл и извлечь все ноты.
    """
    print(f"[2/6] Парсинг GuitarPro файла {gp_path.name}...")
    
    song = guitarpro.parse(str(gp_path))
    events = []
    
    for track in song.tracks:
        for measure in track.measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    for note in beat.notes:
                        string_idx = note.string - 1  # GP: 1-6 → 0-5
                        fret = note.value
                        time_sec = beat.start / 1000.0
                        events.append((time_sec, string_idx, fret))
    
    events.sort(key=lambda x: x[0])
    print(f"  Найдено нот: {len(events)}")
    return events


def detect_onsets(audio: np.ndarray, sr: int, min_gap: float = 0.05, energy_threshold: float = 0.10) -> np.ndarray:
    """
    Детектировать онсеты в аудио.

    Args:
        audio: Аудиосигнал
        sr: Частота дискретизации
        min_gap: Минимальное расстояние между онсетами (сек)
        energy_threshold: Порог энергии онсета (относительно максимума)

    Returns:
        Массив времён онсетов (с фильтром по минимальному зазору и порогу энергии)
    """
    print("[3/6] Детекция онсетов в аудио...")

    # Onset detection через spectral flux
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)

    onset_frames = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        onset_envelope=onset_env,
        wait=1,
        pre_avg=0.03,
        post_avg=0.03
    )

    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    # Получаем энергию для каждого онсета
    onset_energies = onset_env[onset_frames] if len(onset_frames) > 0 else np.array([])
    max_energy = np.max(onset_energies) if len(onset_energies) > 0 else 1.0

    # Фильтр: онсеты не могут быть ближе min_gap И энергия выше порога
    if len(onset_times) > 1:
        filtered = []
        for i, t in enumerate(onset_times):
            # Пропускаем первый онсет если он слишком тихий (шум в начале)
            if i == 0 and onset_energies[i] < max_energy * energy_threshold:
                print(f"  Пропущен тихий онсет {t:.3f}s (энергия={onset_energies[i]/max_energy:.2%})")
                continue
            if len(filtered) == 0 or t - filtered[-1] >= min_gap:
                filtered.append(t)
        onset_times = np.array(filtered)

    print(f"  Найдено онсетов: {len(onset_times)} (после фильтра: {len(onset_times)})")
    return onset_times


def align_events(
    ideal_events: List[Tuple[float, int, int]],
    real_onsets: np.ndarray,
    search_window: float = 1.0
) -> List[Tuple[float, int, int]]:
    """
    Синхронизировать идеальные ноты с реальными онсетами.

    Алгоритм:
    1. Ищем первый онсет в окне вокруг первой ноты GP5 (±search_window)
    2. Вычисляем сдвиг по найденному онсету
    3. Все остальные ноты расставляются по временам из GP с учётом сдвига

    Args:
        ideal_events: Ноты из GuitarPro (time_sec, string, fret)
        real_onsets: Детектированные онсеты (отсортированы)
        search_window: Окно поиска первого онсета вокруг первой ноты (сек)

    Returns:
        Выровненные события (time_sec, string, fret)
    """
    print("[4-5/6] Синхронизация...")

    if len(ideal_events) == 0:
        raise ValueError("Нет событий в GuitarPro файле")
    if len(real_onsets) == 0:
        raise ValueError("Не найдено онсетов в аудио")

    # Ищем первый онсет в окне вокруг первой ноты GP5
    t_ideal_first = ideal_events[0][0]
    
    # Находим ближайший онсет к первой ноте в пределах search_window
    found_onset = None
    best_delta = float('inf')
    
    for t_onset in real_onsets:
        delta = abs(t_onset - t_ideal_first)
        if delta <= search_window and delta < best_delta:
            found_onset = t_onset
            best_delta = delta
    
    if found_onset is None:
        # Если не нашли в окне — берём первый онсет (как fallback)
        found_onset = real_onsets[0]
        print(f"  [WARN] Первый онсет не найден в окне ±{search_window}s от первой ноты GP5")
    
    t_real_first = found_onset
    shift = t_real_first - t_ideal_first

    print(f"  Первая нота: GP={t_ideal_first:.3f} -> аудио={t_real_first:.3f} (сдвиг={shift:+.3f})")

    aligned = []
    for i, (t_gp, string, fret) in enumerate(ideal_events):
        # Все ноты сдвигаем на одинаковое значение
        t_aligned = t_gp + shift
        aligned.append((t_aligned, string, fret))

    return aligned


def save_labels_csv(events: List[Tuple[float, int, int]], output_path: Path) -> None:
    """
    Сохранить аннотации в CSV формат.
    """
    print(f"[6/6] Сохранение в {output_path.name}...")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['time_sec', 'string', 'fret'])
        
        for t, string, fret in events:
            writer.writerow([f'{t:.3f}', string, fret])
    
    print(f"  labels.csv: {len(events)} записей.")


def group_notes_by_time(labels: List[dict], tolerance: float = 0.05) -> List[Tuple[float, List[dict]]]:
    """
    Сгруппировать ноты по времени (для аккордов).
    """
    if not labels:
        return []
    
    labels_sorted = sorted(labels, key=lambda x: x['time'])
    
    groups = []
    current_group = [labels_sorted[0]]
    current_time = labels_sorted[0]['time']
    
    for label in labels_sorted[1:]:
        if abs(label['time'] - current_time) <= tolerance:
            current_group.append(label)
        else:
            groups.append((current_time, current_group))
            current_group = [label]
            current_time = label['time']
    
    groups.append((current_time, current_group))
    return groups


def format_note_label(notes: List[dict]) -> str:
    """
    Форматировать метку для Audacity.
    """
    if len(notes) == 1:
        return f"S{notes[0]['string']}F{notes[0]['fret']}"
    else:
        notes_sorted = sorted(notes, key=lambda x: x['string'])
        parts = [f"S{n['string']}F{n['fret']}" for n in notes_sorted]
        return '+'.join(parts)


def save_audacity_labels(groups: List[Tuple[float, List[dict]]], output_path: Path) -> None:
    """
    Сохранить метки в формате Audacity (3 колонки: start, end, label).
    
    Время начала и конца ОДИНАКОВЫЕ (точечные метки).
    """
    print(f"Сохранение меток для Audacity в {output_path.name}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        for time, notes in groups:
            label = format_note_label(notes)
            # Время начала = время конца (точечная метка)
            f.write(f"{time:.6f}\t{time:.6f}\t{label}\n")

    print(f"  labels.txt: {len(groups)} меток.")


def run_pipeline(session_dir: Path) -> None:
    """
    Запустить полный пайплайн.
    """
    # Поиск входных файлов
    video_files = list(session_dir.glob('video.*'))
    if not video_files:
        raise FileNotFoundError(f"Не найдено видео в {session_dir}")
    video_path = video_files[0]
    
    gp_files = list(session_dir.glob('piece.gp*'))
    if not gp_files:
        raise FileNotFoundError(f"Не найден GuitarPro файл в {session_dir}")
    gp_path = gp_files[0]
    
    audio_path = session_dir / 'audio.wav'
    labels_csv_path = session_dir / 'labels.csv'
    labels_txt_path = session_dir / 'labels.txt'
    
    print(f"\n{'='*50}")
    print(f"Guitar Label Pipeline — Audacity Export")
    print(f"{'='*50}")
    print(f"Сессия: {session_dir}")
    print(f"Видео: {video_path.name}")
    print(f"Tab: {gp_path.name}")
    print(f"{'='*50}\n")
    
    # Конвертация video.mov → video.mp4
    if video_path.suffix.lower() == '.mov':
        mp4_path = session_dir / 'video.mp4'
        print(f"Konversiya {video_path.name} -> video.mp4...")
        shutil.copy(video_path, mp4_path)
        video_path = mp4_path
    
    # Извлечение аудио
    audio, sr = extract_audio(video_path, audio_path)
    
    # Парсинг GuitarPro
    ideal_events = parse_guitarpro(gp_path)
    
    # Детекция онсетов
    real_onsets = detect_onsets(audio, sr)
    
    # Синхронизация
    aligned_events = align_events(ideal_events, real_onsets, search_window=1.0)
    
    # Сохранение CSV
    save_labels_csv(aligned_events, labels_csv_path)
    
    # Конвертация в формат Audacity
    labels = [{'time': t, 'string': s, 'fret': f} for t, s, f in aligned_events]
    groups = group_notes_by_time(labels, tolerance=0.05)
    save_audacity_labels(groups, labels_txt_path)
    
    print(f"\n{'='*50}")
    print("Пайплайн завершён успешно!")
    print(f"{'='*50}")
    print(f"\nСозданные файлы:")
    print(f"  {audio_path.name} — аудио (48 kHz, mono)")
    print(f"  {labels_csv_path.name} — метки (CSV)")
    print(f"  {labels_txt_path.name} — метки для Audacity")
    print(f"\nИмпорт в Audacity:")
    print(f"  1. Открыть {audio_path.name}")
    print(f"  2. File → Import → Labels...")
    print(f"  3. Выбрать {labels_txt_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Генерация меток гитары для Audacity'
    )
    parser.add_argument(
        '--session', '-s',
        type=Path,
        required=True,
        help='Папка сессии с video.* и piece.gp*'
    )
    
    args = parser.parse_args()
    
    if not args.session.exists():
        print(f"Ошибка: папка не существует: {args.session}", file=sys.stderr)
        sys.exit(1)
    
    try:
        run_pipeline(args.session)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

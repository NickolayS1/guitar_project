#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrich Labels — дополнение разметки информацией из GuitarPro.

Вход:
    - labels_updated.csv (времени, струна, лад)
    - piece.gp5 (GuitarPro табулатура)

Выход:
    - labels_enriched.csv (расширенная разметка)

Формат labels_enriched.csv:
    time_sec,midi,string,fret,hammer,pull_off,harmonic,grace,slide,tuning
    1.813,76,0,12,0,0,0,0,0,E2,A2,D3,G3,B3,E4
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import guitarpro


def parse_guitarpro_enriched(gp_path: Path) -> Dict[Tuple[float, int, int], dict]:
    """
    Распарсить GuitarPro файл и извлечь информацию о нотах.

    Returns:
        Словарь: (time_sec, string, fret) → {midi, hammer, pull_off, harmonic, grace, slide, tied, tuning}
    """
    song = guitarpro.parse(str(gp_path))
    events = {}

    # Получаем строй для каждого трека
    track_tunings = {}
    for track in song.tracks:
        # MIDI значения для открытых струн
        tuning_midi = [s.value for s in track.strings]
        # Строй как строка (E2,A2,D3,G3,B3,E4)
        tuning_str = ','.join([midi_to_notation(m) for m in tuning_midi])
        track_tunings[track.number] = {
            'midi': tuning_midi,
            'string': tuning_str
        }

    for track in song.tracks:
        tuning = track_tunings.get(track.number, {'midi': [40, 45, 50, 55, 59, 64], 'string': 'E2,A2,D3,G3,B3,E4'})

        for measure in track.measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    for note in beat.notes:
                        string_idx = note.string - 1  # GP: 1-6 → 0-5
                        fret = note.value
                        time_sec = beat.start / 1000.0

                        # MIDI ноты: значение открытой струны + лад
                        midi = tuning['midi'][string_idx] + fret

                        # Эффекты
                        hammer = bool(note.effect.hammer) if note.effect.hammer is not None else False
                        pull_off = bool(getattr(note.effect, 'pullOff', False))
                        harmonic = bool(note.effect.isHarmonic) if note.effect.isHarmonic is not None else False
                        grace = bool(note.effect.isGrace) if note.effect.isGrace is not None else False
                        slide = len(note.effect.slides) > 0 if note.effect.slides else False
                        tied = hasattr(note.effect, 'tieNote') and note.effect.tieNote is not None

                        key = (time_sec, string_idx, fret)
                        events[key] = {
                            'midi': midi,
                            'hammer': hammer,
                            'pull_off': pull_off,
                            'harmonic': harmonic,
                            'grace': grace,
                            'slide': slide,
                            'tied': tied,
                            'tuning': tuning['string']
                        }

    return events


def parse_tuning(tuning_str: str) -> List[int]:
    """
    Распарсить строй из строки вида "E2,A2,D3,G3,B3,E4" в список MIDI.
    
    Args:
        tuning_str: Строй как строка (например, "E2,A2,D3,G3,B3,E4")
    
    Returns:
        Список MIDI значений для открытых струн
    """
    midi_map = {}
    for midi in range(128):
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_name = notes[midi % 12]
        octave = (midi // 12) - 1
        key = f"{note_name}{octave}"
        midi_map[key] = midi
    
    tuning_midi = []
    for note in tuning_str.split(','):
        note = note.strip()
        if note in midi_map:
            tuning_midi.append(midi_map[note])
        else:
            # Fallback для стандартного строя
            tuning_midi = [64, 59, 55, 50, 45, 40]
            break
    
    return tuning_midi


def midi_to_notation(midi: int) -> str:
    """
    Конвертировать MIDI номер в нотное обозначение.

    MIDI 40 = E2, 45 = A2, 50 = D3, 55 = G3, 59 = B3, 64 = E4
    """
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_name = notes[midi % 12]
    octave = (midi // 12) - 1
    return f"{note_name}{octave}"


def load_labels_updated(labels_path: Path) -> List[Tuple[float, int, int]]:
    """
    Загрузить labels_updated.csv.

    Returns:
        Список кортежей (time_sec, string, fret)
    """
    labels = []

    with open(labels_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_sec = float(row['time_sec'])
            string = int(row['string'])
            fret = int(row['fret'])
            labels.append((time_sec, string, fret))

    return labels


def match_labels_with_gp(
    labels: List[Tuple[float, int, int]],
    gp_events: Dict[Tuple[float, int, int], dict],
    gp_tuning: str,
    time_tolerance: float = 0.05
) -> List[dict]:
    """
    Сопоставить метки с данными из GP5.

    Алгоритм:
    1. Для каждой метки ищем точное совпадение (time, string, fret)
    2. Если не найдено — ищем по (string, fret) с ближайшим временем
    3. Если не найдено — используем строй из GP

    Args:
        labels: Метки из labels_updated.csv
        gp_events: Словарь событий из GP5
        gp_tuning: Строй из GP файла (например, "E2,A2,D3,G3,B3,E4")
        time_tolerance: Допуск по времени (сек)

    Returns:
        Список обогащённых меток
    """
    enriched = []

    # Создаём индекс для быстрого поиска по (string, fret)
    gp_index = {}
    for (t, s, f), data in gp_events.items():
        key = (s, f)
        if key not in gp_index:
            gp_index[key] = []
        gp_index[key].append((t, data))

    # Парсим строй из GP для вычисления MIDI по умолчанию
    tuning_midi = parse_tuning(gp_tuning)

    for time_sec, string, fret in labels:
        key_exact = (time_sec, string, fret)
        key_sf = (string, fret)

        data = None

        # 1. Точное совпадение
        if key_exact in gp_events:
            data = gp_events[key_exact]
        # 2. Совпадение по струне/ладу с ближайшим временем
        elif key_sf in gp_index:
            best_match = None
            best_delta = float('inf')
            for t_gp, d in gp_index[key_sf]:
                delta = abs(t_gp - time_sec)
                if delta < best_delta and delta <= time_tolerance:
                    best_delta = delta
                    best_match = d
            if best_match:
                data = best_match

        # 3. По умолчанию — используем строй из GP
        if data is None:
            # Вычисляем MIDI по строю из GP
            if 0 <= string < len(tuning_midi):
                midi = tuning_midi[string] + fret
            else:
                midi = 64 + fret  # fallback к E4
            
            data = {
                'midi': midi,
                'hammer': False,
                'pull_off': False,
                'harmonic': False,
                'grace': False,
                'slide': False,
                'tied': False,
                'tuning': gp_tuning  # Всегда используем строй из GP
            }
        else:
            # Обновляем tuning на актуальный из GP
            data['tuning'] = gp_tuning

        enriched.append({
            'time_sec': time_sec,
            'midi': data['midi'],
            'string': string,
            'fret': fret,
            'hammer': int(data['hammer']),
            'pull_off': int(data['pull_off']),
            'harmonic': int(data['harmonic']),
            'grace': int(data['grace']),
            'slide': int(data['slide']),
            'tied': int(data['tied']),
            'tuning': data['tuning']
        })

    return enriched


def save_enriched_csv(events: List[dict], output_path: Path) -> None:
    """
    Сохранить обогащённые метки в CSV.
    """
    print(f"Сохранение в {output_path.name}...")

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['time_sec', 'midi', 'string', 'fret', 'hammer', 'pull_off', 'harmonic', 'grace', 'slide', 'tied', 'tuning']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for event in events:
            row = {k: f"{event[k]:.6f}" if k == 'time_sec' else event[k] for k in fieldnames}
            writer.writerow(row)

    print(f"  Сохранено {len(events)} записей.")


def main():
    parser = argparse.ArgumentParser(
        description='Обогащение разметки информацией из GuitarPro'
    )
    parser.add_argument(
        '--session', '-s',
        type=Path,
        required=True,
        help='Папка сессии'
    )
    parser.add_argument(
        '--input-labels', '-i',
        type=Path,
        default=None,
        help='Входной файл меток (по умолчанию: labels_updated.csv)'
    )
    parser.add_argument(
        '--gp-file', '-g',
        type=Path,
        default=None,
        help='GuitarPro файл (по умолчанию: piece.gp5)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Выходной файл (по умолчанию: labels_enriched.csv)'
    )

    args = parser.parse_args()

    session_dir = args.session
    input_path = args.input_labels or session_dir / 'labels_updated.csv'
    gp_path = args.gp_file or session_dir / 'piece.gp5'
    output_path = args.output or session_dir / 'labels_enriched.csv'

    # Проверка файлов
    if not input_path.exists():
        print(f"[ERROR] Файл не найден: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not gp_path.exists():
        print(f"[ERROR] GuitarPro файл не найден: {gp_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"Enrich Labels — дополнение разметки")
    print(f"{'='*50}")
    print(f"Сессия: {session_dir}")
    print(f"Вход: {input_path.name}")
    print(f"GP5: {gp_path.name}")
    print(f"Выход: {output_path.name}")
    print(f"{'='*50}\n")

    # Загрузка меток
    print(f"Загрузка меток из {input_path.name}...")
    labels = load_labels_updated(input_path)
    print(f"  Найдено: {len(labels)} записей")

    # Парсинг GP5
    print(f"Парсинг {gp_path.name}...")
    gp_events = parse_guitarpro_enriched(gp_path)
    print(f"  Найдено нот в GP5: {len(gp_events)}")
    
    # Получаем строй из первой записи gp_events
    gp_tuning = 'E2,A2,D3,G3,B3,E4'  # default
    if gp_events:
        first_event = next(iter(gp_events.values()))
        gp_tuning = first_event.get('tuning', 'E2,A2,D3,G3,B3,E4')
    print(f"  Строй: {gp_tuning}")

    # Сопоставление
    print(f"Сопоставление меток с GP5...")
    enriched = match_labels_with_gp(labels, gp_events, gp_tuning)
    print(f"  Обогащено: {len(enriched)} записей")

    # Сохранение
    save_enriched_csv(enriched, output_path)

    print(f"\n{'='*50}")
    print("Готово!")
    print(f"{'='*50}")

    # Статистика
    stats = {
        'hammer': sum(1 for e in enriched if e['hammer']),
        'pull_off': sum(1 for e in enriched if e['pull_off']),
        'harmonic': sum(1 for e in enriched if e['harmonic']),
        'grace': sum(1 for e in enriched if e['grace']),
        'slide': sum(1 for e in enriched if e['slide']),
        'tied': sum(1 for e in enriched if e['tied'])
    }
    print(f"\nСтатистика эффектов:")
    for effect, count in stats.items():
        print(f"  {effect}: {count}")


if __name__ == '__main__':
    main()

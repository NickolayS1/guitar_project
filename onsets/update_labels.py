#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update Labels from Audacity — обновление меток после редактирования в Audacity.

Вход:
    - labels.csv (оригинальные метки: time_sec, string, fret)
    - labels_edited.txt (экспорт из Audacity: start, end, label)

Выход:
    - labels_updated.csv (обновлённые метки с новым временем)

Формат labels_edited.txt (Audacity):
    время_начала \t время_конца \t комментарий
    1.813333    1.813333    S0F12
    2.154667    2.154667    S0F11+S1F12  (аккорд)

Формат labels_updated.csv:
    time_sec,string,fret
    1.813,0,12
    2.155,0,11
    2.155,1,12  (аккорд - одинаковое время)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import csv
import re
from pathlib import Path
from typing import List, Tuple, Optional


def parse_audacity_label(label: str) -> Optional[Tuple[int, int]]:
    """
    Распарсить метку вида S0F12 или S0F12+S1F11.
    
    Returns:
        Список кортежей (string, fret) или None если не удалось распарсить
    """
    notes = []
    
    # Разделяем аккорды по '+'
    parts = label.split('+')
    
    for part in parts:
        # Ожидаем формат S{n}F{n}
        match = re.match(r'S(\d+)F(\d+)', part.strip())
        if match:
            string = int(match.group(1))
            fret = int(match.group(2))
            notes.append((string, fret))
    
    return notes if notes else None


def load_audacity_labels(input_path: Path) -> List[Tuple[float, List[Tuple[int, int]]]]:
    """
    Загрузить метки из файла Audacity.
    
    Returns:
        Список кортежей (time, [(string, fret), ...])
    """
    labels = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 3:
                print(f"[WARN] Строка {line_num}: недостаточно колонок: {line}")
                continue
            
            try:
                start_time = float(parts[0])
                # end_time = float(parts[1])  # Не используем, т.к. равно start_time
                label = parts[2]
                
                notes = parse_audacity_label(label)
                if notes:
                    labels.append((start_time, notes))
                else:
                    print(f"[WARN] Строка {line_num}: не распаршена метка '{label}'")
            except ValueError as e:
                print(f"[WARN] Строка {line_num}: ошибка parsing: {e}")
    
    return labels


def load_original_labels(labels_csv_path: Path) -> List[Tuple[float, int, int]]:
    """
    Загрузить оригинальные метки из CSV.
    """
    labels = []
    
    with open(labels_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_sec = float(row['time_sec'])
            string = int(row['string'])
            fret = int(row['fret'])
            labels.append((time_sec, string, fret))
    
    return labels


def update_labels(
    original_labels: List[Tuple[float, int, int]],
    audacity_labels: List[Tuple[float, List[Tuple[int, int]]]]
) -> List[Tuple[float, int, int]]:
    """
    Обновить время в оригинальных метках на основе Audacity.

    Алгоритм:
    1. Для каждой метки из Audacity (включая аккорды) ищем соответствующие в оригинальных
    2. Обновляем время для всех нот аккорда на одинаковое значение
    3. Если нота не найдена — добавляем новую

    Args:
        original_labels: Оригинальные метки (time, string, fret)
        audacity_labels: Метки из Audacity (time, [(string, fret), ...])
                        где time одинаково для всех нот в аккорде

    Returns:
        Обновлённые метки (отсортированные по времени)
    """
    # Создаём словарь для поиска: (string, fret) → [индексы в original]
    label_index = {}
    for i, (t, s, f) in enumerate(original_labels):
        key = (s, f)
        if key not in label_index:
            label_index[key] = []
        label_index[key].append(i)
    
    # Копируем оригинальные метки
    updated = [list(x) for x in original_labels]  # mutable copies
    used_indices = set()
    
    # Обновляем время из Audacity
    for time, notes in audacity_labels:
        for string, fret in notes:
            key = (string, fret)
            
            if key in label_index:
                # Находим первый неиспользованный индекс
                for idx in label_index[key]:
                    if idx not in used_indices:
                        # Обновляем время
                        updated[idx][0] = time
                        used_indices.add(idx)
                        break
                else:
                    # Все индексы использованы — добавляем новую метку
                    updated.append([time, string, fret])
            else:
                # Метка не найдена в оригинальных — добавляем новую
                updated.append([time, string, fret])
    
    # Конвертируем обратно в кортежи и сортируем по времени
    updated = [(t, s, f) for t, s, f in updated]
    updated.sort(key=lambda x: (x[0], x[1], x[2]))
    
    return updated


def save_labels_csv(events: List[Tuple[float, int, int]], output_path: Path) -> None:
    """
    Сохранить обновлённые метки в CSV.
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['time_sec', 'string', 'fret'])
        
        for t, string, fret in events:
            writer.writerow([f'{t:.6f}', string, fret])
    
    print(f"Сохранено {len(events)} записей в {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Обновление меток после редактирования в Audacity'
    )
    parser.add_argument(
        '--session', '-s',
        type=Path,
        required=True,
        help='Папка сессии'
    )
    parser.add_argument(
        '--audacity-labels', '-a',
        type=Path,
        default=None,
        help='Файл меток из Audacity (по умолчанию: labels_edited.txt)'
    )
    parser.add_argument(
        '--original-labels', '-o',
        type=Path,
        default=None,
        help='Оригинальный файл меток (по умолчанию: labels.csv)'
    )
    parser.add_argument(
        '--output', '-O',
        type=Path,
        default=None,
        help='Выходной файл (по умолчанию: labels_updated.csv)'
    )
    
    args = parser.parse_args()
    
    # Пуя по умолчанию
    session_dir = args.session
    audacity_path = args.audacity_labels or session_dir / 'labels_edited.txt'
    original_path = args.original_labels or session_dir / 'labels.csv'
    output_path = args.output or session_dir / 'labels_updated.csv'
    
    # Проверка файлов
    if not audacity_path.exists():
        print(f"[ERROR] Файл не найден: {audacity_path}", file=sys.stderr)
        sys.exit(1)
    
    if not original_path.exists():
        print(f"[ERROR] Файл не найден: {original_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"\n{'='*50}")
    print(f"Update Labels from Audacity")
    print(f"{'='*50}")
    print(f"Сессия: {session_dir}")
    print(f"Audacity: {audacity_path.name}")
    print(f"Оригинал: {original_path.name}")
    print(f"Выход: {output_path.name}")
    print(f"{'='*50}\n")
    
    # Загрузка оригинальных меток
    print(f"Загрузка оригинальных меток...")
    original_labels = load_original_labels(original_path)
    print(f"  Найдено: {len(original_labels)} записей")
    
    # Загрузка меток из Audacity
    print(f"Загрузка меток из Audacity...")
    audacity_labels = load_audacity_labels(audacity_path)
    print(f"  Найдено: {len(audacity_labels)} записей")
    
    # Обновление
    print(f"Обновление меток...")
    updated_labels = update_labels(original_labels, audacity_labels)
    print(f"  Получено: {len(updated_labels)} записей")
    
    # Сохранение
    save_labels_csv(updated_labels, output_path)
    
    print(f"\n{'='*50}")
    print("Готово!")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()

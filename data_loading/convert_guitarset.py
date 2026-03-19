#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GuitarSet to CSV converter.

Converts GuitarSet JAMS annotations to a simple CSV format with:
    time_sec, midi, string, fret

Where:
    - time_sec: onset time in seconds
    - midi: MIDI note number
    - string: string index (0 = high E, 5 = low E)
    - fret: fret number

Usage:
    python convert_guitarset.py --input data/guitarset --output data/guitarset/csv_annotations
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# Standard tuning MIDI values for open strings (E2, A2, D3, G3, B3, E4)
# GuitarSet uses: string 0 = low E (6th), string 5 = high E (1st)
# BUT we convert to user's format: string 0 = high E (1st), string 5 = low E (6th)
STRING_MIDI_OPEN_GUITARSET = {
    0: 40,  # E2 (low E, 6th string in GuitarSet)
    1: 45,  # A2
    2: 50,  # D3
    3: 55,  # G3
    4: 59,  # B3
    5: 64,  # E4 (high E, 1st string in GuitarSet)
}

# User's format: string 0 = high E (1st string), string 5 = low E (6th string)
STRING_MIDI_OPEN_USER = {
    0: 64,  # E4 (high E, 1st string)
    1: 59,  # B3
    2: 55,  # G3
    3: 50,  # D3
    4: 45,  # A2
    5: 40,  # E2 (low E, 6th string)
}

# Frequency to MIDI conversion
A4_FREQ = 440.0
A4_MIDI = 69


def freq_to_midi(freq: float) -> float:
    """Convert frequency (Hz) to MIDI note number."""
    if freq <= 0:
        return 0
    return A4_MIDI + 12 * np.log2(freq / A4_FREQ)


def midi_to_string_fret(midi: float) -> Tuple[int, int]:
    """
    Convert MIDI note to string and fret.
    
    For guitar, the same pitch can be played on multiple strings.
    We choose the most natural fingering (lowest position).
    
    Returns:
        (string_index, fret) where string 0 = low E, string 5 = high E
    """
    midi_rounded = int(round(midi))
    
    # Find all possible string/fret combinations
    possibilities = []
    for string_idx, open_midi in STRING_MIDI_OPEN.items():
        fret = midi_rounded - open_midi
        if 0 <= fret <= 24:  # Valid fret range
            possibilities.append((string_idx, fret))
    
    if not possibilities:
        # Out of range - assign to closest string
        if midi_rounded < 40:
            return 0, 0
        else:
            return 5, midi_rounded - 64
    
    # Choose the position with lowest fret (most natural)
    # Prefer higher strings (smaller string index) for same fret
    best = min(possibilities, key=lambda x: (x[1], -x[0]))
    return best


def load_jams(jams_path: Path) -> dict:
    """Load JAMS annotation file."""
    with open(jams_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_notes_from_jams(jams_data: dict) -> List[Dict]:
    """
    Extract note events from JAMS data.
    
    GuitarSet contains annotations per string:
    - data_source: "0" to "5" indicates which string (0 = low E, 5 = high E in GuitarSet)
    - namespace: "note_midi" contains note onsets with MIDI values
    
    We convert GuitarSet string indexing to user's format:
    - GuitarSet: string 0 = low E (6th), string 5 = high E (1st)
    - User format: string 0 = high E (1st), string 5 = low E (6th)
    
    Conversion: user_string = 5 - guitarset_string
    """
    notes = []
    
    # Find all note_midi annotations with string info
    for annotation in jams_data.get('annotations', []):
        namespace = annotation.get('namespace', '')
        data_source = annotation.get('annotation_metadata', {}).get('data_source', None)
        
        # Only process note_midi with valid string number
        if namespace == 'note_midi' and data_source is not None:
            try:
                guitarset_string = int(data_source)
                if not (0 <= guitarset_string <= 5):
                    continue
            except (ValueError, TypeError):
                continue
            
            # Convert GuitarSet string index to user's format
            # GuitarSet: 0=low E (6th) → User: 5=low E (6th)
            # GuitarSet: 5=high E (1st) → User: 0=high E (1st)
            user_string = 5 - guitarset_string
            
            data = annotation.get('data', [])
            for note in data:
                time = note.get('time', 0)
                duration = note.get('duration', 0)
                midi = note.get('value', 0)
                
                # Calculate fret from MIDI and user's string
                open_string_midi = STRING_MIDI_OPEN_USER[user_string]
                fret = int(round(midi)) - open_string_midi
                
                # Skip invalid frets
                if fret < 0 or fret > 24:
                    continue
                
                notes.append({
                    'time_sec': round(time, 6),
                    'midi': round(midi, 2),
                    'string': user_string,  # Use user's string indexing
                    'fret': fret,
                    'duration': round(duration, 6)
                })
    
    # Sort by time
    notes.sort(key=lambda x: x['time_sec'])
    
    return notes


def detect_tuning(notes: List[Dict]) -> str:
    """
    Detect guitar tuning from notes.
    
    For simplicity, assume standard tuning unless bass notes suggest otherwise.
    """
    # Check lowest note
    if notes:
        min_midi = min(n['midi'] for n in notes)
        if min_midi < 40:  # Lower than standard E2
            # Could be drop D or other alternative tuning
            # For now, still use standard
            pass
    
    return "E2,A2,D3,G3,B3,E4"


def save_notes_csv(notes: List[Dict], output_path: Path, tuning: str = "E2,A2,D3,G3,B3,E4"):
    """Save notes to CSV format matching labels_enriched.csv."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['time_sec', 'midi', 'string', 'fret', 'hammer', 'pull_off', 
                      'harmonic', 'grace', 'slide', 'tied', 'tuning']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for note in notes:
            row = {
                'time_sec': note['time_sec'],
                'midi': note['midi'],
                'string': note['string'],
                'fret': note['fret'],
                'hammer': 0,
                'pull_off': 0,
                'harmonic': 0,
                'grace': 0,
                'slide': 0,
                'tied': 0,
                'tuning': tuning
            }
            writer.writerow(row)
    
    print(f"  Saved {len(notes)} notes to {output_path}")


def convert_guitarset_file(jams_path: Path, output_dir: Path) -> int:
    """
    Convert a single GuitarSet JAMS file to CSV.
    
    Returns:
        Number of notes extracted
    """
    # Load JAMS
    jams_data = load_jams(jams_path)
    
    # Extract notes
    notes = extract_notes_from_jams(jams_data)
    
    if len(notes) == 0:
        print(f"  Warning: No notes found in {jams_path}")
        return 0
    
    # Detect tuning
    tuning = detect_tuning(notes)
    
    # Create output filename
    # e.g., 00_BN1-129-Eb_comp.jams -> 00_BN1-129-Eb_comp.csv
    csv_filename = jams_path.stem + '.csv'
    output_path = output_dir / csv_filename
    
    # Save CSV
    save_notes_csv(notes, output_path, tuning)
    
    return len(notes)


def convert_guitarset_dataset(
    input_dir: str,
    output_dir: str,
    pattern: str = '**/*.jams'
) -> Dict[str, int]:
    """
    Convert entire GuitarSet dataset.
    
    Args:
        input_dir: Path to GuitarSet annotation directory
        output_dir: Path to output directory for CSV files
        pattern: Glob pattern for JAMS files
    
    Returns:
        Statistics dictionary
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all JAMS files
    jams_files = list(input_path.glob(pattern))
    
    if len(jams_files) == 0:
        print(f"No JAMS files found in {input_dir}")
        return {}
    
    print(f"Found {len(jams_files)} JAMS files")
    print(f"Converting to CSV in {output_path}...\n")
    
    stats = {
        'total_files': len(jams_files),
        'converted_files': 0,
        'total_notes': 0,
        'failed_files': []
    }
    
    for i, jams_file in enumerate(jams_files, 1):
        try:
            n_notes = convert_guitarset_file(jams_file, output_path)
            if n_notes > 0:
                stats['converted_files'] += 1
                stats['total_notes'] += n_notes
            
            if i % 50 == 0 or i == len(jams_files):
                print(f"Progress: {i}/{len(jams_files)} files")
                
        except Exception as e:
            print(f"  Error processing {jams_file}: {e}")
            stats['failed_files'].append(str(jams_file))
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Convert GuitarSet JAMS annotations to CSV format'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/guitarset/annotation',
        help='Input directory with JAMS files'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/guitarset/csv_annotations',
        help='Output directory for CSV files'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("GuitarSet to CSV Converter")
    print("="*60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print("="*60 + "\n")
    
    # Convert dataset
    stats = convert_guitarset_dataset(args.input, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("Conversion Summary")
    print("="*60)
    print(f"Total JAMS files:     {stats.get('total_files', 0)}")
    print(f"Converted files:      {stats.get('converted_files', 0)}")
    print(f"Total notes:          {stats.get('total_notes', 0):,}")
    print(f"Failed files:         {len(stats.get('failed_files', []))}")
    
    if stats.get('failed_files'):
        print("\nFailed files:")
        for f in stats['failed_files'][:10]:
            print(f"  - {f}")
    
    print("="*60)
    print("Done!")


if __name__ == '__main__':
    main()

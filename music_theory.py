from typing import List, Dict
import numpy as np

# Define base notes and their frequencies (A4 = 440Hz)
BASE_NOTES = {
    'C': 261.63,
    'C#': 277.18,
    'D': 293.66,
    'D#': 311.13,
    'E': 329.63,
    'F': 349.23,
    'F#': 369.99,
    'G': 392.00,
    'G#': 415.30,
    'A': 440.00,
    'A#': 466.16,
    'B': 493.88
}

# Define scale patterns (steps between notes)
SCALE_PATTERNS = {
    'major': [2, 2, 1, 2, 2, 2, 1],  # Whole, Whole, Half, Whole, Whole, Whole, Half
    'minor': [2, 1, 2, 2, 1, 2, 2],  # Whole, Half, Whole, Whole, Half, Whole, Whole
    'pentatonic': [2, 2, 3, 2, 3],   # Major pentatonic
    'blues': [3, 2, 1, 1, 3, 2]      # Blues scale
}

class MusicTheory:
    @staticmethod
    def get_frequency(note: str, octave: int = 4) -> float:
        """Get the frequency for a given note and octave
        
        Args:
            note: Note name (e.g., 'C', 'F#')
            octave: Octave number (default is 4)
            
        Returns:
            Frequency in Hz
        """
        base_freq = BASE_NOTES[note]
        octave_diff = octave - 4
        return base_freq * (2 ** octave_diff)

    @staticmethod
    def get_scale(root: str, scale_type: str = 'major', octave: int = 4) -> List[float]:
        """Get frequencies for all notes in a scale
        
        Args:
            root: Root note of the scale (e.g., 'C', 'F#')
            scale_type: Type of scale ('major', 'minor', 'pentatonic', 'blues')
            octave: Starting octave number
            
        Returns:
            List of frequencies for the scale
        """
        if scale_type not in SCALE_PATTERNS:
            raise ValueError(f"Unknown scale type: {scale_type}")
            
        # Get list of all notes
        notes = list(BASE_NOTES.keys())
        root_idx = notes.index(root)
        
        # Generate scale notes
        scale_notes = []
        current_idx = root_idx
        current_octave = octave
        
        # Add root note
        scale_notes.append(MusicTheory.get_frequency(notes[current_idx], current_octave))
        
        # Add remaining notes
        for step in SCALE_PATTERNS[scale_type]:
            current_idx += step
            # Handle octave wrap-around
            if current_idx >= len(notes):
                current_idx -= len(notes)
                current_octave += 1
            scale_notes.append(MusicTheory.get_frequency(notes[current_idx], current_octave))
            
        return scale_notes

    @staticmethod
    def create_progression(root: str, progression: List[int], 
                         scale_type: str = 'major', octave: int = 4) -> List[float]:
        """Create a chord progression based on scale degrees
        
        Args:
            root: Root note of the key (e.g., 'C', 'F#')
            progression: List of scale degrees (e.g., [1, 4, 5, 1] for I-IV-V-I)
            scale_type: Type of scale ('major', 'minor')
            octave: Starting octave number
            
        Returns:
            List of root note frequencies for the progression
        """
        scale = MusicTheory.get_scale(root, scale_type, octave)
        return [scale[degree - 1] for degree in progression]

    @staticmethod
    def note_to_frequency(note_str: str) -> float:
        """Convert a note string (e.g. 'A4', 'C5') to frequency
        
        Args:
            note_str: Note name with octave (e.g., 'A4', 'C#5')
            
        Returns:
            Frequency in Hz
        """
        # Split note and octave
        if len(note_str) == 2:
            note = note_str[0]
            octave = int(note_str[1])
        else:  # Handle sharps/flats
            note = note_str[:2]
            octave = int(note_str[2])
        
        return MusicTheory.get_frequency(note, octave)

# Example usage:
if __name__ == "__main__":
    # Get C major scale
    c_major = MusicTheory.get_scale('C', 'major')
    print("C major scale frequencies:", c_major)
    
    # Get a I-IV-V-I progression in C major
    progression = MusicTheory.create_progression('C', [1, 4, 5, 1])
    print("C major I-IV-V-I progression frequencies:", progression)
    
    # Get A minor pentatonic scale
    a_pent = MusicTheory.get_scale('A', 'pentatonic')
    print("A pentatonic scale frequencies:", a_pent) 
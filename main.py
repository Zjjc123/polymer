import numpy as np
from typing import List, Optional
import sounddevice as sd
import time
from music_theory import MusicTheory

class PyHouse:
    def __init__(self, bpm: int = 128, sample_rate: int = 44100):
        self.bpm = bpm
        self.sample_rate = sample_rate
        self.beat_length = 60 / bpm  # Length of one beat in seconds
        self.tracks = []

    def create_kick(self, pattern: List[int]) -> np.ndarray:
        """Create a kick drum pattern"""
        duration = 0.1  # Duration of each kick in seconds
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Create a simple kick drum sound using sine wave with exponential decay
        frequency = 150
        decay = np.exp(-10 * t)
        kick = np.sin(2 * np.pi * frequency * t) * decay
        
        # Create full pattern
        pattern_length = len(pattern)
        full_pattern = np.zeros(int(self.sample_rate * self.beat_length * pattern_length))
        
        for i, hit in enumerate(pattern):
            if hit:
                start = int(i * self.sample_rate * self.beat_length)
                end = start + len(kick)
                full_pattern[start:end] += kick
                
        return full_pattern

    def create_hihat(self, pattern: List[int]) -> np.ndarray:
        """Create a hi-hat pattern"""
        duration = 0.05  # Duration of each hi-hat in seconds
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Create a noise-based hi-hat sound
        noise = np.random.normal(0, 1, len(t))
        decay = np.exp(-30 * t)
        hihat = noise * decay
        
        # Create full pattern
        pattern_length = len(pattern)
        full_pattern = np.zeros(int(self.sample_rate * self.beat_length * pattern_length))
        
        for i, hit in enumerate(pattern):
            if hit:
                start = int(i * self.sample_rate * self.beat_length)
                end = start + len(hihat)
                full_pattern[start:end] += hihat
                
        return full_pattern

    def create_bassline(self, notes: List[float], pattern: List[int]) -> np.ndarray:
        """Create a bassline with given notes and pattern"""
        duration = self.beat_length  # Duration of each note
        pattern_length = len(pattern)
        full_pattern = np.zeros(int(self.sample_rate * self.beat_length * pattern_length))
        
        for i, (note, hit) in enumerate(zip(notes, pattern)):
            if hit:
                t = np.linspace(0, duration, int(self.sample_rate * duration))
                frequency = note
                decay = np.exp(-5 * t)
                wave = np.sin(2 * np.pi * frequency * t) * decay
                
                start = int(i * self.sample_rate * self.beat_length)
                end = start + len(wave)
                full_pattern[start:end] += wave
                
        return full_pattern

    def create_synth(self, notes: List[float], pattern: List[int], 
                    waveform: str = 'sine', attack: float = 0.01, 
                    decay: float = 0.1, sustain: float = 0.7, 
                    release: float = 0.2) -> np.ndarray:
        """Create a synth pattern with ADSR envelope
        
        Args:
            notes: List of frequencies in Hz
            pattern: List of 1s and 0s indicating when notes should play
            waveform: Type of waveform ('sine', 'square', 'sawtooth', 'triangle')
            attack: Attack time in seconds
            decay: Decay time in seconds
            sustain: Sustain level (0-1)
            release: Release time in seconds
        """
        duration = self.beat_length
        pattern_length = len(pattern)
        full_pattern = np.zeros(int(self.sample_rate * self.beat_length * pattern_length))
        
        for i, (note, hit) in enumerate(zip(notes, pattern)):
            if hit:
                # Generate time array for this note
                t = np.linspace(0, duration, int(self.sample_rate * duration))
                
                # Generate the basic waveform
                if waveform == 'sine':
                    wave = np.sin(2 * np.pi * note * t)
                elif waveform == 'square':
                    wave = np.sign(np.sin(2 * np.pi * note * t))
                elif waveform == 'sawtooth':
                    wave = 2 * (note * t % 1) - 1
                elif waveform == 'triangle':
                    wave = 2 * np.abs(2 * (note * t % 1) - 1) - 1
                else:
                    wave = np.sin(2 * np.pi * note * t)  # Default to sine
                
                # Create ADSR envelope
                attack_samples = int(attack * self.sample_rate)
                decay_samples = int(decay * self.sample_rate)
                release_samples = int(release * self.sample_rate)
                
                envelope = np.ones(len(t))
                # Attack phase
                envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
                # Decay phase
                envelope[attack_samples:attack_samples + decay_samples] = \
                    np.linspace(1, sustain, decay_samples)
                # Sustain phase is already set to sustain level
                # Release phase
                envelope[-release_samples:] = \
                    np.linspace(sustain, 0, release_samples)
                
                # Apply envelope to waveform
                wave = wave * envelope
                
                # Add to pattern
                start = int(i * self.sample_rate * self.beat_length)
                end = start + len(wave)
                full_pattern[start:end] += wave
        
        return full_pattern

    def add_track(self, track: np.ndarray):
        """Add a track to the composition"""
        self.tracks.append(track)

    def mix(self) -> np.ndarray:
        """Mix all tracks together"""
        if not self.tracks:
            return np.array([])
            
        max_length = max(len(track) for track in self.tracks)
        mixed = np.zeros(max_length)
        
        for track in self.tracks:
            mixed[:len(track)] += track
            
        # Normalize
        mixed = mixed / np.max(np.abs(mixed))
        return mixed

    def play(self):
        """Play the mixed composition"""
        mixed = self.mix()
        sd.play(mixed, self.sample_rate)
        sd.wait()

# Example usage
if __name__ == "__main__":
    # Create a new PyHouse instance at 128 BPM
    house = PyHouse(bpm=128)
    
    # Create a basic four-on-the-floor kick pattern (4 beats)
    kick_pattern = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    kick_track = house.create_kick(kick_pattern)
    
    # Create a hi-hat pattern
    hihat_pattern = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    hihat_track = house.create_hihat(hihat_pattern)
    
    # Create a simple bassline (frequencies in Hz)
    bass_notes = [55, 55, 55, 55, 62, 62, 62, 62, 59, 59, 59, 59, 65, 65, 65, 65]
    bass_pattern = [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
    bass_track = house.create_bassline(bass_notes, bass_pattern)
    
    # Create a melody using C major scale
    c_major = MusicTheory.get_scale('C', 'major')
    melody_notes = [c_major[i] for i in [0, 2, 4, 3, 2, 1, 0]] * 2  # Simple C major melody
    melody_pattern = [1, 0, 1, 0, 1, 0, 1, 0] * 2
    melody_track = house.create_synth(melody_notes, melody_pattern,
                                    waveform='triangle',
                                    attack=0.05, decay=0.1,
                                    sustain=0.5, release=0.1)
    house.add_track(melody_track)
    
    # Add all tracks
    house.add_track(kick_track)
    house.add_track(hihat_track)
    house.add_track(bass_track)
    
    # Play the composition
    house.play()

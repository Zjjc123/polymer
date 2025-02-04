import numpy as np
from typing import List, Optional
import sounddevice as sd
import time
from music_theory import MusicTheory
from effects import EffectChain, Filter, Delay, Distortion, Compressor, Reverb, Chorus

class PyHouse:
    def __init__(self, bpm: int = 128, sample_rate: int = 44100):
        self.bpm = bpm
        self.sample_rate = sample_rate
        self._update_beat_length()
        self.tracks = []
        self.track_effects = []  # List of effect chains for each track

    def _update_beat_length(self):
        """Update beat length based on current BPM"""
        self.beat_length = 60 / self.bpm

    def set_bpm(self, bpm: int):
        """Change the BPM and update all existing tracks
        
        Args:
            bpm: New beats per minute value
        """
        if bpm <= 0:
            raise ValueError("BPM must be positive")
            
        # Calculate time stretch factor
        stretch_factor = self.bpm / bpm
        
        # Update BPM and beat length
        self.bpm = bpm
        self._update_beat_length()
        
        # Time stretch all existing tracks
        for i, track in enumerate(self.tracks):
            # Calculate new length
            new_length = int(len(track) * stretch_factor)
            # Resample the track
            self.tracks[i] = np.interp(
                np.linspace(0, len(track), new_length),
                np.arange(len(track)),
                track
            )

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

    def add_track(self, track: np.ndarray, effects: Optional[EffectChain] = None):
        """Add a track to the composition with optional effects
        
        Args:
            track: Audio track data
            effects: Optional effect chain to apply to the track
        """
        self.tracks.append(track)
        self.track_effects.append(effects if effects else EffectChain())

    def mix(self) -> np.ndarray:
        """Mix all tracks together with their effects"""
        if not self.tracks:
            return np.array([])
            
        max_length = max(len(track) for track in self.tracks)
        mixed = np.zeros(max_length)
        
        for track, effects in zip(self.tracks, self.track_effects):
            # Process track through its effect chain
            processed = effects.process(track, self.sample_rate)
            mixed[:len(processed)] += processed
            
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
    house = PyHouse(bpm=256)
    
    # Create a basic four-on-the-floor kick pattern (4 beats)
    kick_pattern = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    kick_track = house.create_kick(kick_pattern)
    
    # Create a hi-hat pattern
    hihat_pattern = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    hihat_track = house.create_hihat(hihat_pattern)
    
    # Create a simple bassline (frequencies in Hz)
    bass_notes = [55, 55, 55, 55, 62, 62, 62, 62, 59, 59, 59, 59, 65, 65, 65, 65]
    bass_pattern = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
    bass_track = house.create_bassline(bass_notes, bass_pattern)
    
    # Create a melody using C major scale
    c_major = MusicTheory.get_scale('C', 'major')
    melody_notes = [c_major[i] for i in [0, 2, 4, 3, 2, 1, 0]] * 2  # Simple C major melody
    melody_pattern = [1, 0, 1, 0, 1, 0, 1, 0] * 2
    melody_track = house.create_synth(melody_notes, melody_pattern,
                                    waveform='triangle',
                                    attack=0.05, decay=0.1,
                                    sustain=0.5, release=0.1)
    
    # Create more sophisticated effect chains
    kick_effects = EffectChain()
    kick_effects.add_effect(Compressor(threshold=-20, ratio=4.0))
    kick_effects.add_effect(Filter(cutoff=100, filter_type='lowpass'))

    hihat_effects = EffectChain()
    hihat_effects.add_effect(Filter(cutoff=8000, filter_type='highpass'))
    hihat_effects.add_effect(Reverb(room_size=0.3, wet_level=0.2))

    bass_effects = EffectChain()
    bass_effects.add_effect(Filter(cutoff=500, filter_type='lowpass'))
    bass_effects.add_effect(Distortion(drive=2.0, mix=0.3))
    bass_effects.add_effect(Compressor(threshold=-15, ratio=3.0))

    melody_effects = EffectChain()
    melody_effects.add_effect(Chorus(rate=0.8, depth=0.02, voices=3))
    melody_effects.add_effect(Delay(delay_time=0.25, feedback=0.4, mix=0.3))
    melody_effects.add_effect(Reverb(room_size=0.6, wet_level=0.3))
    melody_effects.add_effect(Filter(cutoff=2000, filter_type='lowpass'))

    # Add tracks with effects
    house.add_track(kick_track, kick_effects)
    house.add_track(hihat_track, hihat_effects)
    house.add_track(bass_track, bass_effects)
    house.add_track(melody_track, melody_effects)
    
    # Play the composition
    house.play()

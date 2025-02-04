import numpy as np
from typing import List, Optional
import sounddevice as sd
from effects import EffectChain
from scipy import signal
from scipy.io import wavfile

class PolymerController:
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

    def create_snare(self, pattern: List[int]) -> np.ndarray:
        """Create a snare drum pattern"""
        duration = 0.1
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Mix noise and sine waves for snare sound
        noise = np.random.normal(0, 1, len(t))
        decay = np.exp(-20 * t)
        
        # Add two sine waves for body
        sine1 = np.sin(2 * np.pi * 200 * t)
        sine2 = np.sin(2 * np.pi * 180 * t)
        
        snare = (noise + 0.5 * sine1 + 0.5 * sine2) * decay
        
        # Create full pattern
        pattern_length = len(pattern)
        full_pattern = np.zeros(int(self.sample_rate * self.beat_length * pattern_length))
        
        for i, hit in enumerate(pattern):
            if hit:
                start = int(i * self.sample_rate * self.beat_length)
                end = start + len(snare)
                full_pattern[start:end] += snare
                
        return full_pattern

    def create_clap(self, pattern: List[int]) -> np.ndarray:
        """Create a clap sound pattern"""
        duration = 0.1
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Create multiple short noise bursts
        noise = np.random.normal(0, 1, len(t))
        
        # Create multiple decay envelopes slightly offset
        decay1 = np.exp(-30 * t)
        decay2 = np.roll(decay1, int(0.01 * self.sample_rate))
        decay3 = np.roll(decay1, int(0.02 * self.sample_rate))
        decay = decay1 + decay2 + decay3
        
        # Apply bandpass filter to noise
        clap = noise * decay
        nyquist = self.sample_rate / 2
        b, a = signal.butter(2, [1000/nyquist, 4000/nyquist], btype='band')
        clap = signal.filtfilt(b, a, clap)
        
        # Create full pattern
        pattern_length = len(pattern)
        full_pattern = np.zeros(int(self.sample_rate * self.beat_length * pattern_length))
        
        for i, hit in enumerate(pattern):
            if hit:
                start = int(i * self.sample_rate * self.beat_length)
                end = start + len(clap)
                full_pattern[start:end] += clap
                
        return full_pattern

    def create_shaker(self, pattern: List[int]) -> np.ndarray:
        """Create a shaker pattern
        
        Args:
            pattern: List of 1s and 0s indicating when shaker should play
            
        Returns:
            np.ndarray: The generated shaker pattern
        """
        duration = 0.08  # Duration of each shaker sound in seconds
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Create noise-based shaker sound
        noise = np.random.normal(0, 1, len(t))
        
        # Quick attack, quick decay envelope
        envelope = np.exp(-40 * t)  # Faster decay than hi-hat
        
        # Apply bandpass filter to create characteristic shaker sound
        nyquist = self.sample_rate / 2
        b, a = signal.butter(2, [3000/nyquist, 7000/nyquist], btype='band')
        shaker = signal.filtfilt(b, a, noise * envelope)
        
        # Create full pattern
        pattern_length = len(pattern)
        full_pattern = np.zeros(int(self.sample_rate * self.beat_length * pattern_length))
        
        for i, hit in enumerate(pattern):
            if hit:
                start = int(i * self.sample_rate * self.beat_length)
                end = start + len(shaker)
                full_pattern[start:end] += shaker
                
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

    def loop(self):
        """Loop the composition continuously without gaps"""
        mixed = self.mix()  # Mix once outside the loop
        
        # Keep track of current position in the audio
        position = 0
        
        def callback(outdata, frames, time, status):
            nonlocal position
            if status:
                print(status)
            
            # Calculate how many samples we need
            remaining = len(mixed) - position
            
            if remaining >= frames:
                # We have enough samples remaining
                outdata[:] = mixed[position:position + frames].reshape(-1, 1)
                position += frames
            else:
                # We need to wrap around to the beginning
                # First, fill with remaining samples
                outdata[:remaining] = mixed[position:].reshape(-1, 1)
                # Then fill the rest from the beginning
                outdata[remaining:] = mixed[:frames-remaining].reshape(-1, 1)
                position = frames - remaining
            
            # Reset position if we've reached the end
            if position >= len(mixed):
                position = 0
        
        # Create continuous stream
        stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            callback=callback,
            finished_callback=None
        )
        
        with stream:
            stream.start()
            while True:
                sd.sleep(100)

    def export(self, filename: str):
        """Export the mixed composition to a WAV file
        
        Args:
            filename: Path to save the WAV file
        """
        mixed = self.mix()
        # Ensure the audio is in the correct range (-1 to 1) and convert to 32-bit float
        mixed = np.clip(mixed, -1, 1).astype(np.float32)
        wavfile.write(filename, self.sample_rate, mixed)

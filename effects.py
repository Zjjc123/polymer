import numpy as np
from typing import List, Optional
from scipy import signal

class AudioEffect:
    """Base class for audio effects"""
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process audio through the effect
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal
        """
        return audio

class Filter(AudioEffect):
    def __init__(self, cutoff: float, filter_type: str = 'lowpass', resonance: float = 1.0):
        """Initialize filter
        
        Args:
            cutoff: Cutoff frequency in Hz
            filter_type: Type of filter ('lowpass', 'highpass', 'bandpass')
            resonance: Filter resonance/Q factor
        """
        self.cutoff = cutoff
        self.filter_type = filter_type
        self.resonance = resonance

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply filter to audio signal"""
        nyquist = sample_rate / 2
        normalized_cutoff = self.cutoff / nyquist
        
        # Ensure cutoff is in valid range
        normalized_cutoff = min(0.99, max(0.01, normalized_cutoff))
        
        # Create filter coefficients
        if self.filter_type == 'lowpass':
            b, a = signal.butter(2, normalized_cutoff, 'low', analog=False)
        elif self.filter_type == 'highpass':
            b, a = signal.butter(2, normalized_cutoff, 'high', analog=False)
        elif self.filter_type == 'bandpass':
            b, a = signal.butter(2, [normalized_cutoff * 0.5, normalized_cutoff], 'band', analog=False)
        
        # Apply filter
        return signal.filtfilt(b, a, audio)

class Delay(AudioEffect):
    def __init__(self, delay_time: float = 0.5, feedback: float = 0.3, mix: float = 0.5):
        """Initialize delay effect
        
        Args:
            delay_time: Delay time in seconds
            feedback: Feedback amount (0-1)
            mix: Wet/dry mix (0-1)
        """
        self.delay_time = delay_time
        self.feedback = min(0.99, max(0, feedback))
        self.mix = min(1, max(0, mix))

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply delay effect to audio signal"""
        # Calculate delay in samples
        delay_samples = int(self.delay_time * sample_rate)
        
        # Create delayed signal
        delayed = np.zeros_like(audio)
        delayed[delay_samples:] = audio[:-delay_samples]
        
        # Apply feedback
        output = audio.copy()
        current_feedback = self.feedback
        current_delay = delayed
        
        for _ in range(5):  # Limit feedback iterations
            output += current_delay * current_feedback
            current_delay = np.zeros_like(audio)
            current_delay[delay_samples:] = current_delay[:-delay_samples]
            current_feedback *= self.feedback
            
        # Apply wet/dry mix
        return (1 - self.mix) * audio + self.mix * output

class Distortion(AudioEffect):
    def __init__(self, drive: float = 1.0, mix: float = 0.5):
        """Initialize distortion effect
        
        Args:
            drive: Distortion amount (>= 1.0)
            mix: Wet/dry mix (0-1)
        """
        self.drive = max(1.0, drive)
        self.mix = min(1, max(0, mix))

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply distortion effect to audio signal"""
        # Create distorted signal using tanh for soft clipping
        distorted = np.tanh(audio * self.drive)
        
        # Normalize distorted signal
        if np.max(np.abs(distorted)) > 0:
            distorted = distorted / np.max(np.abs(distorted))
        
        # Apply wet/dry mix
        return (1 - self.mix) * audio + self.mix * distorted

class EffectChain:
    def __init__(self):
        """Initialize empty effect chain"""
        self.effects: List[AudioEffect] = []

    def add_effect(self, effect: AudioEffect):
        """Add effect to chain"""
        self.effects.append(effect)

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process audio through entire effect chain"""
        output = audio.copy()
        for effect in self.effects:
            output = effect.process(output, sample_rate)
        return output

class Reverb(AudioEffect):
    def __init__(self, room_size: float = 0.8, damping: float = 0.5, 
                 wet_level: float = 0.3, dry_level: float = 0.7):
        """Initialize reverb effect
        
        Args:
            room_size: Size of the virtual room (0-1)
            damping: High frequency damping factor (0-1)
            wet_level: Level of processed signal (0-1)
            dry_level: Level of original signal (0-1)
        """
        self.room_size = min(1, max(0, room_size))
        self.damping = min(1, max(0, damping))
        self.wet_level = min(1, max(0, wet_level))
        self.dry_level = min(1, max(0, dry_level))

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply reverb effect to audio signal"""
        # Create impulse response
        decay_length = int(sample_rate * self.room_size)
        impulse = np.random.randn(decay_length)
        decay = np.exp(-self.damping * np.linspace(0, 3, len(impulse)))
        impulse = impulse * decay
        
        # Convolve with audio
        wet = signal.convolve(audio, impulse, mode='same')
        
        # Normalize wet signal
        if np.max(np.abs(wet)) > 0:
            wet = wet / np.max(np.abs(wet))
        
        # Mix wet and dry signals
        return self.dry_level * audio + self.wet_level * wet

class Chorus(AudioEffect):
    def __init__(self, rate: float = 1.0, depth: float = 0.02, 
                 voices: int = 3, mix: float = 0.5):
        """Initialize chorus effect
        
        Args:
            rate: LFO rate in Hz
            depth: Modulation depth in seconds
            voices: Number of chorus voices
            mix: Wet/dry mix (0-1)
        """
        self.rate = rate
        self.depth = depth
        self.voices = voices
        self.mix = min(1, max(0, mix))

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply chorus effect to audio signal"""
        t = np.arange(len(audio)) / sample_rate
        output = audio.copy()
        
        for voice in range(self.voices):
            # Create LFO with random phase
            phase = np.random.random() * 2 * np.pi
            lfo = np.sin(2 * np.pi * self.rate * t + phase)
            
            # Calculate delay times
            delay_samples = (self.depth * sample_rate * lfo).astype(int)
            
            # Create delayed signal
            delayed = np.zeros_like(audio)
            for i in range(len(audio)):
                read_pos = i - delay_samples[i]
                if read_pos >= 0 and read_pos < len(audio):
                    delayed[i] = audio[read_pos]
            
            output += delayed
        
        # Normalize and mix
        output = output / (self.voices + 1)
        return (1 - self.mix) * audio + self.mix * output

class Compressor(AudioEffect):
    def __init__(self, threshold: float = -20, ratio: float = 4.0,
                 attack: float = 0.01, release: float = 0.1):
        """Initialize compressor effect
        
        Args:
            threshold: Threshold in dB
            ratio: Compression ratio (e.g., 4.0 means 4:1 compression)
            attack: Attack time in seconds
            release: Release time in seconds
        """
        self.threshold = threshold
        self.ratio = max(1.0, ratio)
        self.attack = attack
        self.release = release

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply compression to audio signal"""
        # Convert threshold to linear
        threshold_linear = 10 ** (self.threshold / 20)
        
        # Calculate envelope
        abs_audio = np.abs(audio)
        envelope = np.zeros_like(audio)
        
        # Attack and release in samples
        attack_samples = int(self.attack * sample_rate)
        release_samples = int(self.release * sample_rate)
        
        # Simple envelope follower
        for i in range(len(audio)):
            if abs_audio[i] > envelope[i-1]:
                envelope[i] = abs_audio[i]
            else:
                envelope[i] = envelope[i-1] * 0.9
        
        # Apply compression
        gain_reduction = np.ones_like(audio)
        mask = envelope > threshold_linear
        gain_reduction[mask] = (threshold_linear + 
            (envelope[mask] - threshold_linear) / self.ratio) / envelope[mask]
        
        return audio * gain_reduction

class Phaser(AudioEffect):
    def __init__(self, rate: float = 0.5, depth: float = 0.7, 
                 feedback: float = 0.5, mix: float = 0.5):
        """Initialize phaser effect
        
        Args:
            rate: LFO rate in Hz
            depth: Modulation depth (0-1)
            feedback: Feedback amount (0-1)
            mix: Wet/dry mix (0-1)
        """
        self.rate = rate
        self.depth = min(1, max(0, depth))
        self.feedback = min(0.95, max(0, feedback))
        self.mix = min(1, max(0, mix))

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply phaser effect to audio signal"""
        t = np.arange(len(audio)) / sample_rate
        
        # Create LFO
        lfo = (1 + np.sin(2 * np.pi * self.rate * t)) / 2
        
        # All-pass filter chain
        stages = 6
        allpass_freqs = np.zeros((stages, len(audio)))
        for i in range(stages):
            min_freq = 200 * (i + 1)
            max_freq = 2000 * (i + 1)
            allpass_freqs[i] = min_freq + (max_freq - min_freq) * lfo
        
        output = audio.copy()
        for i in range(stages):
            freq = allpass_freqs[i] / sample_rate
            alpha = (np.sin(2 * np.pi * freq) * (1 - self.depth)) / \
                   (np.cos(2 * np.pi * freq) + (1 + self.depth))
            
            filtered = np.zeros_like(audio)
            filtered[0] = audio[0]
            
            for j in range(1, len(audio)):
                filtered[j] = alpha * (output[j] - filtered[j-1]) + output[j-1]
            
            output = filtered + self.feedback * output
        
        # Normalize and mix
        if np.max(np.abs(output)) > 0:
            output = output / np.max(np.abs(output))
            
        return (1 - self.mix) * audio + self.mix * output 
from controller import PolymerController
from effects import EffectChain, Filter, Delay, Distortion, Compressor, Reverb, Chorus
from music_theory import MusicTheory

# Create a new PyHouse instance at 128 BPM
polymer = PolymerController(bpm=512)

shaker_pattern = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
shaker_track = polymer.create_shaker(shaker_pattern)

hihat1_pattern = [0,0,1,0,0,0,1,1,0,1,1,0,0,0,1,0,
                0,0,1,0,0,0,1,1,0,1,1,0,0,1,1,1]
hihat1_track = polymer.create_hihat(hihat1_pattern)

hihat2_pattern = [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,
                1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0]
hihat2_track = polymer.create_hihat(hihat2_pattern)

kick_pattern = [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,
                0,0,0,1,0,0,0,1,0,0,0,1,0,0,0]
kick_track = polymer.create_kick(kick_pattern)

snare_pattern = [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,
                0,0,0,1,0,0,0,0,0,0,0,1,0,0,0]
snare_track = polymer.create_snare(snare_pattern)

clap_pattern = snare_pattern
clap_track = polymer.create_clap(clap_pattern)

# Create a melodic pattern using notes from A minor scale (simplified)
melody_notes = ['A3', 'C4', 'E4', 'A4', 'G4', 'E4', 'C4', 'A3',
                'A3', 'C4', 'F4', 'A4', 'G4', 'E4', 'C4', 'E4',
                'A3', 'C4', 'E4', 'A4', 'G4', 'E4', 'C4', 'A3',
                'C4', 'E4', 'G4', 'E4', 'C4', 'A3', 'G3', 'E3']

# Convert note names to frequencies
theory = MusicTheory()
melody_frequencies = [theory.note_to_frequency(note) for note in melody_notes]

# Use same pattern as kick for testing
melody_pattern = [1] * 32

# Create a synth melody track with shorter envelope times
melody_track = polymer.create_synth(
    notes=melody_frequencies, 
    pattern=melody_pattern,
    attack=0.01,    # Very short attack
    decay=0.05,     # Short decay
    sustain=0.7,    # Moderate sustain level
    release=0.05,    # Short release
    waveform='sawtooth'  # Use square wave for melody
)

# Create bassline pattern (syncopated with kick)
bass_pattern = [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,
                1,0,0,1,0,0,1,0,0,1,0,0,1,0,1,0]

# Bass notes following the chord progression (A minor)
bass_notes = ['A2', 'A2', 'F2', 'F2', 'C2', 'C2', 'E2', 'E2',
              'A2', 'A2', 'F2', 'F2', 'C2', 'C2', 'E2', 'E2'] * 2

# Convert bass notes to frequencies
bass_frequencies = [theory.note_to_frequency(note) for note in bass_notes]

# Create bass track with appropriate envelope settings
bass_track = polymer.create_synth(
    notes=bass_frequencies,
    pattern=bass_pattern,
    attack=0.05,    # Quick attack but not too sharp
    decay=0.05,      # Medium-short decay
    sustain=0.8,    # Strong sustain
    release=0.1,    # Longer release for smoother transitions
    waveform='sine' # Sine wave for clean bass sound
)

# Add tracks with effects
polymer.add_track(kick_track)
polymer.add_track(hihat1_track)
polymer.add_track(hihat2_track)
polymer.add_track(snare_track)
polymer.add_track(clap_track)
polymer.add_track(shaker_track)
polymer.add_track(melody_track)
polymer.add_track(bass_track)  # Add the bass track

# Play the composition
polymer.loop()

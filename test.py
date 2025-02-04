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

# Create effect chains
melody_effects = EffectChain()
melody_effects.add_effect(Compressor(threshold=-15, ratio=4, attack=0.005, release=0.1))
melody_effects.add_effect(Reverb(room_size=0.3, damping=0.5, wet_level=0.2))
melody_effects.add_effect(Chorus(rate=1.5, depth=0.2, mix=0.3))

# Create melody track with effects
melody_track = polymer.create_synth(
    notes=melody_frequencies, 
    pattern=melody_pattern,
    attack=0.01,
    decay=0.05,
    sustain=0.7,
    release=0.05,
    waveform='sawtooth',
)

# Create bassline pattern (syncopated with kick)
bass_pattern = [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,
                1,0,0,1,0,0,1,0,0,1,0,0,1,0,1,0]

# Bass notes following the chord progression (A minor)
bass_notes = ['A2', 'A2', 'F2', 'F2', 'C2', 'C2', 'E2', 'E2',
              'A2', 'A2', 'F2', 'F2', 'C2', 'C2', 'E2', 'E2'] * 2

# Convert bass notes to frequencies
bass_frequencies = [theory.note_to_frequency(note) for note in bass_notes]

# Create effect chains
bass_effects = EffectChain()
bass_effects.add_effect(Filter(cutoff=500, resonance=0.1, filter_type='lowpass'))
bass_effects.add_effect(Compressor(threshold=-15, ratio=4, attack=0.01, release=0.2))
bass_effects.add_effect(Distortion(drive=0.1, mix=0.15))
bass_effects.add_effect(Reverb(room_size=0.3, damping=0.5, wet_level=0.2))

# Create bass track with effects
bass_track = polymer.create_synth(
    notes=bass_frequencies,
    pattern=bass_pattern,
    attack=0.05,
    decay=0.05,
    sustain=0.8,
    release=0.1,
    waveform='sine',
)

percussion_effects = EffectChain()
percussion_effects.add_effect(Compressor(threshold=-10, ratio=4, attack=0.005, release=0.1))
percussion_effects.add_effect(Reverb(room_size=0.3, damping=0.5, wet_level=0.2))
percussion_effects.add_effect(Chorus(rate=1.5, depth=0.2, mix=0.3))
percussion_effects.add_effect(Filter(cutoff=7000, resonance=0.1, filter_type='lowpass'))
# Add tracks with effects
polymer.add_track(kick_track, percussion_effects)
polymer.add_track(hihat1_track, percussion_effects)
polymer.add_track(hihat2_track, percussion_effects)
polymer.add_track(snare_track, percussion_effects)
polymer.add_track(clap_track, percussion_effects)
polymer.add_track(shaker_track, percussion_effects)
polymer.add_track(melody_track, melody_effects)  # Add the melody track
polymer.add_track(bass_track, bass_effects)  # Add the bass track

polymer.export('test.wav')

# Play the composition
polymer.loop()

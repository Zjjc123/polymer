from controller import PolymerController
from effects import EffectChain, Reverb, Delay, Compressor, Filter, Chorus
from music_theory import MusicTheory

# Initialize controller at high BPM for intense beat
controller = PolymerController(bpm=512)

# Create a more complex kick pattern with some variation
kick_pattern = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0]
kick = controller.create_kick(kick_pattern)

# Add compression to kick for punchiness
kick_effects = EffectChain()
kick_effects.add_effect(Compressor(threshold=-20, ratio=4.0, attack=0.01, release=0.1))

# More complex hihat pattern with velocity variation
hihat_pattern = [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
hihat = controller.create_hihat(hihat_pattern)

# Add reverb to hihat
hihat_effects = EffectChain()
hihat_effects.add_effect(Reverb(room_size=0.2, damping=0.7, wet_level=0.2, dry_level=0.8))
hihat_effects.add_effect(Filter(cutoff=5000, filter_type='highpass'))

# Add shaker for extra rhythm
shaker_pattern = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
shaker = controller.create_shaker(shaker_pattern)

shaker_effects = EffectChain()
shaker_effects.add_effect(Reverb(room_size=0.3, damping=0.6, wet_level=0.2, dry_level=0.8))

# More interesting clap pattern
clap_pattern = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0]
clap = controller.create_clap(clap_pattern)

# Add reverb and delay to clap
clap_effects = EffectChain()
clap_effects.add_effect(Reverb(room_size=0.6, damping=0.4, wet_level=0.3, dry_level=0.7))
clap_effects.add_effect(Delay(delay_time=0.16, feedback=0.4, mix=0.3))

# Create a more complex bassline in C minor
bass_notes = MusicTheory.get_scale('A', 'minor', octave=2)
bass_pattern = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]
bassline_notes = [
    bass_notes[0], bass_notes[0], bass_notes[2], bass_notes[3],
    bass_notes[2], bass_notes[2], bass_notes[0], bass_notes[0],
    bass_notes[3], bass_notes[3], bass_notes[2], bass_notes[0],
    bass_notes[0], bass_notes[2], bass_notes[3], bass_notes[2]
]
bassline = controller.create_bassline(bassline_notes, bass_pattern)

# Add compression to bassline
bass_effects = EffectChain()
bass_effects.add_effect(Compressor(threshold=-15, ratio=3.0, attack=0.05, release=0.01))
bass_effects.add_effect(Filter(cutoff=500, filter_type='lowpass', resonance=2.0))

# Add a synth lead melody
lead_notes = MusicTheory.get_scale('A', 'minor', octave=3)
lead_pattern = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0]
lead_melody_notes = [
    lead_notes[4], lead_notes[4], lead_notes[2], lead_notes[4],
    lead_notes[5], lead_notes[4], lead_notes[2], lead_notes[1],
    lead_notes[2], lead_notes[2], lead_notes[4], lead_notes[5],
    lead_notes[4], lead_notes[2], lead_notes[1], lead_notes[0]
]
lead = controller.create_synth(
    lead_melody_notes, 
    lead_pattern,
    waveform='quad_saw',
    attack=0.001,
    decay=0.01,
    sustain=0.7,
    release=0.001
)

lead_effects = EffectChain()
lead_effects.add_effect(Chorus(rate=0.8, depth=0.02, voices=3, mix=0.5))
lead_effects.add_effect(Delay(delay_time=0.32, feedback=0.4, mix=0.3))
lead_effects.add_effect(Reverb(room_size=0.4, damping=0.5, wet_level=0.3, dry_level=0.7))

# Add all tracks to the controller with different volumes
controller.add_track(kick, kick_effects, volume=0.8)  # Slightly reduced kick volume
controller.add_track(hihat, hihat_effects, volume=0.5)  # Lower hihat volume
controller.add_track(shaker, shaker_effects, volume=0.3)  # Even lower shaker volume
controller.add_track(clap, clap_effects, volume=0.7)  # Moderate clap volume
controller.add_track(bassline, bass_effects, volume=0.9)  # Strong bass presence
controller.add_track(lead, lead_effects, volume=0.75)  # Balanced lead volume

# Set master volume to 0.8 for some headroom
controller.set_master_volume(0.8)

if __name__ == "__main__":
    print("Playing house beat... Press Ctrl+C to stop")
    try:
        controller.loop()
    except KeyboardInterrupt:
        print("\nStopped playback")

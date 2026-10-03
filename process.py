import argparse
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, windows
import pyloudnorm as pyln
from pedalboard import (
    Pedalboard,
    HighpassFilter,
    PeakFilter,
    HighShelfFilter,
    LowShelfFilter,
    Compressor,
    Chorus,
    Delay,
    Reverb,
    Limiter
)

def load_audio(path, target_sr=None):
    """Loads audio, validates sample rate, and outputs float32 (channels, samples) for Pedalboard."""
    data, sr = sf.read(path, dtype='float32')
    if target_sr is not None and sr != target_sr:
        raise ValueError(f"Sample rate mismatch: {path} has {sr}Hz, expected {target_sr}Hz.")
    
    # Pedalboard expects (channels, samples)
    if data.ndim == 1:
        data = np.stack([data, data], axis=0) # duplicate mono to stereo
    else:
        data = data.T  # (samples, channels) -> (channels, samples)
    return data, sr

def compute_spectrum(signal_2ch, n_fft=4096, hop_length=1024):
    """Computes average PSD across both channels."""
    mono = np.mean(signal_2ch, axis=0)
    win = windows.hann(n_fft)
    num_frames = (len(mono) - n_fft) // hop_length
    if num_frames <= 0:
        raise ValueError("Audio clip too short.")
        
    psd = np.zeros(n_fft // 2 + 1)
    for i in range(num_frames):
        frame = mono[i * hop_length : i * hop_length + n_fft] * win
        psd += np.abs(np.fft.rfft(frame)) ** 2
    return psd / num_frames

def apply_match_eq(target_audio, orig_audio, n_fft=4096, smooth_bins=12, max_boost_db=10.0):
    """Aligns target spectral response to the original vocal profile."""
    orig_psd = compute_spectrum(orig_audio, n_fft=n_fft)
    tgt_psd = compute_spectrum(target_audio, n_fft=n_fft)
    
    eps = 1e-10
    gain_curve = np.sqrt((orig_psd + eps) / (tgt_psd + eps))
    
    # Smooth frequency curve
    kernel = np.ones(smooth_bins) / smooth_bins
    gain_curve_smooth = np.convolve(gain_curve, kernel, mode='same')
    
    # Clamp spikes
    max_gain = 10 ** (max_boost_db / 20.0)
    min_gain = 10 ** (-max_boost_db / 20.0)
    gain_clamped = np.clip(gain_curve_smooth, min_gain, max_gain)
    
    # Linear-phase FIR
    full_gain = np.concatenate([gain_clamped, gain_clamped[-2:0:-1]])
    fir_raw = np.real(np.fft.ifft(full_gain))
    fir = np.fft.fftshift(fir_raw) * windows.hann(len(fir_raw))
    fir /= np.sum(fir)
    
    # Apply per channel
    out = np.zeros_like(target_audio)
    delay = len(fir) // 2
    for ch in range(target_audio.shape[0]):
        filtered = fftconvolve(target_audio[ch], fir, mode='full')
        out[ch] = filtered[delay : delay + target_audio.shape[1]]
    return out

def build_cla_vocal_chain(
    bass_mode="upper",        # 'sub' (80Hz), 'lower' (120Hz), 'upper' (250Hz)
    bass_boost_db=2.5,        # Body/weight
    treble_boost_db=3.5,      # Sheen and air
    comp_mode="spank",        # 'push' (gentle), 'spank' (punchy 1176), 'wall' (brickwall)
    pitch_width=0.25,         # Stereo chorus / micro-pitch widening
    reverb_mix=0.18,          # Vocal chamber/hall wetness
    delay_mix=0.10,           # Quarter/slap echo wetness
    delay_time_s=0.25         # Delay time in seconds (e.g. 1/8 or 1/4 note)
):
    """Replicates the 6 active faders in Chris Lord-Alge's vocal processor."""
    chain = [
        # Clean up sub-rumble and mic thumps
        HighpassFilter(cutoff_frequency_hz=80.0)
    ]
    
    # 1. CLA Bass Module
    freq_map = {"sub": 80.0, "lower": 120.0, "upper": 240.0}
    chain.append(LowShelfFilter(cutoff_frequency_hz=freq_map.get(bass_mode, 240.0), gain_db=bass_boost_db))

    # 2. CLA Treble Module (High shelf air + 3.5kHz bite)
    chain.append(PeakFilter(cutoff_frequency_hz=3500.0, gain_db=1.5, q=1.0))
    chain.append(HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=treble_boost_db))

    # 3. CLA Compressor Module
    if comp_mode == "push":      # Gentle opto leveling
        chain.append(Compressor(threshold_db=-18.0, ratio=3.0, attack_ms=25.0, release_ms=120.0))
    elif comp_mode == "spank":   # Aggressive, punchy FET / 1176 vocal sound
        chain.append(Compressor(threshold_db=-24.0, ratio=8.0, attack_ms=5.0, release_ms=60.0))
    elif comp_mode == "wall":    # Heavy limiting
        chain.append(Compressor(threshold_db=-30.0, ratio=20.0, attack_ms=1.0, release_ms=40.0))

    # 4. CLA Pitch / Widener Module (Detuned micro-pitch & spread)
    if pitch_width > 0:
        chain.append(Chorus(rate_hz=1.0, depth=pitch_width * 0.5, mix=pitch_width))

    # 5. CLA Delay Module
    if delay_mix > 0:
        chain.append(Delay(delay_seconds=delay_time_s, feedback=0.22, mix=delay_mix))

    # 6. CLA Reverb Module (Chamber / Hall space)
    if reverb_mix > 0:
        chain.append(Reverb(room_size=0.45, damping=0.5, wet_level=reverb_mix, dry_level=1.0))

    # Ceiling safety to prevent internal saturation
    chain.append(Limiter(threshold_db=-0.5))

    return Pedalboard(chain)

def process_and_master(
    orig_vocal_path="vocals.wav",
    rvc_vocal_path="rvc_vocals.wav",
    inst_path="no_vocals.wav",
    out_vocal_path="rvc_cla_polished.wav",
    out_master_path="final_song_master.wav",
    # CLA chain knobs
    bass_mode="upper",
    bass_boost_db=2.0,
    treble_boost_db=3.0,
    comp_mode="spank",
    pitch_width=0.20,
    reverb_mix=0.15,
    delay_mix=0.08,
    delay_time_s=0.25,
):
    print("Loading tracks...")
    orig_voc, sr = load_audio(orig_vocal_path)
    rvc_voc, _ = load_audio(rvc_vocal_path, target_sr=sr)
    inst, _ = load_audio(inst_path, target_sr=sr)

    # Align lengths
    min_len = min(orig_voc.shape[1], rvc_voc.shape[1], inst.shape[1])
    orig_voc = orig_voc[:, :min_len]
    rvc_voc = rvc_voc[:, :min_len]
    inst = inst[:, :min_len]

    # Step 1: Neutralize AI artifacts and match spectral contour
    print("Step 1: Running Match EQ against original vocals...")
    eq_voc = apply_match_eq(rvc_voc, orig_voc)

    # Step 2: Run the CLA Vocals signal processing chain
    print("Step 2: Applying CLA Vocals chain (EQ, 1176 Spank Comp, Chorus, Space)...")
    cla_board = build_cla_vocal_chain(
        bass_mode=bass_mode,
        bass_boost_db=bass_boost_db,
        treble_boost_db=treble_boost_db,
        comp_mode=comp_mode,
        pitch_width=pitch_width,
        reverb_mix=reverb_mix,
        delay_mix=delay_mix,
        delay_time_s=delay_time_s,
    )
    polished_voc = cla_board(eq_voc, sr)

    # Step 3: Match LUFS loudness to the original vocal placement
    print("Step 3: Calibrating Loudness (LUFS) to anchor vocal in the mix...")
    meter = pyln.Meter(sr)
    ref_lufs = meter.integrated_loudness(orig_voc.T)
    cur_lufs = meter.integrated_loudness(polished_voc.T)
    gain_db = ref_lufs - cur_lufs
    polished_voc = polished_voc * (10 ** (gain_db / 20.0))

    # Step 4: Sum with backing track and output final files
    print("Step 4: Summing mix & running master limiter...")
    final_mix = inst + polished_voc
    
    # Peak normalization/limiting
    max_peak = np.max(np.abs(final_mix))
    if max_peak > 0.98:
        final_mix = (final_mix / max_peak) * 0.98

    # Save output (Pedalboard arrays are (channels, samples), soundfile needs (samples, channels))
    sf.write(out_vocal_path, polished_voc.T, sr, subtype='PCM_24')
    sf.write(out_master_path, final_mix.T, sr, subtype='PCM_24')
    print(f"\nDone!\n- Polished Vocal: {out_vocal_path}\n- Mastered Song:  {out_master_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-process an RVC vocal against the original and mix it with an instrumental."
    )

    # --- Input / Output paths ---
    parser.add_argument(
        "--orig-vocal", default="vocals.wav", metavar="FILE",
        help="Original (dry) vocal stem for Match EQ reference (default: vocals.wav)"
    )
    parser.add_argument(
        "--rvc-vocal", default="rvc_vocals.wav", metavar="FILE",
        help="RVC-converted vocal to process (default: rvc_vocals.wav)"
    )
    parser.add_argument(
        "--inst", default="no_vocals.wav", metavar="FILE",
        help="Instrumental backing track (default: no_vocals.wav)"
    )
    parser.add_argument(
        "--out-vocal", default="rvc_cla_polished.wav", metavar="FILE",
        help="Output path for the polished vocal stem (default: rvc_cla_polished.wav)"
    )
    parser.add_argument(
        "--out-master", default="final_song_master.wav", metavar="FILE",
        help="Output path for the final master mix (default: final_song_master.wav)"
    )

    # --- CLA vocal chain parameters ---
    parser.add_argument(
        "--bass-mode", default="upper", choices=["sub", "lower", "upper"],
        help="CLA bass module frequency band: sub (80Hz), lower (120Hz), upper (240Hz) (default: upper)"
    )
    parser.add_argument(
        "--bass-boost", type=float, default=2.0, metavar="DB",
        help="Low-shelf boost in dB (default: 2.0)"
    )
    parser.add_argument(
        "--treble-boost", type=float, default=3.0, metavar="DB",
        help="High-shelf boost in dB (default: 3.0)"
    )
    parser.add_argument(
        "--comp-mode", default="spank", choices=["push", "spank", "wall"],
        help="CLA compressor mode: push (gentle), spank (1176), wall (brickwall) (default: spank)"
    )
    parser.add_argument(
        "--pitch-width", type=float, default=0.20, metavar="W",
        help="Micro-pitch / chorus stereo width 0.0-1.0 (default: 0.20)"
    )
    parser.add_argument(
        "--reverb-mix", type=float, default=0.15, metavar="W",
        help="Reverb wet level 0.0-1.0 (default: 0.15)"
    )
    parser.add_argument(
        "--delay-mix", type=float, default=0.08, metavar="W",
        help="Delay wet level 0.0-1.0 (default: 0.08)"
    )
    parser.add_argument(
        "--delay-time", type=float, default=0.25, metavar="S",
        help="Delay time in seconds (default: 0.25)"
    )

    args = parser.parse_args()

    process_and_master(
        orig_vocal_path=args.orig_vocal,
        rvc_vocal_path=args.rvc_vocal,
        inst_path=args.inst,
        out_vocal_path=args.out_vocal,
        out_master_path=args.out_master,
        bass_mode=args.bass_mode,
        bass_boost_db=args.bass_boost,
        treble_boost_db=args.treble_boost,
        comp_mode=args.comp_mode,
        pitch_width=args.pitch_width,
        reverb_mix=args.reverb_mix,
        delay_mix=args.delay_mix,
        delay_time_s=args.delay_time,
    )
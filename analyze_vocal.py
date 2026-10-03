"""
analyze_vocal.py
----------------
Analyzes a dry lead vocal stem and writes a <filename>.conf JSON file
containing pitch statistics and recommended RVC inference parameters.

Usage:
    python analyze_vocal.py <vocal.wav> [options]

The .conf is then read by rvc_infer.py to drive RVC conversion.
"""

import argparse
import json
import math
import os
import sys
import warnings

import numpy as np
import soundfile as sf
import scipy.signal as sps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def hz_to_midi(hz):
    """Convert frequency in Hz to fractional MIDI note number (A4 = 69 = 440 Hz)."""
    return 69.0 + 12.0 * math.log2(hz / 440.0)

def midi_to_note_name(midi):
    """Return a human-readable note name for a MIDI note number."""
    semitone = int(round(midi)) % 12
    octave   = int(round(midi)) // 12 - 1
    return f"{NOTE_NAMES[semitone]}{octave}"

def hz_to_note_name(hz):
    return midi_to_note_name(hz_to_midi(hz))

def load_mono_float32(path, resample_to=16000):
    """Load audio, convert to mono float32 at resample_to Hz (for F0 analysis)."""
    data, sr = sf.read(path, dtype='float32')
    if data.ndim > 1:
        data = data.mean(axis=1)          # stereo -> mono

    # Resample if needed (lightweight polyphase)
    if sr != resample_to:
        g = math.gcd(sr, resample_to)
        data = sps.resample_poly(data, resample_to // g, sr // g).astype('float32')
        sr = resample_to

    return data, sr


def extract_f0_librosa(audio, sr, fmin=60.0, fmax=1100.0):
    """
    Extract F0 using librosa's pYIN algorithm.
    Returns (f0_hz, voiced_flag) arrays aligned to hop frames.
    """
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=2048,
            hop_length=512,
        )
    return f0, voiced_flag


def pitch_statistics(f0, voiced_flag):
    """Return a dict of useful pitch stats from voiced frames only."""
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    if len(voiced_f0) == 0:
        return None

    voiced_midi = np.array([hz_to_midi(h) for h in voiced_f0])

    stats = {
        "voiced_frame_count": int(len(voiced_f0)),
        "median_hz":  float(np.median(voiced_f0)),
        "mean_hz":    float(np.mean(voiced_f0)),
        "min_hz":     float(np.min(voiced_f0)),
        "max_hz":     float(np.max(voiced_f0)),
        "median_note": hz_to_note_name(float(np.median(voiced_f0))),
        "min_note":    hz_to_note_name(float(np.min(voiced_f0))),
        "max_note":    hz_to_note_name(float(np.max(voiced_f0))),
        "range_semitones": float(np.max(voiced_midi) - np.min(voiced_midi)),
        "median_midi": float(np.median(voiced_midi)),
    }
    return stats


def classify_voice_type(median_hz):
    """Rough SATB classification from median pitch."""
    midi = hz_to_midi(median_hz)
    if midi < 46:   return "Bass"
    if midi < 52:   return "Baritone"
    if midi < 57:   return "Tenor"
    if midi < 60:   return "Mezzo-soprano / Alto"
    if midi < 65:   return "Soprano"
    return "Unknown"


def suggest_pitch_shift(source_median_midi, target_voice_type=None):
    """
    Suggest a semitone shift to a target voice type's typical median.
    Target midpoints (rough): Bass~40, Baritone~49, Tenor~54,
                               Alto~57, Mezzo~59, Soprano~64
    Returns 0 if target_voice_type is None (user must set manually).
    """
    TARGET_MIDI = {
        "bass": 40, "baritone": 49, "tenor": 54,
        "alto": 57, "mezzo": 59, "soprano": 64,
    }
    if target_voice_type is None:
        return 0
    key = target_voice_type.lower()
    if key not in TARGET_MIDI:
        return 0
    return int(round(TARGET_MIDI[key] - source_median_midi))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_conf(
    vocal_path,
    target_voice=None,
    f0_method="rmvpe",
    index_ratio=0.75,
    filter_radius=3,
    rms_mix_rate=0.75,
    protect=0.33,
    resample_sr=0,
):
    print(f"Loading: {vocal_path}")
    audio, sr = load_mono_float32(vocal_path, resample_to=16000)
    duration_s = len(audio) / sr
    print(f"  Duration : {duration_s:.1f}s  |  Analysis SR: {sr} Hz")

    print("Extracting F0 with pYIN ...")
    f0, voiced = extract_f0_librosa(audio, sr)
    stats = pitch_statistics(f0, voiced)

    if stats is None:
        print("ERROR: No voiced frames detected. Is this actually a vocal file?")
        sys.exit(1)

    voiced_pct = 100.0 * stats["voiced_frame_count"] / len(f0)
    voice_type  = classify_voice_type(stats["median_hz"])
    pitch_shift = suggest_pitch_shift(stats["median_midi"], target_voice)

    print(f"\n  Median pitch : {stats['median_hz']:.1f} Hz  ({stats['median_note']})")
    print(f"  Pitch range  : {stats['min_note']} - {stats['max_note']}  ({stats['range_semitones']:.1f} semitones)")
    print(f"  Voice type   : {voice_type}")
    print(f"  Voiced frames: {voiced_pct:.1f}%")
    if target_voice:
        print(f"  Suggested shift to '{target_voice}': {pitch_shift:+d} semitones")
    else:
        print("  f0_up_key set to 0 -- adjust manually in the .conf if your target voice differs")

    conf = {
        "_comment": (
            "Generated by analyze_vocal.py. Edit rvc.f0_up_key (semitones) to match your RVC model's "
            "voice range before running rvc_infer.py."
        ),
        "input_vocal_path": os.path.abspath(vocal_path),

        # Source analysis (read-only metadata)
        "source_analysis": {
            "duration_s":        round(duration_s, 2),
            "median_hz":         round(stats["median_hz"], 2),
            "median_note":       stats["median_note"],
            "min_note":          stats["min_note"],
            "max_note":          stats["max_note"],
            "range_semitones":   round(stats["range_semitones"], 1),
            "voice_type":        voice_type,
            "voiced_pct":        round(voiced_pct, 1),
        },

        # RVC inference parameters -- edit these before running rvc_infer.py
        "rvc": {
            "f0_up_key":      pitch_shift,   # semitone transpose -- ADJUST THIS
            "f0_method":      f0_method,     # rmvpe | crepe | harvest | pm
            "index_ratio":    index_ratio,   # 0.0 - 1.0
            "filter_radius":  filter_radius, # median filter on F0 (higher = smoother)
            "rms_mix_rate":   rms_mix_rate,  # 0=use input loudness, 1=use model loudness
            "protect":        protect,       # protect voiceless consonants (0.0 - 0.5)
            "resample_sr":    resample_sr,   # 0 = keep source SR
        },
    }
    return conf


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a dry vocal stem and write an RVC .conf parameter file."
    )
    parser.add_argument(
        "vocal",
        help="Path to the dry lead vocal WAV file"
    )
    parser.add_argument(
        "-o", "--output", default=None, metavar="FILE",
        help="Output .conf path (default: <vocal_basename>.conf)"
    )
    parser.add_argument(
        "--target-voice", default=None, metavar="TYPE",
        choices=["bass", "baritone", "tenor", "alto", "mezzo", "soprano"],
        help="Target voice type for auto pitch-shift suggestion"
    )
    parser.add_argument(
        "--f0-method", default="rmvpe",
        choices=["rmvpe", "crepe", "harvest", "pm", "dio"],
        help="F0 extraction method to embed in the conf (default: rmvpe)"
    )
    parser.add_argument(
        "--index-ratio", type=float, default=0.75, metavar="R",
        help="Feature index mix ratio 0.0-1.0 (default: 0.75)"
    )
    parser.add_argument(
        "--filter-radius", type=int, default=3, metavar="N",
        help="Median filter radius for F0 smoothing (default: 3)"
    )
    parser.add_argument(
        "--rms-mix-rate", type=float, default=0.75, metavar="R",
        help="RMS volume envelope mix 0=input, 1=model (default: 0.75)"
    )
    parser.add_argument(
        "--protect", type=float, default=0.33, metavar="P",
        help="Protect voiceless consonants 0.0-0.5 (default: 0.33)"
    )
    parser.add_argument(
        "--resample-sr", type=int, default=0, metavar="HZ",
        help="Resample output to this SR; 0 = keep source (default: 0)"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.vocal):
        print(f"ERROR: File not found: {args.vocal}", file=sys.stderr)
        sys.exit(1)

    conf = build_conf(
        vocal_path=args.vocal,
        target_voice=args.target_voice,
        f0_method=args.f0_method,
        index_ratio=args.index_ratio,
        filter_radius=args.filter_radius,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
        resample_sr=args.resample_sr,
    )

    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(args.vocal)[0]
        out_path = base + ".conf"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(conf, f, indent=2)

    print(f"\nConf written -> {out_path}")
    print("Edit 'rvc.f0_up_key' if needed, then run:\n")
    print(f"  python rvc_infer.py --conf \"{out_path}\" --model <voice.pth> --index <voice.index>")


if __name__ == "__main__":
    main()

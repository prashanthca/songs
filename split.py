import argparse
import os
import subprocess
import sys
import tempfile
from audio_separator.separator import Separator

# Formats natively supported by libsndfile (no conversion needed)
_SNDFILE_FORMATS = {'.wav', '.flac', '.aiff', '.aif', '.ogg', '.opus', '.w64', '.rf64'}

def ensure_wav(input_path):
    """
    If input_path is not in a libsndfile-native format (e.g. .m4a, .mp3, .aac),
    convert it to a temporary WAV via ffmpeg and return the temp path.
    Otherwise return the original path unchanged.
    The caller is responsible for deleting any temp file returned.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext in _SNDFILE_FORMATS:
        return input_path, None  # no temp file created

    print(f"[convert] {ext} is not natively supported — converting to WAV via ffmpeg...")
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-ar', '44100',
        '-ac', '2',
        '-sample_fmt', 's16',
        tmp.name
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        os.unlink(tmp.name)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"ffmpeg conversion failed for: {input_path}")
    print(f"[convert] -> {tmp.name}")
    return tmp.name, tmp.name  # (path_to_use, path_to_delete)

def separate_song_layers(input_audio_path, output_dir="separated_stems", output_format="WAV"):
    os.makedirs(output_dir, exist_ok=True)

    # Auto-convert unsupported formats (e.g. .m4a) to WAV before processing
    input_audio_path, _tmp_wav = ensure_wav(input_audio_path)
    try:
        _separate_song_layers_impl(input_audio_path, output_dir, output_format)
    finally:
        if _tmp_wav and os.path.exists(_tmp_wav):
            os.unlink(_tmp_wav)

def _separate_song_layers_impl(input_audio_path, output_dir, output_format):

    # -------------------------------------------------------------
    # Step 1: High-Fidelity Instrumental & Vocal Split (BS-RoPE)
    # -------------------------------------------------------------
    print("\n--- Pass 1: Isolating Vocals and Instrumental (BS-RoPE) ---")
    sep_main = Separator(
        output_dir=output_dir,
        output_format=output_format,
        normalization_threshold=0.9
    )
    # SOTA model for vocal extraction
    sep_main.load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    stems_pass1 = sep_main.separate(input_audio_path)
    
    # Identify outputs
    inst_file = [s for s in stems_pass1 if "Instrumental" in s][0]
    raw_vocal_file = [s for s in stems_pass1 if "Vocals" in s][0]
    print(f"-> Instrumental: {inst_file}")
    print(f"-> Combined Vocals: {raw_vocal_file}")

    # -------------------------------------------------------------
    # Step 2: Separate Lead Vocals from Backing Vocals / Harmonies
    # -------------------------------------------------------------
    print("\n--- Pass 2: Splitting Lead Vocals and Backing Vocals ---")
    sep_lead = Separator(
        output_dir=output_dir,
        output_format=output_format
    )
    # Specialized karaoke/backing vocal separation model
    sep_lead.load_model("UVR-BVE-4B_SN-44100-1.pth")
    vocal_input_path = os.path.join(output_dir, raw_vocal_file)
    stems_pass2 = sep_lead.separate(vocal_input_path)
    
    lead_vocal_file = [s for s in stems_pass2 if "Instrumental" in s or "Lead" in s][0]
    backing_vocal_file = [s for s in stems_pass2 if "Vocals" in s or "Backing" in s][0]
    print(f"-> Lead Vocals (Wet): {lead_vocal_file}")
    print(f"-> Backing Vocals: {backing_vocal_file}")

    # -------------------------------------------------------------
    # Step 3: Remove Reverb/Echo from Lead Vocals for Clean RVC Input
    # -------------------------------------------------------------
    print("\n--- Pass 3: De-Reverb Lead Vocals for RVC ---")
    sep_dry = Separator(
        output_dir=output_dir,
        output_format=output_format
    )
    # De-reverb model to isolate raw, dry phonemes
    sep_dry.load_model("UVR-DeEcho-DeReverb.pth")
    lead_input_path = os.path.join(output_dir, lead_vocal_file)
    stems_pass3 = sep_dry.separate(lead_input_path)
    
    dry_lead_file = [s for s in stems_pass3 if "No Echo" in s or "Instrumental" in s][0]
    reverb_tail_file = [s for s in stems_pass3 if "Echo" in s or "Vocals" in s][0]
    print(f"-> Pure Dry Lead Vocal (Send to RVC): {dry_lead_file}")
    print(f"-> Isolated Reverb Tail: {reverb_tail_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Separate an audio file into stems: instrumental, lead vocals, backing vocals, and dry lead vocals."
    )
    parser.add_argument(
        "input",
        help="Path to the input audio file (e.g. song.m4a)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="separated_stems",
        metavar="DIR",
        help="Directory to write output stems into (default: separated_stems)"
    )
    parser.add_argument(
        "-f", "--format",
        default="WAV",
        choices=["WAV", "FLAC", "MP3"],
        metavar="FMT",
        help="Output audio format: WAV, FLAC, or MP3 (default: WAV)"
    )
    args = parser.parse_args()
    separate_song_layers(args.input, args.output_dir, args.format)
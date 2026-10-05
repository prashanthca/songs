"""
rvc_infer.py
------------
Reads a <filename>.conf (produced by analyze_vocal.py) and performs
RVC voice conversion using a .pth model file and a .index file.

Requires:  infer-rvc-python  (pip install infer-rvc-python)
           torch, soundfile, numpy, scipy

Usage:
    python rvc_infer.py --conf vocal.conf --model Chester.pth --index Chester.index

All RVC parameters come from the .conf; individual flags override them.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import soundfile as sf
import scipy.signal as sps


# ---------------------------------------------------------------------------
# RVC backend loader
# ---------------------------------------------------------------------------

# infer_rvc_python uses different pitch_algo strings than the conf
# (rmvpe -> rmvpe+ for better quality, etc.)
_ALGO_MAP = {
    "rmvpe":   "rmvpe+",
    "rmvpe+":  "rmvpe+",
    "crepe":   "crepe",
    "harvest": "harvest",
    "pm":      "pm",
    "dio":     "harvest",   # dio not supported; harvest is closest
}


def load_rvc_backend():
    """
    Import infer_rvc_python.BaseLoader.

    Install:
        pip install infer-rvc-python
    """
    try:
        from infer_rvc_python import BaseLoader
        return BaseLoader
    except ImportError as e:
        import traceback
        print("\n--- Import error (full traceback) ---", file=sys.stderr)
        traceback.print_exc()
        print("-------------------------------------", file=sys.stderr)
        print(
            f"\nERROR: Failed to import infer_rvc_python: {e}\n"
            "\nInstall with:\n"
            "    pip install infer-rvc-python\n",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def resample_audio(data, src_sr, dst_sr):
    """Polyphase resample; works on (samples,) or (channels, samples)."""
    if src_sr == dst_sr:
        return data
    g = math.gcd(src_sr, dst_sr)
    return sps.resample_poly(data, dst_sr // g, src_sr // g, axis=-1).astype('float32')


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run_rvc(
    conf_path,
    model_path,
    index_path,
    output_path=None,
    # Per-run overrides (None = use conf value)
    f0_up_key=None,
    f0_method=None,
    index_ratio=None,
    filter_radius=None,
    rms_mix_rate=None,
    protect=None,
    resample_sr=None,
    device="cpu",
):
    # ------------------------------------------------------------------
    # 1. Load conf
    # ------------------------------------------------------------------
    if not os.path.isfile(conf_path):
        print(f"ERROR: conf not found: {conf_path}", file=sys.stderr)
        sys.exit(1)

    with open(conf_path, "r", encoding="utf-8") as f:
        conf = json.load(f)

    rvc_params = conf.get("rvc", {})
    input_path = conf.get("input_vocal_path", "")

    # CLI overrides take precedence over conf
    def _p(override, key, default):
        return override if override is not None else rvc_params.get(key, default)

    p_f0_up_key     = _p(f0_up_key,     "f0_up_key",     0)
    p_f0_method     = _p(f0_method,     "f0_method",     "rmvpe")
    p_index_ratio   = _p(index_ratio,   "index_ratio",   0.75)
    p_filter_radius = _p(filter_radius, "filter_radius", 3)
    p_rms_mix_rate  = _p(rms_mix_rate,  "rms_mix_rate",  0.75)
    p_protect       = _p(protect,       "protect",       0.33)
    p_resample_sr   = _p(resample_sr,   "resample_sr",   0)

    if not os.path.isfile(input_path):
        print(f"ERROR: input vocal not found: {input_path}", file=sys.stderr)
        print("Check 'input_vocal_path' in the conf.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(model_path):
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(index_path):
        print(f"ERROR: index not found: {index_path}", file=sys.stderr)
        sys.exit(1)

    # Default output path
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        model_stem = os.path.splitext(os.path.basename(model_path))[0]
        output_path = f"{base}_{model_stem}_rvc.wav"

    # ------------------------------------------------------------------
    # 2. Print run summary
    # ------------------------------------------------------------------
    src_analysis = conf.get("source_analysis", {})
    print("=" * 60)
    print("RVC Inference")
    print("=" * 60)
    print(f"  Input      : {input_path}")
    print(f"  Model      : {model_path}")
    print(f"  Index      : {index_path}")
    print(f"  Output     : {output_path}")
    print(f"  Device     : {device}")
    print()
    if src_analysis:
        print(f"  Source voice : {src_analysis.get('voice_type','?')}  "
              f"({src_analysis.get('median_note','?')} median, "
              f"{src_analysis.get('min_note','?')}-{src_analysis.get('max_note','?')})")
    print(f"  f0_up_key    : {p_f0_up_key:+d} semitones")
    print(f"  f0_method    : {p_f0_method}")
    print(f"  index_ratio  : {p_index_ratio}")
    print(f"  filter_radius: {p_filter_radius}")
    print(f"  rms_mix_rate : {p_rms_mix_rate}")
    print(f"  protect      : {p_protect}")
    print(f"  resample_sr  : {p_resample_sr if p_resample_sr else 'source SR'}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 3. Load RVC backend and run
    # ------------------------------------------------------------------
    BaseLoader = load_rvc_backend()

    # Map our generic param names to infer_rvc_python's apply_conf names
    pitch_algo  = _ALGO_MAP.get(p_f0_method, p_f0_method)
    only_cpu    = not (device.startswith("cuda") or device == "mps")

    print("\nInitializing RVC model ...")
    converter = BaseLoader(only_cpu=only_cpu)
    converter.apply_conf(
        tag="voice",
        file_model=model_path,
        pitch_algo=pitch_algo,
        pitch_lvl=p_f0_up_key,
        file_index=index_path,
        index_influence=p_index_ratio,
        respiration_median_filtering=p_filter_radius,
        envelope_ratio=p_rms_mix_rate,
        consonant_breath_protection=p_protect,
        resample_sr=p_resample_sr,
    )

    print("Running conversion ...")
    result = converter.generate_from_cache(audio_data=input_path, tag="voice")

    # generate_from_cache returns (result_array, sample_rate)
    if isinstance(result, (tuple, list)) and len(result) == 2:
        elem0, elem1 = result
        # Discriminate audio array vs sample rate scalar
        if hasattr(elem0, '__len__') and len(elem0) > 1000:
            out_audio, out_sr = elem0, elem1
        elif hasattr(elem1, '__len__') and len(elem1) > 1000:
            out_audio, out_sr = elem1, elem0
        else:
            out_audio, out_sr = elem0, elem1
    else:
        # fall back: read SR from input file
        out_audio = result
        _, out_sr = sf.read(input_path)

    def _to_numpy(x):
        """Convert torch tensor or any array-like to a plain numpy float32 array."""
        if hasattr(x, 'detach'):          # torch.Tensor
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype='float32')

    out_audio = _to_numpy(out_audio)
    out_sr_scalar = int(np.asarray(out_sr).flat[0])

    # Ensure valid audio shape
    if out_audio.ndim > 2:
        out_audio = out_audio.squeeze()

    sf.write(output_path, out_audio, out_sr_scalar)

    print(f"\nDone -> {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run RVC voice conversion using parameters from a .conf file "
            "generated by analyze_vocal.py."
        )
    )

    # Required
    parser.add_argument(
        "--conf", required=True, metavar="FILE",
        help="Path to the .conf file from analyze_vocal.py"
    )
    parser.add_argument(
        "--model", required=True, metavar="FILE",
        help="Path to the RVC .pth model file"
    )
    parser.add_argument(
        "--index", required=True, metavar="FILE",
        help="Path to the RVC .index file"
    )

    # Optional
    parser.add_argument(
        "-o", "--output", default=None, metavar="FILE",
        help="Output WAV path (default: <input>_<model>_rvc.wav)"
    )
    parser.add_argument(
        "--device", default="cuda" if __import__('torch').cuda.is_available() else "cpu",
        metavar="DEV",
        help="Torch device: cpu | cuda | cuda:0 (default: cuda if available, else cpu)"
    )

    # Per-run parameter overrides (override .conf values)
    parser.add_argument(
        "--f0-up-key", type=int, default=None, metavar="N",
        help="Override semitone transpose from conf"
    )
    parser.add_argument(
        "--f0-method", default=None,
        choices=["rmvpe", "rmvpe+", "crepe", "harvest", "pm", "dio"],
        help="Override F0 extraction method from conf"
    )
    parser.add_argument(
        "--index-ratio", type=float, default=None, metavar="R",
        help="Override feature index ratio (0.0-1.0) from conf"
    )
    parser.add_argument(
        "--filter-radius", type=int, default=None, metavar="N",
        help="Override F0 median filter radius from conf"
    )
    parser.add_argument(
        "--rms-mix-rate", type=float, default=None, metavar="R",
        help="Override RMS envelope mix rate from conf"
    )
    parser.add_argument(
        "--protect", type=float, default=None, metavar="P",
        help="Override consonant protection value from conf"
    )
    parser.add_argument(
        "--resample-sr", type=int, default=None, metavar="HZ",
        help="Override output resample SR from conf (0=keep source)"
    )

    args = parser.parse_args()

    run_rvc(
        conf_path=args.conf,
        model_path=args.model,
        index_path=args.index,
        output_path=args.output,
        f0_up_key=args.f0_up_key,
        f0_method=args.f0_method,
        index_ratio=args.index_ratio,
        filter_radius=args.filter_radius,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
        resample_sr=args.resample_sr,
        device=args.device,
    )


if __name__ == "__main__":
    main()

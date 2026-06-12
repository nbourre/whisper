#!/usr/bin/env python3
"""
sillage.py
Local cross-platform audio transcription.

Automatically detected backends:
  macOS Apple Silicon  -> mlx-whisper    (pip install -r requirements-mac.txt)
  Windows/Linux NVIDIA -> faster-whisper (pip install -r requirements-pc.txt)

CLI mode (with arguments):
  python sillage.py audio.mp3
  python sillage.py audio.mp3 --mode fast
  python sillage.py audio.mp3 --mode accuracy --output transcript.txt

Interactive mode (without arguments):
  python sillage.py
"""

import argparse
import json
import os
import platform
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# Platform detection
# --------------------------------------------------------------------------- #

def detect_backend() -> str:
    """
    Returns 'mlx' on Apple Silicon, 'faster' everywhere else.
    Detection uses platform.machine(), which returns 'arm64' on M1/M2/M3/M4.
    """
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    return "faster"


BACKEND = detect_backend()

MODELS = {
    "mlx": {
        "fast":     "mlx-community/whisper-small-mlx",
        "accuracy": "mlx-community/whisper-large-v3-mlx",
    },
    "faster": {
        "fast":     "small",
        "accuracy": "large-v3",
    },
}

BACKEND_LABEL = {
    "mlx":    "mlx-whisper  (Apple Silicon)",
    "faster": "faster-whisper (CUDA / CPU)",
}


# --------------------------------------------------------------------------- #
# Preferences persistence
# --------------------------------------------------------------------------- #

PREFS_FILE = Path.home() / ".sillage_prefs.json"

DEFAULT_PREFS = {
    "mode": "accuracy",
    "output_dir": "",   # empty = same folder as the audio file
    "last_audio": "",
}

def load_prefs() -> dict:
    if PREFS_FILE.exists():
        try:
            with open(PREFS_FILE, "r") as f:
                prefs = json.load(f)
            return {**DEFAULT_PREFS, **prefs}
        except Exception:
            pass
    return DEFAULT_PREFS.copy()

def save_prefs(prefs: dict):
    try:
        with open(PREFS_FILE, "w") as f:
            json.dump(prefs, f, indent=2)
    except Exception as e:
        print(f"  Warning: unable to save preferences ({e})")


# --------------------------------------------------------------------------- #
# Formatage des timestamps
# --------------------------------------------------------------------------- #

def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"[{h:02d}:{m:02d}:{s:02d}]"
    return f"[{m:02d}:{s:02d}]"


def format_transcript(segments) -> str:
    lines = []
    for seg in segments:
        # Compatibility for mlx (dict) and faster-whisper (object with attributes)
        start = seg["start"] if isinstance(seg, dict) else seg.start
        text  = seg["text"]  if isinstance(seg, dict) else seg.text
        text  = text.strip()
        if text:
            lines.append(f"{format_timestamp(start)} {text}")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# Backends de transcription
# --------------------------------------------------------------------------- #

def transcribe_mlx(audio_path: str, mode: str) -> dict:
    try:
        import mlx_whisper
    except ImportError:
        print("\n  mlx-whisper is not installed.")
        print("  Run: pip install -r requirements-mac.txt\n")
        sys.exit(1)

    model_repo = MODELS["mlx"][mode]
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_repo,
        word_timestamps=(mode == "accuracy"),
        verbose=False,
    )
    # Normalize to a consistent dict
    return {
        "language": result.get("language", "inconnu"),
        "segments": result.get("segments", []),
    }


def transcribe_faster(audio_path: str, mode: str) -> dict:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("\n  faster-whisper is not installed.")
        print("  Run: pip install -r requirements-pc.txt\n")
        sys.exit(1)

    model_name = MODELS["faster"][mode]
    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    segments_gen, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=(mode == "accuracy"),
    )
    # Materialize the generator into a list so it can be iterated multiple times
    segments = list(segments_gen)
    return {
        "language": info.language,
        "segments": segments,
    }


def transcribe(audio_path: str, mode: str = "accuracy") -> dict:
    model_id = MODELS[BACKEND][mode]
    print(f"\n  Backend : {BACKEND_LABEL[BACKEND]}")
    print(f"  Model   : {model_id}")
    print(f"  File    : {audio_path}")
    print(f"  Mode    : {mode}")
    print()

    if BACKEND == "mlx":
        return transcribe_mlx(audio_path, mode)
    else:
        return transcribe_faster(audio_path, mode)


# --------------------------------------------------------------------------- #
# Affichage et sauvegarde
# --------------------------------------------------------------------------- #

def print_result(result: dict, audio_path: str, mode: str):
    lang     = result.get("language", "inconnu")
    segments = result.get("segments", [])
    transcript = format_transcript(segments)

    width = 70
    print("\n" + "=" * width)
    print(" TRANSCRIPTION".center(width))
    print("=" * width)
    print(f"  File     : {Path(audio_path).name}")
    print(f"  Language : {lang.upper()}")
    print(f"  Mode     : {mode.capitalize()}")
    print(f"  Backend  : {BACKEND_LABEL[BACKEND]}")
    print(f"  Segments : {len(segments)}")
    print("-" * width)
    print()
    print(transcript)
    print()
    print("=" * width)


def save_result(result: dict, output_path: str):
    segments   = result.get("segments", [])
    transcript = format_transcript(segments)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcript)
        f.write("\n")
    print(f"\n  Transcript saved : {output_path}")


def resolve_output_path(audio_path: str, output_dir: str) -> str:
    audio = Path(audio_path)
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / audio.with_suffix(".txt").name)
    return str(audio.with_suffix(".txt"))


# --------------------------------------------------------------------------- #
# Path completion (readline)
# --------------------------------------------------------------------------- #

def setup_path_completion():
    try:
        import readline
        import glob

        def path_completer(text, state):
            results = glob.glob(os.path.expanduser(text) + "*")
            results = [r + "/" if os.path.isdir(r) else r for r in results]
            return results[state] if state < len(results) else None

        readline.set_completer(path_completer)
        readline.set_completer_delims(" \t\n;")
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass


def remove_path_completion():
    try:
        import readline
        readline.set_completer(None)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Interactive menu
# --------------------------------------------------------------------------- #

BANNER = r"""
   _____ ____             
  / __(_) / /__ ____ ____ 
 _\ \/ / / / _ `/ _ `/ -_)
/___/_/_/_/\_,_/\_, /\__/ 
    ~^~^~^~^~  /___/~^~^~^
"""

SUPPORTED_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".mov"}


def clear():
    os.system("cls" if platform.system() == "Windows" else "clear")

def prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    return input(f"{msg}{suffix} : ").strip()

def ask_quit(value: str) -> bool:
    return value.lower() == "q"


def menu_main(prefs: dict) -> dict | None:
    while True:
        clear()
        print(BANNER)
        print(f"  Backend : {BACKEND_LABEL[BACKEND]}")
        print()

        mode_label = "Fast (whisper-small)" if prefs["mode"] == "fast" else "Accuracy (whisper-large-v3)"
        out_label  = prefs["output_dir"] if prefs["output_dir"] else "same folder as the audio file"
        last_audio = prefs["last_audio"] if prefs["last_audio"] else "none"

        print("  Current settings")
        print(f"    [1] Mode           : {mode_label}")
        print(f"    [2] Output folder   : {out_label}")
        print()
        print("  Actions")
        print(f"    [3] Transcribe a file (last: {last_audio})")
        print()
        print("    [q] Quit")
        print()

        choice = input("  Choice : ").strip().lower()

        if ask_quit(choice):
            print("\n  See you soon!\n")
            return None
        elif choice == "1":
            prefs = menu_mode(prefs)
        elif choice == "2":
            prefs = menu_output_dir(prefs)
        elif choice == "3":
            params = menu_transcribe(prefs)
            if params:
                return params
        else:
            input("  Invalid choice. Press Enter to continue...")


def menu_mode(prefs: dict) -> dict:
    clear()
    print(BANNER)
    print("  Transcription mode selection")
    print()
    print("    [1] Fast     (whisper-small    — fast, less accurate)")
    print("    [2] Accuracy (whisper-large-v3 — slow, best quality)")
    print()
    print("    [q] Back")
    print()

    current = "1" if prefs["mode"] == "fast" else "2"
    choice = input(f"  Mode actuel [{current}] : ").strip().lower()

    if ask_quit(choice) or choice == "":
        return prefs
    elif choice == "1":
        prefs["mode"] = "fast"
        print("\n  Mode set to: Fast")
    elif choice == "2":
        prefs["mode"] = "accuracy"
        print("\n  Mode set to: Accuracy")
    else:
        print("\n  Invalid choice, mode unchanged.")

    save_prefs(prefs)
    input("\n  Press Enter to continue...")
    return prefs


def menu_output_dir(prefs: dict) -> dict:
    clear()
    print(BANNER)
    print("  Transcript output folder")
    print()
    print("  Leave blank to save in the same folder as the audio file.")
    print("  Type 'q' to go back without making changes.")
    print()

    setup_path_completion()
    val = prompt("  Folder", prefs["output_dir"] or "")
    remove_path_completion()

    if ask_quit(val):
        return prefs

    val = str(Path(val).expanduser()) if val else ""
    prefs["output_dir"] = val
    print(f"\n  Output folder : {val or 'same folder as the audio file'}")

    save_prefs(prefs)
    input("\n  Press Enter to continue...")
    return prefs


def menu_transcribe(prefs: dict) -> dict | None:
    clear()
    print(BANNER)
    print("  Audio file transcription")
    print()
    print("  Supported formats: MP3, WAV, M4A, FLAC, OGG, WebM, MP4, MOV")
    print("  Type 'q' to go back.")
    print()

    setup_path_completion()
    default_audio = prefs["last_audio"] or ""
    audio_input = prompt("  Audio file", default_audio)
    remove_path_completion()

    if ask_quit(audio_input):
        return None

    if audio_input == "" and default_audio:
        audio_input = default_audio

    if not audio_input:
        input("\n  No file specified. Press Enter to continue...")
        return None

    audio_path = str(Path(audio_input).expanduser())

    if not Path(audio_path).exists():
        print(f"\n  File not found: {audio_path}")
        input("\n  Press Enter to continue...")
        return None

    if Path(audio_path).suffix.lower() not in SUPPORTED_EXT:
        print(f"\n  Warning: unrecognized extension ({Path(audio_path).suffix}), trying anyway.")

    mode_label = "Fast (whisper-small)" if prefs["mode"] == "fast" else "Accuracy (whisper-large-v3)"
    out_path   = resolve_output_path(audio_path, prefs["output_dir"])
    out_label  = prefs["output_dir"] if prefs["output_dir"] else "same folder as the audio file"

    print()
    print("  " + "-" * 50)
    print(f"  File     : {Path(audio_path).name}")
    print(f"  Mode     : {mode_label}")
    print(f"  Backend  : {BACKEND_LABEL[BACKEND]}")
    print(f"  Output   : {out_label}")
    print("  " + "-" * 50)
    print()

    confirm = input("  Start transcription? [Y/n] : ").strip().lower()
    if confirm in ("n", "non", "q"):
        return None

    prefs["last_audio"] = audio_path
    save_prefs(prefs)

    return {
        "audio_path": audio_path,
        "mode": prefs["mode"],
        "output_path": out_path,
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def run_transcription(audio_path: str, mode: str, output_path: str):
    result = transcribe(audio_path, mode=mode)
    print_result(result, audio_path, mode)
    save_result(result, output_path)


def main():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Sillage — local cross-platform audio transcription"
        )
        parser.add_argument("audio", help="Path to the audio file")
        parser.add_argument(
            "--mode",
            choices=["fast", "accuracy"],
            default=None,
            help="fast = whisper-small | accuracy = whisper-large-v3",
        )
        parser.add_argument("--output", default=None, help="Output .txt file")
        args = parser.parse_args()

        prefs = load_prefs()
        mode  = args.mode or prefs["mode"]

        audio_path = str(Path(args.audio).expanduser())
        if not Path(audio_path).exists():
            print(f"\nError: file not found -> {audio_path}\n")
            sys.exit(1)

        output_path = args.output or resolve_output_path(audio_path, prefs["output_dir"])
        run_transcription(audio_path, mode, output_path)

    else:
        prefs = load_prefs()
        params = menu_main(prefs)
        if params:
            run_transcription(
                params["audio_path"],
                params["mode"],
                params["output_path"],
            )
            input("\n  Press Enter to quit...")


if __name__ == "__main__":
    main()

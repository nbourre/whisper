#!/usr/bin/env python3
"""
sillage.py
Transcription audio locale avec mlx-whisper sur Apple Silicon.

Mode CLI (avec arguments) :
  python sillage.py audio.mp3
  python sillage.py audio.mp3 --mode fast
  python sillage.py audio.mp3 --mode accuracy --output transcript.txt

Mode interactif (sans argument) :
  python sillage.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Persistance des préférences
# --------------------------------------------------------------------------- #

PREFS_FILE = Path.home() / ".sillage_prefs.json"

DEFAULT_PREFS = {
    "mode": "accuracy",
    "output_dir": "",   # vide = même dossier que le fichier audio
    "last_audio": "",
}

def load_prefs() -> dict:
    if PREFS_FILE.exists():
        try:
            with open(PREFS_FILE, "r") as f:
                prefs = json.load(f)
            # Fusionner avec les défauts pour les clés manquantes
            return {**DEFAULT_PREFS, **prefs}
        except Exception:
            pass
    return DEFAULT_PREFS.copy()

def save_prefs(prefs: dict):
    try:
        with open(PREFS_FILE, "w") as f:
            json.dump(prefs, f, indent=2)
    except Exception as e:
        print(f"  Avertissement : impossible de sauvegarder les préférences ({e})")


# --------------------------------------------------------------------------- #
# Formatage des timestamps (style FoziScribe : [00:01:23])
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
        timestamp = format_timestamp(seg["start"])
        text = seg["text"].strip()
        if text:
            lines.append(f"{timestamp} {text}")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #

MODELS = {
    "fast":     "mlx-community/whisper-small-mlx",
    "accuracy": "mlx-community/whisper-large-v3-mlx",
}

SUPPORTED_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".mov"}

def transcribe(audio_path: str, mode: str = "accuracy") -> dict:
    try:
        import mlx_whisper
    except ImportError:
        print("\n  mlx-whisper n'est pas installé.")
        print("  Lance : pip install mlx-whisper\n")
        sys.exit(1)

    model_repo = MODELS.get(mode, MODELS["accuracy"])

    print(f"\n  Modèle  : {model_repo}")
    print(f"  Fichier : {audio_path}")
    print(f"  Mode    : {mode}")
    print()

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_repo,
        word_timestamps=(mode == "accuracy"),
        verbose=False,
    )
    return result


def print_result(result: dict, audio_path: str, mode: str):
    lang = result.get("language", "inconnu")
    segments = result.get("segments", [])
    transcript = format_transcript(segments)

    width = 70
    print("\n" + "=" * width)
    print(" TRANSCRIPTION".center(width))
    print("=" * width)
    print(f"  Fichier  : {Path(audio_path).name}")
    print(f"  Langue   : {lang.upper()}")
    print(f"  Mode     : {mode.capitalize()}")
    print(f"  Segments : {len(segments)}")
    print("-" * width)
    print()
    print(transcript)
    print()
    print("=" * width)


def save_result(result: dict, output_path: str):
    segments = result.get("segments", [])
    transcript = format_transcript(segments)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcript)
        f.write("\n")
    print(f"\n  Transcript sauvegardé : {output_path}")


def resolve_output_path(audio_path: str, output_dir: str) -> str:
    audio = Path(audio_path)
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / audio.with_suffix(".txt").name)
    return str(audio.with_suffix(".txt"))


# --------------------------------------------------------------------------- #
# Autocomplétion de chemin (readline)
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
        pass  # Pas critique si ça ne marche pas


def remove_path_completion():
    try:
        import readline
        readline.set_completer(None)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Menu interactif
# --------------------------------------------------------------------------- #

BANNER = r"""
   _____ ____             
  / __(_) / /__ ____ ____ 
 _\ \/ / / / _ `/ _ `/ -_)
/___/_/_/_/\_,_/\_, /\__/ 
    ~^~^~^~^~  /___/~^~^~^
  Apple Silicon  •  mlx-whisper
"""

def clear():
    os.system("clear")

def prompt(msg: str, default: str = "") -> str:
    """Affiche un prompt avec la valeur par défaut entre crochets."""
    suffix = f" [{default}]" if default else ""
    return input(f"{msg}{suffix} : ").strip()

def ask_quit(value: str) -> bool:
    return value.lower() == "q"


def menu_main(prefs: dict) -> dict | None:
    """
    Menu principal. Retourne un dict de paramètres à lancer,
    ou None si l'utilisateur veut quitter.
    """
    while True:
        clear()
        print(BANNER)

        # Résumé des réglages actuels
        mode_label = "Rapide (whisper-small)" if prefs["mode"] == "fast" else "Précis (whisper-large-v3)"
        out_label  = prefs["output_dir"] if prefs["output_dir"] else "même dossier que le fichier audio"
        last_audio = prefs["last_audio"] if prefs["last_audio"] else "aucun"

        print("  Réglages actuels")
        print(f"    [1] Mode           : {mode_label}")
        print(f"    [2] Dossier sortie : {out_label}")
        print()
        print("  Actions")
        print(f"    [3] Transcire un fichier (dernier : {last_audio})")
        print()
        print("    [q] Quitter")
        print()

        choice = input("  Choix : ").strip().lower()

        if ask_quit(choice):
            print("\n  À bientôt!\n")
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
            input("  Choix invalide. Appuie sur Entrée pour continuer...")


def menu_mode(prefs: dict) -> dict:
    clear()
    print(BANNER)
    print("  Choix du mode de transcription")
    print()
    print("    [1] Fast     (whisper-small    — rapide, moins précis)")
    print("    [2] Accuracy (whisper-large-v3 — lent, meilleur résultat)")
    print()
    print("    [q] Retour")
    print()

    current = "1" if prefs["mode"] == "fast" else "2"
    choice = input(f"  Mode actuel [{current}] : ").strip().lower()

    if ask_quit(choice) or choice == "":
        return prefs
    elif choice == "1":
        prefs["mode"] = "fast"
        print("\n  Mode réglé sur : Fast")
    elif choice == "2":
        prefs["mode"] = "accuracy"
        print("\n  Mode réglé sur : Accuracy")
    else:
        print("\n  Choix invalide, mode inchangé.")

    save_prefs(prefs)
    input("\n  Entrée pour continuer...")
    return prefs


def menu_output_dir(prefs: dict) -> dict:
    clear()
    print(BANNER)
    print("  Dossier de sortie des transcripts")
    print()
    print("  Laisse vide pour sauvegarder dans le même dossier que le fichier audio.")
    print("  Tape 'q' pour revenir sans changer.")
    print()

    setup_path_completion()
    current = prefs["output_dir"] or ""
    val = prompt("  Dossier", current)
    remove_path_completion()

    if ask_quit(val):
        return prefs

    val = str(Path(val).expanduser()) if val else ""
    prefs["output_dir"] = val

    label = val if val else "même dossier que le fichier audio"
    print(f"\n  Dossier de sortie : {label}")

    save_prefs(prefs)
    input("\n  Entrée pour continuer...")
    return prefs


def menu_transcribe(prefs: dict) -> dict | None:
    """Demande le fichier audio et confirme les paramètres. Retourne les params ou None."""
    clear()
    print(BANNER)
    print("  Transcription d'un fichier audio")
    print()
    print("  Formats supportés : MP3, WAV, M4A, FLAC, OGG, WebM, MP4, MOV")
    print("  Tape 'q' pour revenir.")
    print()

    setup_path_completion()
    default_audio = prefs["last_audio"] or ""
    audio_input = prompt("  Fichier audio", default_audio)
    remove_path_completion()

    if ask_quit(audio_input):
        return None

    if audio_input == "" and default_audio:
        audio_input = default_audio

    if not audio_input:
        input("\n  Aucun fichier spécifié. Entrée pour continuer...")
        return None

    audio_path = str(Path(audio_input).expanduser())

    if not Path(audio_path).exists():
        print(f"\n  Fichier introuvable : {audio_path}")
        input("\n  Entrée pour continuer...")
        return None

    if Path(audio_path).suffix.lower() not in SUPPORTED_EXT:
        print(f"\n  Attention : extension non reconnue ({Path(audio_path).suffix}), on essaie quand même.")

    # Résumé avant lancement
    mode_label = "Rapide (whisper-small)" if prefs["mode"] == "fast" else "Précis (whisper-large-v3)"
    out_path   = resolve_output_path(audio_path, prefs["output_dir"])
    out_label  = prefs["output_dir"] if prefs["output_dir"] else "même dossier que le fichier audio"

    print()
    print("  " + "-" * 50)
    print(f"  Fichier  : {Path(audio_path).name}")
    print(f"  Mode     : {mode_label}")
    print(f"  Sortie   : {out_label}")
    print("  " + "-" * 50)
    print()

    confirm = input("  Lancer la transcription ? [O/n] : ").strip().lower()
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
# Entrée principale
# --------------------------------------------------------------------------- #

def run_transcription(audio_path: str, mode: str, output_path: str):
    result = transcribe(audio_path, mode=mode)
    print_result(result, audio_path, mode)
    save_result(result, output_path)


def main():
    if len(sys.argv) > 1:
        # Mode CLI classique
        parser = argparse.ArgumentParser(
            description="Transcription audio locale style FoziScribe (Apple Silicon)"
        )
        parser.add_argument("audio", help="Chemin vers le fichier audio")
        parser.add_argument(
            "--mode",
            choices=["fast", "accuracy"],
            default=None,
            help="fast = whisper-small | accuracy = whisper-large-v3",
        )
        parser.add_argument("--output", default=None, help="Fichier de sortie .txt")
        args = parser.parse_args()

        prefs = load_prefs()
        mode  = args.mode or prefs["mode"]

        audio_path = str(Path(args.audio).expanduser())
        if not Path(audio_path).exists():
            print(f"\nErreur : fichier introuvable -> {audio_path}\n")
            sys.exit(1)

        output_path = args.output or resolve_output_path(audio_path, prefs["output_dir"])
        run_transcription(audio_path, mode, output_path)

    else:
        # Mode interactif
        prefs = load_prefs()
        params = menu_main(prefs)
        if params:
            run_transcription(
                params["audio_path"],
                params["mode"],
                params["output_path"],
            )
            input("\n  Entrée pour quitter...")


if __name__ == "__main__":
    main()
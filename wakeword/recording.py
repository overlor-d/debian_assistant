"""
Capture d'echantillons audio pour constituer un dataset wakeword.

Pourquoi: simplifier la creation d'exemples positifs/negatifs sans autre outil.
Comment: enregistrements WAV via sounddevice/soundfile avec une session interactive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import sounddevice as sd
import soundfile as sf


def next_index(folder: Path, prefix: str) -> int:
    """
    Retourne l'index suivant pour eviter d'ecraser les enregistrements.

    Pourquoi: nommer les fichiers de maniere deterministic sans collision.
    Comment: on scanne les fichiers existants et on prend max+1.
    """
    existing = list(folder.glob(f"{prefix}_*.wav"))
    if not existing:
        return 1
    nums = [
        int(f.stem.split("_")[-1])
        for f in existing
        if f.stem.split("_")[-1].isdigit()
    ]
    return max(nums) + 1


def record_sample(
    folder: Path,
    prefix: str,
    mic_index: int,
    samplerate: int,
    duration_s: float,
    channels: int = 1,
    dtype: str = "int16",
) -> Path:
    """
    Capture un echantillon micro et le sauve dans folder.

    Pourquoi: produire des WAV courts et homogènes pour l'entrainement.
    Comment: enregistrement bloquant sounddevice, puis ecriture WAV via soundfile.
    """
    folder.mkdir(parents=True, exist_ok=True)
    idx = next_index(folder, prefix)
    filename = folder / f"{prefix}_{idx:03d}.wav"

    # Lancement de l'enregistrement (bloquant jusqu'a la fin).
    audio = sd.rec(
        int(duration_s * samplerate),
        samplerate=samplerate,
        channels=channels,
        device=mic_index,
        dtype=dtype,
        blocking=True,
    )
    # Sauvegarde sur disque au format WAV.
    sf.write(filename, audio, samplerate)
    return filename


def interactive_recording_session(
    positive_dir: Path,
    negative_dir: Path,
    mic_index: int,
    samplerate: int = 48000,
    duration_s: float = 1.2,
    channels: int = 1,
    positive_prefix: str = "wake",
    negative_prefix: str = "neg",
    intro: Optional[str] = None,
) -> None:
    """
    Petit shell interactif pour capturer des exemples positifs/negatifs.

    Pourquoi: alterner rapidement entre classes sans relancer le script.
    Comment: boucle simple avec commandes p/n/q et enregistrement a la demande.
    """
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)

    if intro:
        print(intro)
    print("Entrée : enregistre dans le mode courant")
    print("p + Entrée : basculer POSITIF")
    print("n + Entrée : basculer NÉGATIF")
    print("q + Entrée : quitter\n")

    mode = "positive"

    while True:
        # Lecture d'une commande utilisateur, vide = enregistrer.
        cmd = input(f"[mode={mode}] > ").strip().lower()

        if cmd == "q":
            break
        elif cmd == "p":
            mode = "positive"
            print("→ Mode POSITIF\n")
        elif cmd == "n":
            mode = "negative"
            print("→ Mode NÉGATIF\n")
        else:
            if mode == "positive":
                path = record_sample(positive_dir, positive_prefix, mic_index, samplerate, duration_s, channels=channels)
                print(f"✓ Sauvegardé {path.name}\n")
            else:
                path = record_sample(negative_dir, negative_prefix, mic_index, samplerate, duration_s, channels=channels)
                print(f"✓ Sauvegardé {path.name}\n")

"""
CLI de detection wakeword temps reel avec regression logistique.

Pourquoi: tester un modele entraine directement au micro.
Comment: charger le modele, instancier l'embedder, puis lancer la boucle audio.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wakeword.embedder import WakewordEmbedder
from wakeword.inference import run_realtime_detection
from wakeword.training import LogisticWakewordModel


def build_parser() -> argparse.ArgumentParser:
    """
    Construit le parser d'arguments.

    Pourquoi: regler micro, seuils temporels et performance.
    Comment: options CLI pour sr, hop, cooldown, etc.
    """
    parser = argparse.ArgumentParser(description="Détection wakeword en temps réel (régression logistique).")
    parser.add_argument("--model", default="ovos_lr_model.npz", help="Chemin du modèle entraîné")
    parser.add_argument("--mic", type=int, default=9, help="Index du micro (voir detect_mic.py)")
    parser.add_argument("--sr", type=int, default=48000, help="Fréquence micro (48 kHz requis)")
    parser.add_argument("--hop_ms", type=int, default=100, help="Taille de bloc micro (ms)")
    parser.add_argument("--cooldown_s", type=float, default=1.5, help="Anti-flapping : délai avant nouveau trigger")
    parser.add_argument("--consecutive", type=int, default=2, help="Nombre de blocs consécutifs au-dessus du seuil")
    parser.add_argument("--threads", type=int, default=1, help="Threads ONNXRuntime")
    return parser


def sync_model_audio_len(model: LogisticWakewordModel, embedder: WakewordEmbedder) -> None:
    """
    Aligne la longueur audio du modele sur celle de l'embedder.

    Pourquoi: eviter des decalages si le modele a ete entraine autrement.
    Comment: copie directe de embedder.audio_len.
    """
    model.audio_len = int(embedder.audio_len)


def main() -> None:
    """
    Point d'entree CLI de detection.

    Pourquoi: fil conducteur simple pour executer la boucle temps reel.
    Comment: charge modele, aligne la longueur audio, puis lance detection.
    """
    args = build_parser().parse_args()
    model = LogisticWakewordModel.load(Path(args.model))
    embedder = WakewordEmbedder(threads=args.threads)

    # Sécurité : si le modèle a été entraîné avec une autre longueur audio, on s'aligne sur l'embedder.
    sync_model_audio_len(model, embedder)

    # Lance la boucle audio bloquante.
    run_realtime_detection(
        model=model,
        embedder=embedder,
        mic_index=args.mic,
        sample_rate=args.sr,
        hop_ms=args.hop_ms,
        cooldown_s=args.cooldown_s,
        consecutive=args.consecutive,
    )


if __name__ == "__main__":
    main()

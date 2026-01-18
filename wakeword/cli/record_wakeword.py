"""
CLI d'enregistrement d'exemples positifs/negatifs.

Pourquoi: fournir un outil simple pour construire un dataset local.
Comment: parse d'arguments puis appel a la session interactive.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wakeword.recording import interactive_recording_session


def build_parser() -> argparse.ArgumentParser:
    """
    Construit le parser d'arguments.

    Pourquoi: centraliser les options CLI (micro, duree, dossiers).
    Comment: argparse standard avec valeurs par defaut.
    """
    parser = argparse.ArgumentParser(description="Enregistrer des échantillons positifs/négatifs.")
    parser.add_argument("--mic", type=int, default=9, help="Index du micro (voir detect_mic.py)")
    parser.add_argument("--samplerate", type=int, default=48000, help="Fréquence d'échantillonnage du micro")
    parser.add_argument("--duration", type=float, default=1.2, help="Durée d'un échantillon (secondes)")
    parser.add_argument("--base-dir", type=str, default="wakeword_data/ovos", help="Dossier racine des échantillons")
    parser.add_argument("--wakeword-prefix", type=str, default="ovos", help="Préfixe de nommage pour les positifs")
    parser.add_argument("--negative-prefix", type=str, default="neg", help="Préfixe de nommage pour les négatifs")
    return parser


def resolve_recording_dirs(base_dir: Path) -> tuple[Path, Path]:
    """
    Deduit les sous-dossiers positifs/negatifs a partir d'un dossier racine.

    Pourquoi: centraliser la convention de structure de dataset.
    Comment: "positive" et "negative" sous base_dir.
    """
    return base_dir / "positive", base_dir / "negative"


def build_intro_message(wakeword_prefix: str) -> str:
    """
    Construit le message d'introduction affiche a l'utilisateur.

    Pourquoi: garder la logique d'affichage isolee du main.
    Comment: message multi-lignes avec consignes de base.
    """
    return (
        "Mode enregistrement wakeword\n"
        f"Positif: dire '{wakeword_prefix}' (ou le mot choisi)\n"
        "Négatif: dire autre chose qui s'en rapproche pour robustifier."
    )


def main() -> None:
    """
    Point d'entree CLI.

    Pourquoi: enchaîner parsing + session interactive en une commande.
    Comment: derive les dossiers et passe un texte d'intro.
    """
    args = build_parser().parse_args()
    base = Path(args.base_dir)
    pos_dir, neg_dir = resolve_recording_dirs(base)
    intro = build_intro_message(args.wakeword_prefix)

    interactive_recording_session(
        positive_dir=pos_dir,
        negative_dir=neg_dir,
        mic_index=args.mic,
        samplerate=args.samplerate,
        duration_s=args.duration,
        positive_prefix=args.wakeword_prefix,
        negative_prefix=args.negative_prefix,
        intro=intro,
    )


if __name__ == "__main__":
    main()

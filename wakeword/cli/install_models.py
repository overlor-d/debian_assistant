"""
Telecharge les modeles openWakeWord necessaires a l'embedder.

Pourquoi: l'embedder ONNX depend de fichiers de modele stockes localement.
Comment: appelle l'utilitaire openwakeword pour telecharger dans le cache.
"""

from __future__ import annotations

from openwakeword.utils import download_models


def install_openwakeword_models() -> None:
    """
    Declenche le telechargement des modeles openWakeWord.

    Pourquoi: isoler la logique d'installation du point d'entree CLI.
    Comment: delegation directe a openwakeword.utils.download_models.
    """
    download_models()


def main() -> None:
    """
    Telecharge les modeles openWakeWord necessaires a l'embedder.

    Pourquoi: sans ces fichiers, l'embedder ne peut pas charger les graphes.
    Comment: openwakeword.utils.download_models place les fichiers au bon endroit.
    """
    install_openwakeword_models()
    print("Téléchargement terminé.")


if __name__ == "__main__":
    main()

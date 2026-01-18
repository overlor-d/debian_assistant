"""
Liste les peripheriques micro disponibles.

Pourquoi: aider l'utilisateur a trouver l'index du micro a passer aux CLI.
Comment: interrogation sounddevice des devices d'entree.
"""

from __future__ import annotations

import sounddevice as sd


def list_input_devices() -> list[tuple[int, str, int]]:
    """
    Retourne la liste des devices micro disponibles.

    Pourquoi: separer la collecte (API sounddevice) de l'affichage CLI.
    Comment: on filtre les devices avec au moins un canal d'entree.
    """
    devices: list[tuple[int, str, int]] = []
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            name = dev.get("name", "unknown")
            channels = dev.get("max_input_channels", 0)
            devices.append((index, name, channels))
    return devices


def main() -> None:
    """
    Affiche les entrees audio disponibles avec leur index.

    Pourquoi: l'index est necessaire pour l'enregistrement et la detection.
    Comment: filtre des devices avec au moins un canal d'entree.
    """
    for index, name, channels in list_input_devices():
        print(f"{index}: {name} (inputs={channels})")


if __name__ == "__main__":
    main()

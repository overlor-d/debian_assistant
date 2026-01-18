"""
Outils pour enregistrer, entraîner et tester un wakeword léger basé sur openWakeWord.

Les sous-modules exposent les briques de base :
- audio : utilitaires d'E/S et de prétraitement audio.
- embedder : wrapper ONNX pour générer les embeddings openWakeWord.
- dataset : préparation du jeu d'entraînement (chargement + augmentations).
- training : apprentissage/logistic regression et sérialisation du modèle.
- inference : boucle de détection temps réel.
- recording : helpers pour capturer des jeux de données d'exemples.

Pourquoi ce package:
- proposer un pipeline complet et léger (collecte -> embeddings -> modele -> detection).
- garder des dependances minimales pour tourner sur des machines modestes.

Comment l'utiliser (vue rapide):
- enregistrer des exemples avec wakeword/cli/record_wakeword.py
- entrainer un modele via wakeword/cli/train_wakeword_lr.py
- tester en temps reel avec wakeword/cli/run_wakeword_lr.py
"""

__all__ = [
    "audio",
    "dataset",
    "embedder",
    "inference",
    "recording",
    "training",
]

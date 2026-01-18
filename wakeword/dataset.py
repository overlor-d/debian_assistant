"""
Construction d'un dataset d'embeddings a partir de WAV positifs/negatifs.

Pourquoi: separer la logique d'augmentation/embedding de l'entrainement.
Comment: charger, normaliser, crop, augmenter, puis embedder chaque fichier.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from wakeword import audio
from wakeword.embedder import WakewordEmbedder


@dataclass
class EmbeddingDataset:
    """
    Jeu d'entrainement construit a partir des embeddings openWakeWord.

    Pourquoi: garder ensemble les matrices et les metadonnees utiles.
    Comment: embeddings X, labels y, listes de fichiers et longueur audio.
    """

    embeddings: np.ndarray
    labels: np.ndarray
    positive_files: list[Path]
    negative_files: list[Path]
    audio_len: int


def _embed_file_with_augmentations(
    file_path: Path,
    label: float,
    embedder: WakewordEmbedder,
    rng: np.random.Generator,
    augmentations: int,
    target_sr: int,
) -> Iterable[tuple[np.ndarray, float]]:
    """
    Genere l'embedding de base + n augmentations pour un fichier.

    Pourquoi: augmenter la diversite sans collecte audio supplementaire.
    Comment: pipeline de pretraitement puis n augmentations rapides.
    """
    # Charge le WAV et force un mono float32.
    waveform, sr = audio.load_mono_audio(file_path)
    # Re-echantillonne vers la frequence cible.
    waveform = audio.resample_linear(waveform, sr, target_sr)
    # Normalise le niveau pour stabiliser l'embedder.
    waveform = audio.rms_normalize(waveform)
    # Centre la fenetre sur l'energie la plus forte.
    waveform = audio.center_crop_on_energy(waveform, target_sr, target_len=embedder.audio_len)

    # Embedding sans augmentation.
    yield embedder.embed_waveform(waveform), label
    for _ in range(augmentations):
        # Augmentations rapides: gain/bruit/shift.
        augmented = audio.augment_waveform(waveform, rng)
        yield embedder.embed_waveform(augmented), label


def build_embedding_dataset(
    positive_dir: Path,
    negative_dir: Path,
    embedder: WakewordEmbedder,
    rng: np.random.Generator,
    augmentations: int = 5,
    target_sr: int = audio.DEFAULT_TARGET_SR,
) -> EmbeddingDataset:
    """
    Construit les embeddings à partir des WAV positifs/négatifs.

    On applique une normalisation, un crop centré sur l'énergie, puis des
    augmentations rapides (gain + bruit + shift) pour enrichir l'entraînement.

    Pourquoi: produire X/y directement utilisables par la regression logistique.
    Comment: on parcourt les fichiers positifs puis negatifs et on empile.
    """
    # Liste des fichiers source.
    pos_files = audio.list_wav_files(positive_dir)
    neg_files = audio.list_wav_files(negative_dir)

    if not pos_files or not neg_files:
        raise ValueError("Les dossiers positifs/négatifs doivent contenir des fichiers WAV.")

    embeddings: list[np.ndarray] = []
    labels: list[float] = []

    # Embeddings positifs (label 1.0).
    for f in pos_files:
        for emb, lab in _embed_file_with_augmentations(f, 1.0, embedder, rng, augmentations, target_sr):
            embeddings.append(emb)
            labels.append(lab)

    # Embeddings negatifs (label 0.0).
    for f in neg_files:
        for emb, lab in _embed_file_with_augmentations(f, 0.0, embedder, rng, augmentations, target_sr):
            embeddings.append(emb)
            labels.append(lab)

    # Empilement final en matrices NumPy.
    X = np.vstack(embeddings).astype(np.float32)
    y = np.array(labels, dtype=np.float32)

    return EmbeddingDataset(
        embeddings=X,
        labels=y,
        positive_files=pos_files,
        negative_files=neg_files,
        audio_len=int(embedder.audio_len),
    )

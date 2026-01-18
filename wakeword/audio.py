"""
Utilitaires audio pour la preparation du wakeword.

Pourquoi: regrouper les operations basiques (I/O, normalisation, resampling).
Comment: fonctions NumPy simples, sans dependances lourdes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

# Fréquence cible commune à tous les traitements.
DEFAULT_TARGET_SR = 16000
# Paramètres d'analyse d'énergie (en secondes).
ENERGY_FRAME_S = 0.02
ENERGY_HOP_S = 0.01


def list_wav_files(directory: Path) -> list[Path]:
    """
    Liste triee des fichiers WAV presents dans un dossier.

    Pourquoi: ordre stable pour l'entrainement et la reproductibilite.
    Comment: glob + tri.
    """
    return sorted([p for p in directory.glob("*.wav") if p.is_file()])


def load_mono_audio(path: Path) -> tuple[np.ndarray, int]:
    """
    Charge un fichier audio en mono float32.

    Retourne le signal (1D) et la fréquence d'échantillonnage. On supprime
    l'offset DC pour éviter les biais à l'entraînement.

    Pourquoi: simplifier le pipeline en un signal mono propre.
    Comment: moyenne des canaux + retrait de la moyenne (DC offset).
    """
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    # Conversion multi-canal -> mono par moyenne.
    mono = audio.mean(axis=1).astype(np.float32)
    # Retrait de l'offset DC si le signal n'est pas vide.
    mono = mono - float(np.mean(mono)) if mono.size else mono
    return mono, int(sample_rate)


def resample_linear(samples: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """
    Rééchantillonne linéairement un signal 1D vers target_sr.

    C'est un fallback simple mais suffisant ici pour rester léger en dépendances.

    Pourquoi: normaliser la frequence d'entree pour l'embedder (16 kHz).
    Comment: interpolation lineaire sur une base temporelle normalisee.
    """
    if source_sr == target_sr or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    n_out = int(round(samples.size * (target_sr / source_sr)))
    if n_out <= 1:
        return np.zeros((0,), dtype=np.float32)
    t_in = np.linspace(0.0, 1.0, num=samples.size, endpoint=False, dtype=np.float32)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float32)
    return np.interp(t_out, t_in, samples).astype(np.float32)


def rms_normalize(samples: np.ndarray, target_rms: float = 0.03) -> np.ndarray:
    """
    Normalise le volume RMS de maniere douce et clippe via tanh.

    Pourquoi: stabiliser les embeddings malgre des volumes variables.
    Comment: calcul RMS -> gain borne -> tanh pour limiter les pics.
    """
    if samples.size == 0:
        return samples
    rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))
    gain = float(np.clip(target_rms / (rms + 1e-12), 0.1, 10.0))
    return np.tanh(samples * gain).astype(np.float32)


def center_crop_on_energy(samples: np.ndarray, sample_rate: int, target_len: int) -> np.ndarray:
    """
    Extrait target_len échantillons centrés sur le frame RMS le plus énergétique.

    Pourquoi: recentrer sur la zone la plus informative (souvent le mot).
    Comment: fenetrage court -> RMS par frame -> crop centre.
    """
    if target_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    if samples.size == 0:
        return np.zeros((target_len,), dtype=np.float32)

    # Fenetres courtes de 20 ms avec hop 10 ms.
    frame = max(int(round(ENERGY_FRAME_S * sample_rate)), 1)
    hop = max(int(round(ENERGY_HOP_S * sample_rate)), 1)

    # Pad minimal si le signal est trop court pour un frame.
    padded = samples if samples.size >= frame else np.pad(samples, (0, frame - samples.size))
    n_frames = 1 + (padded.size - frame) // hop
    rms = np.empty((n_frames,), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        seg = padded[start:start + frame]
        rms[i] = float(np.sqrt(np.mean(seg * seg) + 1e-12))
    idx_max = int(np.argmax(rms))
    center = idx_max * hop + frame // 2

    start = center - target_len // 2
    end = start + target_len

    if start < 0:
        # Pad a gauche si le crop deborde.
        samples = np.pad(samples, (-start, 0))
        start = 0
        end = target_len
    if end > samples.size:
        # Pad a droite si le crop deborde.
        samples = np.pad(samples, (0, end - samples.size))
    return samples[start:end].astype(np.float32)


def augment_waveform(
    samples: np.ndarray,
    rng: np.random.Generator,
    gain_range: Tuple[float, float] = (0.7, 1.3),
    noise_std: float = 0.003,
    max_shift: int = 200,
) -> np.ndarray:
    """
    Augmentation légère et rapide : gain aléatoire, bruit blanc, petit décalage.

    Pourquoi: rendre le modele plus robuste a la variabilite reelle.
    Comment: gain random, ajout de bruit, rotation temporelle simple.
    """
    # Copie pour ne pas modifier l'original.
    augmented = samples.astype(np.float32, copy=True)
    gain = float(rng.uniform(gain_range[0], gain_range[1]))
    augmented *= gain

    noise = rng.normal(0.0, noise_std, size=augmented.shape).astype(np.float32)
    augmented += noise

    shift = int(rng.integers(-max_shift, max_shift))
    augmented = np.roll(augmented, shift)

    return np.tanh(augmented).astype(np.float32)


def downsample_48k_to_16k(samples: np.ndarray) -> np.ndarray:
    """
    Sous-échantillonne par facteur 3 (48 kHz -> 16 kHz). Ne pas utiliser si la
    fréquence n'est pas exactement 48 kHz.

    Pourquoi: rapide et suffisant pour un flux micro 48 kHz fixe.
    Comment: on prend un echantillon sur trois (decimation brute).
    """
    return samples[::3].astype(np.float32, copy=False)

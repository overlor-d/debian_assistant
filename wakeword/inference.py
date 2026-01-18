"""
Inferer un wakeword en temps reel.

Pourquoi: isoler la logique de streaming (buffer, seuil, cooldown) pour
pouvoir la reutiliser depuis une CLI ou un service.
Comment: un ring buffer conserve une fenetre audio 16 kHz, l'embedder
produit un embedding, puis la regression logistique donne une probabilite.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

from wakeword import audio
from wakeword.embedder import WakewordEmbedder
from wakeword.training import LogisticWakewordModel

DetectorResult = tuple[float, bool]


class SlidingWindow:
    """
    Buffer circulaire 1D de taille fixe.

    Pourquoi: garder en permanence la derniere fenetre audio complete sans
    allouer a chaque bloc micro.
    Comment: on decale le buffer et on injecte les derniers echantillons.
    """

    def __init__(self, length: int):
        # Buffer mono 16 kHz, initialise a zero.
        self.buffer = np.zeros((length,), dtype=np.float32)
        self.filled = False

    def push(self, samples: np.ndarray) -> None:
        # Ajoute une tranche de samples dans la fenetre glissante.
        n = samples.size
        if n <= 0:
            return
        if n >= self.buffer.size:
            # Si le bloc est plus grand que la fenetre, on garde la fin.
            self.buffer[:] = samples[-self.buffer.size :]
            self.filled = True
            return
        # Decale puis remplit la fin avec les nouveaux samples.
        self.buffer[:-n] = self.buffer[n:]
        self.buffer[-n:] = samples
        self.filled = True

    def get(self) -> np.ndarray:
        # Copie defensive pour eviter les effets de bord.
        return self.buffer.copy()


class WakewordScorer:
    """
    Calcule la probabilite wakeword pour une fenetre audio 16 kHz.

    Pourquoi: separer la partie "feature -> proba" de la logique streaming.
    Comment: normalisation RMS, embedding ONNX, puis score logistique.
    """

    def __init__(self, model: LogisticWakewordModel, embedder: WakewordEmbedder):
        self.model = model
        self.embedder = embedder

    def probability(self, waveform_16k: np.ndarray) -> float:
        # Normalisation de niveau pour stabiliser l'embedder.
        normalized = audio.rms_normalize(waveform_16k)
        # Extraction d'embedding openWakeWord.
        embedding = self.embedder.embed_waveform(normalized)
        # Score de la regression logistique.
        return self.model.score(embedding)


class StreamingWakewordDetector:
    """
    Gere le flux micro + logique de declenchement (streak + cooldown).

    Pourquoi: eviter les faux positifs isoles (streak) et les triggers en
    rafale (cooldown).
    Comment: on accumule des blocs, on score la fenetre, puis on applique
    la logique de seuil/consecutifs.
    """

    def __init__(
        self,
        scorer: WakewordScorer,
        cooldown_s: float = 1.5,
        consecutive: int = 2,
    ):
        self.scorer = scorer
        self.cooldown_s = float(cooldown_s)
        self.consecutive = int(consecutive)
        self.window = SlidingWindow(length=scorer.model.audio_len)
        self.lock = threading.Lock()
        self.last_fire = 0.0
        self.streak = 0

    def push_block_48k(self, block_48k: np.ndarray) -> None:
        """
        Reduit un bloc 48 kHz vers 16 kHz puis pousse dans le buffer.

        Pourquoi: l'embedder attend du 16 kHz.
        Comment: sous-echantillonnage x3 (48k -> 16k).
        """
        block_16k = audio.downsample_48k_to_16k(block_48k.astype(np.float32))
        with self.lock:
            self.window.push(block_16k)

    def evaluate(self) -> Optional[DetectorResult]:
        """
        Calcule la proba sur la fenetre courante.

        Pourquoi: separer l'alimentation du buffer de la decision de trigger.
        Comment: score -> mise a jour du streak -> respect du cooldown.
        Retourne (probabilite, declenchement) ou None si buffer vide.
        """
        with self.lock:
            if not self.window.filled:
                return None
            waveform = self.window.get()

        # Score sur la fenetre courante.
        prob = self.scorer.probability(waveform)
        # Streak: combien de fenetres consecutives au-dessus du seuil.
        if prob >= self.scorer.model.threshold:
            self.streak += 1
        else:
            self.streak = 0

        triggered = False
        now = time.time()
        # Cooldown: evite les triggers trop rapproches.
        if self.streak >= self.consecutive and (now - self.last_fire) >= self.cooldown_s:
            triggered = True
            self.last_fire = now
            self.streak = 0

        return prob, triggered


def run_realtime_detection(
    model: LogisticWakewordModel,
    embedder: WakewordEmbedder,
    mic_index: int,
    sample_rate: int = 48000,
    hop_ms: int = 100,
    cooldown_s: float = 1.5,
    consecutive: int = 2,
) -> None:
    """
    Boucle CLI bloquante : lit le micro, calcule la proba et imprime les triggers.

    Pourquoi: fournir une demo en temps reel directement depuis le terminal.
    Comment: InputStream sounddevice + callback qui alimente le detecteur.
    """
    if sample_rate != 48000:
        raise SystemExit("Le flux micro est supposé en 48 kHz (downsample x[::3]).")

    hop_48 = int(round(sample_rate * (hop_ms / 1000.0)))
    if hop_48 <= 0:
        raise SystemExit("hop_ms invalide.")

    # Assemble les briques (embedder -> scorer -> detecteur).
    scorer = WakewordScorer(model=model, embedder=embedder)
    detector = StreamingWakewordDetector(scorer=scorer, cooldown_s=cooldown_s, consecutive=consecutive)

    def callback(indata, frames, time_info, status):
        # Callback audio: 1er canal uniquement (mono).
        block = indata[:, 0].astype(np.float32)
        detector.push_block_48k(block)

    # Boucle principale: on attend des blocs et on evalue periodiquement.
    with sd.InputStream(device=mic_index, channels=1, samplerate=sample_rate, blocksize=hop_48, callback=callback):
        while True:
            time.sleep(hop_ms / 1000.0)
            result = detector.evaluate()
            if result is None:
                continue
            prob, fired = result
            if fired:
                print(f"TRUE (p={prob:.3f})", flush=True)

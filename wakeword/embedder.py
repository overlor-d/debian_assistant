"""
Extraction d'embeddings openWakeWord avec ONNXRuntime.

Pourquoi: eviter la dependance tflite_runtime et uniformiser les formats
d'entree/sortie des graphes openWakeWord.
Comment: passe mel -> formatage -> passe embedding -> normalisation L2.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import onnxruntime as ort
import openwakeword

# Noms de fichiers des graphes ONNX fournis par openwakeword.
MEL_MODEL_NAME = "melspectrogram.onnx"
EMB_MODEL_NAME = "embedding_model.onnx"
# Valeurs par défaut si les shapes ne sont pas renseignées par ONNX.
DEFAULT_AUDIO_LEN = 16000
DEFAULT_T_REQ = 76
DEFAULT_F_REQ = 32


def resolve_oww_model_path(model_name: str) -> Path:
    """
    Renvoie le chemin d'un modele openWakeWord embarque dans le package Python.

    Pourquoi: les modeles sont deja fournis par le package openwakeword.
    Comment: on derive le chemin a partir du module installe.
    """
    base = Path(openwakeword.__file__).resolve().parent / "resources" / "models"
    path = base / model_name
    if not path.exists():
        raise FileNotFoundError(f"Modèle openWakeWord introuvable: {path}")
    return path


def _get_shape_dim(shape: object, index: int, default: int) -> int:
    """
    Extrait une dimension d'une shape ONNX si elle est connue.

    Pourquoi: certains exports contiennent des dimensions dynamiques (None).
    Comment: on ne retient la valeur que si elle est un int.
    """
    if isinstance(shape, (list, tuple)) and len(shape) > index:
        if isinstance(shape[index], int):
            return int(shape[index])
    return int(default)


def _crop_or_pad_axis(arr: np.ndarray, axis: int, target: int) -> np.ndarray:
    """
    Centre crop/pad sur un axe donne.

    Pourquoi: les graphes ONNX attendent des dimensions fixes.
    Comment: on coupe au centre ou on pad avec des zeros.
    """
    current = int(arr.shape[axis])
    if current == target:
        return arr
    if current > target:
        # Crop centre pour ne pas decaler l'information temporelle.
        start = (current - target) // 2
        slicer = [slice(None)] * arr.ndim
        slicer[axis] = slice(start, start + target)
        return arr[tuple(slicer)]

    pad = target - current
    before = pad // 2
    after = pad - before
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (before, after)
    return np.pad(arr, pad_width, mode="constant")


def _format_mel_to_nhwc(mel: np.ndarray, expected_freq_bins: int) -> np.ndarray:
    """
    Convertit une sortie mel variée vers un format NHWC [1, T, F, 1] float32.

    Gère les formats rencontrés (NCHW, NHWC, 2D ou 3D) en détectant l'axe fréquence.
    """
    mel = np.asarray(mel, dtype=np.float32)

    if mel.ndim == 4:
        # NHWC standard.
        if mel.shape[0] == 1 and mel.shape[-1] == 1 and mel.shape[-2] == expected_freq_bins:
            return mel

        # NCHW [1, 1, T, F].
        if mel.shape[0] == 1 and mel.shape[1] == 1 and mel.shape[-1] == expected_freq_bins:
            return np.transpose(mel, (0, 2, 3, 1)).astype(np.float32)

        # Dernier recours : on détecte l'axe fréquence et on aligne en NHWC.
        shape = mel.shape
        try:
            freq_axis = shape.index(expected_freq_bins)
        except ValueError:
            raise ValueError(f"Mel ne contient pas F={expected_freq_bins}. shape={shape}")

        channel_axis = None
        for ax in range(1, 4):
            if ax != freq_axis and shape[ax] == 1:
                channel_axis = ax
                break

        time_candidates = [ax for ax in range(1, 4) if ax not in (freq_axis, channel_axis)]
        if not time_candidates:
            raise ValueError(f"Impossible d'inférer l'axe temps pour mel shape={shape}")
        time_axis = time_candidates[0]

        if channel_axis is None:
            # On ajoute un channel artificiel via moyenne sur l'axe restant.
            remaining = [ax for ax in range(1, 4) if ax not in (time_axis, freq_axis)]
            mel = np.transpose(mel, (0, time_axis, freq_axis, remaining[0]))
            mel = mel.mean(axis=-1, keepdims=True)
            return mel.astype(np.float32)

        return np.transpose(mel, (0, time_axis, freq_axis, channel_axis)).astype(np.float32)

    if mel.ndim == 3:
        # [1, T, F]
        if mel.shape[0] == 1 and mel.shape[-1] == expected_freq_bins:
            return mel[:, :, :, np.newaxis].astype(np.float32)
        # [1, F, T]
        if mel.shape[0] == 1 and mel.shape[1] == expected_freq_bins:
            return np.transpose(mel, (0, 2, 1))[:, :, :, np.newaxis].astype(np.float32)
        raise ValueError(f"Shape mel 3D non gérée: {mel.shape}")

    if mel.ndim == 2:
        if mel.shape[1] == expected_freq_bins:
            return mel[np.newaxis, :, :, np.newaxis].astype(np.float32)
        if mel.shape[0] == expected_freq_bins:
            return mel.T[np.newaxis, :, :, np.newaxis].astype(np.float32)
        raise ValueError(f"Shape mel 2D non gérée: {mel.shape}")

    raise ValueError(f"Dimension mel inattendue: ndim={mel.ndim}, shape={mel.shape}")


def _prepare_waveform(audio_16k: np.ndarray, target_len: int) -> np.ndarray:
    """
    Centre et tronque/pad un signal 16 kHz vers la longueur attendue par le modele.

    Pourquoi: le graphe mel impose une longueur fixe.
    Comment: on pad en fin ou on coupe au centre si trop long.
    """
    x = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
    if x.size < target_len:
        x = np.pad(x, (0, target_len - x.size))
    elif x.size > target_len:
        start = (x.size - target_len) // 2
        x = x[start:start + target_len]
    return x.astype(np.float32)


class WakewordEmbedder:
    """
    Génère des embeddings openWakeWord via les graphes ONNX fournis par le projet.

    Cette classe contourne la dépendance tflite_runtime en utilisant ONNXRuntime.
    """

    def __init__(self, threads: int = 1):
        # Localisation des graphes ONNX embarques par openwakeword.
        self.mel_path = resolve_oww_model_path(MEL_MODEL_NAME)
        self.emb_path = resolve_oww_model_path(EMB_MODEL_NAME)

        # Configuration ONNXRuntime: controle du parallélisme CPU.
        session_opts = ort.SessionOptions()
        session_opts.intra_op_num_threads = int(threads)
        session_opts.inter_op_num_threads = int(threads)

        # Sessions ONNX separees: mel-spectrogramme puis embedding.
        self.mel = ort.InferenceSession(
            str(self.mel_path),
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )
        self.emb = ort.InferenceSession(
            str(self.emb_path),
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

        self.mel_in = self.mel.get_inputs()[0]
        self.mel_out = self.mel.get_outputs()[0]
        self.emb_in = self.emb.get_inputs()[0]
        self.emb_out = self.emb.get_outputs()[0]

        # Longueur audio attendue par le graphe mel (par defaut 16000).
        self.audio_len = _get_shape_dim(self.mel_in.shape, 1, DEFAULT_AUDIO_LEN)

        # Dimensions attendues par le graphe embedding (T, F).
        self.t_req = _get_shape_dim(self.emb_in.shape, 1, DEFAULT_T_REQ)
        self.f_req = _get_shape_dim(self.emb_in.shape, 2, DEFAULT_F_REQ)

    def _run_mel(self, waveform: np.ndarray) -> np.ndarray:
        """
        Execute le graphe mel et harmonise la sortie au format attendu.

        Pourquoi: differents exports ONNX peuvent varier (NCHW/NHWC).
        Comment: formatage + crop/pad temporel/frequentiel.
        """
        xin = waveform.reshape(1, -1).astype(np.float32)
        mel_raw = self.mel.run([self.mel_out.name], {self.mel_in.name: xin})[0]
        mel_nhwc = _format_mel_to_nhwc(mel_raw, self.f_req)
        mel_nhwc = _crop_or_pad_axis(mel_nhwc, axis=2, target=self.f_req)
        return _crop_or_pad_axis(mel_nhwc, axis=1, target=self.t_req).astype(np.float32)

    def embed_waveform(self, audio_16k: np.ndarray) -> np.ndarray:
        """
        Transforme un signal mono 16 kHz en embedding normalise L2.

        Pourquoi: un vecteur L2-normalise se compare mieux avec une LR.
        Comment: waveform -> mel -> embedding -> normalisation L2.
        """
        # Ajuste la longueur d'entree au graphe mel.
        prepared = _prepare_waveform(audio_16k, self.audio_len)
        # Mel-spectrogramme format NHWC.
        mel = self._run_mel(prepared)
        # Embedding brut (1, D) -> (D,).
        emb = self.emb.run([self.emb_out.name], {self.emb_in.name: mel})[0]
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        # Normalisation L2 pour stabiliser le score.
        emb /= float(np.linalg.norm(emb) + 1e-12)
        return emb

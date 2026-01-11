# oww_embedder_onnx.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Any

import numpy as np
import onnxruntime as ort
import openwakeword


def _oww_model_path(name: str) -> str:
    base = Path(openwakeword.__file__).resolve().parent / "resources" / "models"
    p = base / name
    if not p.exists():
        raise FileNotFoundError(f"openWakeWord model not found: {p}")
    return str(p)


def _to_nhwc_1tfx1(m: np.ndarray, f_req: int) -> np.ndarray:
    """
    Convertit une sortie mel quelconque vers [1, T, F, 1] (NHWC) float32.

    Gère typiquement :
      - [1, 1, T, F] (NCHW)
      - [1, T, F, 1] (NHWC)
      - [1, T, F]
      - [T, F]
      - [1, F, T]
      - [F, T]
    """
    m = np.asarray(m, dtype=np.float32)

    if m.ndim == 4:
        # NHWC déjà
        if m.shape[0] == 1 and m.shape[-1] == 1 and m.shape[-2] == f_req:
            return m
        # NCHW classique [1,1,T,F]
        if m.shape[0] == 1 and m.shape[1] == 1 and m.shape[-1] == f_req:
            return np.transpose(m, (0, 2, 3, 1)).astype(np.float32)

        # Fallback: détecter l'axe F
        shape = m.shape
        try:
            f_axis = shape.index(f_req)
        except ValueError:
            raise ValueError(f"Mel output doesn't contain F={f_req}. shape={shape}")

        # batch = 0 ; channel = axe de taille 1 (hors batch) si existe
        c_axis = None
        for ax in range(1, 4):
            if ax != f_axis and shape[ax] == 1:
                c_axis = ax
                break

        # temps = axe restant
        candidates = [ax for ax in range(1, 4) if ax != f_axis and ax != c_axis]
        if not candidates:
            raise ValueError(f"Cannot infer time axis from mel shape={shape}")
        t_axis = candidates[0]

        if c_axis is None:
            # pas de channel -> on ajoute
            # transpose (batch, time, freq, ?)
            m2 = np.transpose(m, (0, t_axis, f_axis, [ax for ax in range(1, 4) if ax not in (t_axis, f_axis)][0]))
            # écrase la dernière dim en channel=1 via moyenne
            m2 = m2.mean(axis=-1, keepdims=True)
            return m2.astype(np.float32)

        return np.transpose(m, (0, t_axis, f_axis, c_axis)).astype(np.float32)

    if m.ndim == 3:
        # [1, T, F]
        if m.shape[0] == 1 and m.shape[-1] == f_req:
            return m[:, :, :, np.newaxis].astype(np.float32)
        # [1, F, T]
        if m.shape[0] == 1 and m.shape[1] == f_req:
            return np.transpose(m, (0, 2, 1))[:, :, :, np.newaxis].astype(np.float32)
        raise ValueError(f"Unexpected 3D mel shape={m.shape}")

    if m.ndim == 2:
        # [T, F]
        if m.shape[1] == f_req:
            return m[np.newaxis, :, :, np.newaxis].astype(np.float32)
        # [F, T]
        if m.shape[0] == f_req:
            return m.T[np.newaxis, :, :, np.newaxis].astype(np.float32)
        raise ValueError(f"Unexpected 2D mel shape={m.shape}")

    raise ValueError(f"Unexpected mel ndim={m.ndim}, shape={m.shape}")


def _crop_or_pad_time(m: np.ndarray, t_req: int) -> np.ndarray:
    """
    m: [1, T, F, 1] -> [1, t_req, F, 1] (center crop/pad).
    """
    t = int(m.shape[1])
    if t == t_req:
        return m
    if t > t_req:
        start = (t - t_req) // 2
        return m[:, start:start + t_req, :, :]
    pad = t_req - t
    left = pad // 2
    right = pad - left
    return np.pad(m, ((0, 0), (left, right), (0, 0), (0, 0)), mode="constant")


def _crop_or_pad_freq(m: np.ndarray, f_req: int) -> np.ndarray:
    """
    m: [1, T, F, 1] -> [1, T, f_req, 1]
    """
    f = int(m.shape[2])
    if f == f_req:
        return m
    if f > f_req:
        return m[:, :, :f_req, :]
    return np.pad(m, ((0, 0), (0, 0), (0, f_req - f), (0, 0)), mode="constant")


class OWWEmbedderONNX:
    """
    Embedder openWakeWord (mel -> embedding) en ONNXRuntime.
    Évite complètement les problèmes tflite_runtime/XNNPACK.

    - melspectrogram.onnx
    - embedding_model.onnx
    """

    def __init__(self, threads: int = 1):
        self.mel_path = _oww_model_path("melspectrogram.onnx")
        self.emb_path = _oww_model_path("embedding_model.onnx")

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = int(threads)

        self.mel = ort.InferenceSession(self.mel_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.emb = ort.InferenceSession(self.emb_path, sess_options=so, providers=["CPUExecutionProvider"])

        self.mel_in = self.mel.get_inputs()[0]
        self.mel_out = self.mel.get_outputs()[0]
        self.emb_in = self.emb.get_inputs()[0]
        self.emb_out = self.emb.get_outputs()[0]

        # audio length attendu par mel (si statique)
        self.audio_len = 16000
        if isinstance(self.mel_in.shape, (list, tuple)) and len(self.mel_in.shape) >= 2:
            if isinstance(self.mel_in.shape[1], int):
                self.audio_len = int(self.mel_in.shape[1])

        # shape attendue par embedding (si statique)
        self.t_req = 76
        self.f_req = 32
        if isinstance(self.emb_in.shape, (list, tuple)) and len(self.emb_in.shape) >= 4:
            if isinstance(self.emb_in.shape[1], int):
                self.t_req = int(self.emb_in.shape[1])
            if isinstance(self.emb_in.shape[2], int):
                self.f_req = int(self.emb_in.shape[2])

    def embed(self, audio_16k: np.ndarray) -> np.ndarray:
        x = np.asarray(audio_16k, dtype=np.float32).reshape(-1)

        # pad/crop à audio_len (centre si trop long)
        if x.size < self.audio_len:
            x = np.pad(x, (0, self.audio_len - x.size))
        elif x.size > self.audio_len:
            start = (x.size - self.audio_len) // 2
            x = x[start:start + self.audio_len]

        xin = x.reshape(1, -1).astype(np.float32)
        mel_raw = self.mel.run([self.mel_out.name], {self.mel_in.name: xin})[0]

        mel_nhwc = _to_nhwc_1tfx1(mel_raw, self.f_req)
        mel_nhwc = _crop_or_pad_freq(mel_nhwc, self.f_req)
        mel_in = _crop_or_pad_time(mel_nhwc, self.t_req).astype(np.float32)

        e = self.emb.run([self.emb_out.name], {self.emb_in.name: mel_in})[0]
        e = np.asarray(e, dtype=np.float32).reshape(-1)
        e /= float(np.linalg.norm(e) + 1e-12)
        return e

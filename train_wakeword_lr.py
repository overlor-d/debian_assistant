# train_ovos_lr.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

from oww_embedder_onnx import OWWEmbedderONNX


def list_wavs(d: Path) -> List[Path]:
    return sorted([p for p in d.glob("*.wav") if p.is_file()])


def load_audio_mono(path: Path) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=True, dtype="float32")
    x = x.mean(axis=1)
    x = x - float(np.mean(x)) if x.size else x
    return x, int(sr)


def resample_linear(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr or x.size == 0:
        return x.astype(np.float32, copy=False)
    n_out = int(round(x.size * (target_sr / sr)))
    if n_out <= 1:
        return np.zeros((0,), dtype=np.float32)
    t_in = np.linspace(0.0, 1.0, num=x.size, endpoint=False, dtype=np.float32)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float32)
    return np.interp(t_out, t_in, x).astype(np.float32)


def rms_normalize(x: np.ndarray, target_rms: float = 0.03) -> np.ndarray:
    if x.size == 0:
        return x
    r = float(np.sqrt(np.mean(x * x) + 1e-12))
    g = float(np.clip(target_rms / (r + 1e-12), 0.1, 10.0))
    return np.tanh(x * g).astype(np.float32)


def center_crop_on_energy_len(x: np.ndarray, sr: int, target_len: int) -> np.ndarray:
    """
    Retourne exactement target_len échantillons, centré sur un pic d'énergie (RMS frame).
    """
    if target_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    if x.size == 0:
        return np.zeros((target_len,), dtype=np.float32)

    frame = max(int(round(0.02 * sr)), 1)
    hop = max(int(round(0.01 * sr)), 1)

    x2 = x if x.size >= frame else np.pad(x, (0, frame - x.size))
    n_frames = 1 + (x2.size - frame) // hop
    rms = np.empty((n_frames,), dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        seg = x2[s:s + frame]
        rms[i] = float(np.sqrt(np.mean(seg * seg) + 1e-12))
    i_max = int(np.argmax(rms))
    center = i_max * hop + frame // 2

    start = center - target_len // 2
    end = start + target_len

    if start < 0:
        x = np.pad(x, (-start, 0))
        start = 0
        end = target_len
    if end > x.size:
        x = np.pad(x, (0, end - x.size))
    return x[start:end].astype(np.float32)


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Augmentation minimaliste (CPU cheap) :
    - gain
    - petit bruit blanc
    - petit shift temporel
    """
    y = x.astype(np.float32, copy=True)

    gain = float(rng.uniform(0.7, 1.3))
    y *= gain

    noise = rng.normal(0.0, 0.003, size=y.shape).astype(np.float32)
    y += noise

    shift = int(rng.integers(-200, 200))
    y = np.roll(y, shift)

    return np.tanh(y).astype(np.float32)


def train_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, lr: float = 0.1, steps: int = 2500) -> Tuple[np.ndarray, float]:
    n, d = X.shape
    w = np.zeros((d,), dtype=np.float32)
    b = 0.0
    for _ in range(steps):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        gw = (X.T @ (p - y)) / n + l2 * w
        gb = float(np.mean(p - y))
        w -= (lr * gw).astype(np.float32)
        b -= lr * gb
    return w, float(b)


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    z = X @ w + b
    p = 1.0 / (1.0 + np.exp(-z))
    return p.astype(np.float32)


def auc_eer(pos: np.ndarray, neg: np.ndarray) -> Tuple[float, float, float]:
    s = np.concatenate([pos, neg]).astype(np.float64)
    y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)]).astype(np.int32)

    thr = np.unique(s)[::-1]
    thr = np.concatenate([[thr[0] + 1e-9], thr, [thr[-1] - 1e-9]])

    P = float(pos.size)
    N = float(neg.size)

    tpr_list = []
    fpr_list = []
    eer = 1.0
    eer_thr = float(thr[0])
    best = 1e9

    for t in thr:
        pred = (s >= t).astype(np.int32)
        tp = float(np.sum((pred == 1) & (y == 1)))
        fp = float(np.sum((pred == 1) & (y == 0)))
        fn = P - tp
        tn = N - fp
        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

        diff = abs(fpr - (1.0 - tpr))
        if diff < best:
            best = diff
            eer = (fpr + (1.0 - tpr)) / 2.0
            eer_thr = float(t)

    fpr_arr = np.array(fpr_list, dtype=np.float64)
    tpr_arr = np.array(tpr_list, dtype=np.float64)
    order = np.argsort(fpr_arr)
    
    if hasattr(np, "trapezoid"):
        auc = float(np.trapezoid(tpr_arr[order], fpr_arr[order]))
    else:
        auc = float(np.trapz(tpr_arr[order], fpr_arr[order]))

    return auc, float(eer), float(eer_thr)


def zero_fp_threshold(pos: np.ndarray, neg: np.ndarray) -> Tuple[float, float]:
    t = float(np.max(neg) + 1e-9)
    fnr = float(np.mean(pos < t))
    return t, fnr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positive", required=True, help="Dossier WAV positifs")
    ap.add_argument("--negative", required=True, help="Dossier WAV négatifs")
    ap.add_argument("--out", default="ovos_lr_model.npz", help="Modèle entraîné (npz)")
    ap.add_argument("--report", default="ovos_lr_report.json", help="Rapport JSON")
    ap.add_argument("--augment", type=int, default=5, help="Nb augmentations par fichier (défaut 5)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2500)
    args = ap.parse_args()

    pos_files = list_wavs(Path(args.positive))
    neg_files = list_wavs(Path(args.negative))
    if not pos_files or not neg_files:
        raise SystemExit("Dossier positifs/négatifs vide.")

    rng = np.random.default_rng(args.seed)
    embedder = OWWEmbedderONNX(threads=args.threads)

    def embed_file(f: Path) -> np.ndarray:
        x, sr = load_audio_mono(f)
        x = resample_linear(x, sr, 16000)
        x = rms_normalize(x)
        x = center_crop_on_energy_len(x, 16000, target_len=embedder.audio_len)
        return embedder.embed(x)

    X_list = []
    y_list = []

    # positifs
    for f in pos_files:
        base = embed_file(f)
        X_list.append(base); y_list.append(1.0)
        for _ in range(args.augment):
            # augment sur waveform déjà crop (longueur fixe)
            x, sr = load_audio_mono(f)
            x = resample_linear(x, sr, 16000)
            x = rms_normalize(x)
            x = center_crop_on_energy_len(x, 16000, target_len=embedder.audio_len)
            xa = augment(x, rng)
            X_list.append(embedder.embed(xa)); y_list.append(1.0)

    # négatifs
    for f in neg_files:
        base = embed_file(f)
        X_list.append(base); y_list.append(0.0)
        for _ in range(args.augment):
            x, sr = load_audio_mono(f)
            x = resample_linear(x, sr, 16000)
            x = rms_normalize(x)
            x = center_crop_on_energy_len(x, 16000, target_len=embedder.audio_len)
            xa = augment(x, rng)
            X_list.append(embedder.embed(xa)); y_list.append(0.0)

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)

    # standardisation
    mu = X.mean(axis=0).astype(np.float32)
    sd = (X.std(axis=0) + 1e-6).astype(np.float32)
    Xs = (X - mu) / sd

    w, b = train_logreg(Xs, y, l2=1e-3, lr=0.1, steps=int(args.steps))
    p = predict_proba(Xs, w, b)
    p_pos = p[y == 1.0]
    p_neg = p[y == 0.0]

    auc, eer, eer_thr = auc_eer(p_pos, p_neg)
    zfp_thr, zfp_fnr = zero_fp_threshold(p_pos, p_neg)

    report = {
        "counts": {"pos_files": len(pos_files), "neg_files": len(neg_files), "train_examples": int(X.shape[0])},
        "embedder": {"audio_len": int(embedder.audio_len), "t_req": int(embedder.t_req), "f_req": int(embedder.f_req)},
        "metrics": {"auc": auc, "eer": eer, "eer_threshold": eer_thr, "zero_fp_threshold": zfp_thr, "fnr_at_zero_fp": zfp_fnr},
        "note": "zero_fp_threshold = 0 FP sur TES négatifs. Fiabilité réelle nécessite des négatifs 'vraie vie' (TV, autres voix, etc.)."
    }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    np.savez(
        args.out,
        w=w, b=b, mu=mu, sd=sd,
        threshold=zfp_thr,
        audio_len=np.array([embedder.audio_len], dtype=np.int32),
    )

    print("OK")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

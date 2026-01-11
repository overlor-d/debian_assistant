# run_ovos_lr.py
from __future__ import annotations

import argparse
import time
import threading
import numpy as np
import sounddevice as sd

from oww_embedder_onnx import OWWEmbedderONNX


def rms_normalize(x: np.ndarray, target_rms: float = 0.03) -> np.ndarray:
    r = float(np.sqrt(np.mean(x * x) + 1e-12))
    g = float(np.clip(target_rms / (r + 1e-12), 0.1, 10.0))
    return np.tanh(x * g).astype(np.float32)


def downsample_48k_to_16k(x: np.ndarray) -> np.ndarray:
    # facteur 3 exact (48k -> 16k). Si ton device n'est pas exactement 48k, ne pas utiliser.
    return x[::3].astype(np.float32, copy=False)


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ovos_lr_model.npz")
    ap.add_argument("--mic", type=int, default=9)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--hop_ms", type=int, default=100)
    ap.add_argument("--cooldown_s", type=float, default=1.5)
    ap.add_argument("--consecutive", type=int, default=2, help="nb hops consécutifs au-dessus du seuil")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    pack = np.load(args.model)
    w = pack["w"].astype(np.float32)
    b = float(pack["b"])
    mu = pack["mu"].astype(np.float32)
    sdv = pack["sd"].astype(np.float32)
    thr = float(pack["threshold"])
    audio_len = int(pack["audio_len"][0]) if "audio_len" in pack else 16000

    embedder = OWWEmbedderONNX(threads=args.threads)
    # sécurité : audio_len doit matcher l’embedder
    audio_len = int(embedder.audio_len) if int(embedder.audio_len) > 0 else audio_len

    hop_48 = int(round(args.sr * (args.hop_ms / 1000.0)))
    if hop_48 <= 0:
        raise SystemExit("hop_ms invalide.")
    if args.sr != 48000:
        raise SystemExit("Ce script suppose sr=48000 pour downsample x[::3]. Mets sr=48000 ou adapte le resampling.")

    # ring buffer en 16k
    ring = np.zeros((audio_len,), dtype=np.float32)
    lock = threading.Lock()
    filled = False
    last_fire = 0.0
    streak = 0

    def push_16k(samples_16k: np.ndarray):
        nonlocal ring, filled
        n = samples_16k.size
        if n <= 0:
            return
        if n >= ring.size:
            ring[:] = samples_16k[-ring.size:]
            filled = True
            return
        ring[:-n] = ring[n:]
        ring[-n:] = samples_16k
        filled = True

    def callback(indata, frames, time_info, status):
        x = indata[:, 0].astype(np.float32)
        x16 = downsample_48k_to_16k(x)
        with lock:
            push_16k(x16)

    with sd.InputStream(device=args.mic, channels=1, samplerate=args.sr, blocksize=hop_48, callback=callback):
        while True:
            time.sleep(args.hop_ms / 1000.0)

            with lock:
                if not filled:
                    continue
                x16 = ring.copy()

            x16 = rms_normalize(x16)
            e = embedder.embed(x16)

            xs = (e - mu) / sdv
            p = float(sigmoid(float(xs @ w + b)))

            if p >= thr:
                streak += 1
            else:
                streak = 0

            now = time.time()
            if streak >= args.consecutive and (now - last_fire) >= args.cooldown_s:
                last_fire = now
                streak = 0
                print("TRUE", flush=True)


if __name__ == "__main__":
    main()

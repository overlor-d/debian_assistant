"""
Entrainement et evaluation d'un modele wakeword (regression logistique).

Pourquoi: fournir un apprentissage simple, interpretable et portable.
Comment: normalisation des embeddings, descente de gradient, puis metriques
(AUC/EER) et seuils de decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Transforme un score lineaire en probabilite (logistique).

    Pourquoi: la regression logistique apprend un score lineaire non borne.
    Cette fonction le comprime dans [0, 1] pour l'interpreter comme proba.
    Comment: formule standard 1 / (1 + exp(-z)), vectorisee par NumPy.
    """
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class LogisticWakewordModel:
    """
    Modele de regression logistique avec normalisation des embeddings.

    Pourquoi: les embeddings openWakeWord ont des distributions variables
    selon le jeu de donnees; la normalisation (moyenne/ecart-type) stabilise
    l'apprentissage et l'inference. La regression logistique fournit ensuite
    un score probabiliste interpretable.
    Comment: on stocke w, b, mean, std, un seuil d'activation et l'audio_len.
    """

    weights: np.ndarray
    bias: float
    mean: np.ndarray
    std: np.ndarray
    threshold: float
    audio_len: int

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        # Centre-reduit: on reutilise mean/std appris a l'entrainement.
        return (embedding - self.mean) / self.std

    def score(self, embedding: np.ndarray) -> float:
        """
        Retourne la probabilite de wakeword pour un embedding 1D.

        Pourquoi: on veut un score interpretable entre 0 et 1 pour un seul
        exemple. Comment: normalisation puis produit scalaire + sigmoid.
        """
        xs = self._normalize(embedding)
        # Score lineaire -> probabilite.
        return float(_sigmoid(float(xs @ self.weights + self.bias)))

    def predict_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Retourne les probabilites pour un lot d'embeddings 2D.

        Pourquoi: l'entrainement/evaluation traite des batchs.
        Comment: operations vectorisees pour la vitesse.
        """
        # Normalisation vectorisee (shape: [N, D]).
        xs = (embeddings - self.mean) / self.std
        # Sigmoid appliquee a tous les scores en meme temps.
        return _sigmoid(xs @ self.weights + self.bias).astype(np.float32)

    def save(self, path: Path) -> None:
        """
        Sauvegarde le modele dans un fichier npz.

        Pourquoi: format compact, chargeable rapidement, compatible NumPy.
        Comment: on stocke chaque tenseur dans une cle stable.
        """
        np.savez(
            path,
            w=self.weights.astype(np.float32),
            b=np.array([self.bias], dtype=np.float32),
            mu=self.mean.astype(np.float32),
            sd=self.std.astype(np.float32),
            threshold=np.array([self.threshold], dtype=np.float32),
            audio_len=np.array([self.audio_len], dtype=np.int32),
        )

    @classmethod
    def load(cls, path: Path) -> "LogisticWakewordModel":
        """
        Charge un modele depuis un npz en preservant la compatibilite.

        Pourquoi: certains champs peuvent manquer si le modele a ete cree
        par une version plus ancienne. Comment: valeurs par defaut propres.
        """
        pack = np.load(path)
        weights = pack["w"].astype(np.float32)
        b_arr = pack["b"]
        bias = float(b_arr.reshape(-1)[0])
        mean = pack["mu"].astype(np.float32)
        std = pack["sd"].astype(np.float32)
        threshold = float(pack["threshold"].reshape(-1)[0]) if "threshold" in pack else 0.5
        audio_len = int(pack["audio_len"][0]) if "audio_len" in pack else 16000
        return cls(weights=weights, bias=bias, mean=mean, std=std, threshold=threshold, audio_len=audio_len)


def train_logreg(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1e-3,
    lr: float = 0.1,
    steps: int = 2500,
) -> tuple[np.ndarray, float]:
    """
    Apprend une regression logistique L2 par descente de gradient.

    Pourquoi: controle fin et simplicite (pas de dependance externe).
    Comment: on minimise la log-loss + l2 en iterant des mises a jour.
    """
    n, d = X.shape
    w = np.zeros((d,), dtype=np.float32)
    b = 0.0
    for _ in range(int(steps)):
        # Logits lineaires pour chaque exemple.
        z = X @ w + b
        # Probabilites courant (sigmoid).
        p = _sigmoid(z)
        # Gradients de la loss logistique + regularisation L2.
        grad_w = (X.T @ (p - y)) / n + l2 * w
        grad_b = float(np.mean(p - y))
        # Mise a jour des parametres (SGD batch complet).
        w -= (lr * grad_w).astype(np.float32)
        b -= lr * grad_b
    return w, float(b)


def auc_eer(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float, float]:
    """
    Calcule l'AUC et l'EER en balayant tous les seuils.

    Pourquoi: ces metriques caracterisent la separabilite pos/neg et le
    compromis entre faux positifs et faux negatifs.
    Comment: on trie les scores, on evalue TPR/FPR, puis on integre.
    Retourne (auc, eer, thr_eer).
    """
    scores = np.concatenate([pos, neg]).astype(np.float64)
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)]).astype(np.int32)

    # Seuils possibles: toutes les valeurs observees + marges.
    thresholds = np.unique(scores)[::-1]
    thresholds = np.concatenate([[thresholds[0] + 1e-9], thresholds, [thresholds[-1] - 1e-9]])

    P = float(pos.size)
    N = float(neg.size)

    tpr_list = []
    fpr_list = []
    eer = 1.0
    eer_thr = float(thresholds[0])
    best = 1e9

    for t in thresholds:
        # Predire 1 si score >= seuil.
        pred = (scores >= t).astype(np.int32)
        tp = float(np.sum((pred == 1) & (labels == 1)))
        fp = float(np.sum((pred == 1) & (labels == 0)))
        fn = P - tp
        tn = N - fp
        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

        diff = abs(fpr - (1.0 - tpr))
        if diff < best:
            best = diff
            # EER: point ou FPR ~= FNR (1 - TPR).
            eer = (fpr + (1.0 - tpr)) / 2.0
            eer_thr = float(t)

    fpr_arr = np.array(fpr_list, dtype=np.float64)
    tpr_arr = np.array(tpr_list, dtype=np.float64)
    order = np.argsort(fpr_arr)
    x = fpr_arr[order]
    y = tpr_arr[order]
    # Aire sous la courbe (AUC) par integration trapezoidale.
    if hasattr(np, "trapezoid"):
        auc = float(np.trapezoid(y, x))
    elif hasattr(np, "trapz"):
        auc = float(np.trapz(y, x))
    else:
        # intégration trapèzes manuelle
        auc = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))

    return auc, float(eer), float(eer_thr)


def zero_fp_threshold(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """
    Seuil minimal eliminant les faux positifs sur les negatifs observes.

    Pourquoi: en detection wakeword, eviter les faux positifs est prioritaire.
    Comment: on prend le max des scores negatifs + une marge epsilon.
    Retourne (threshold, fnr) ou fnr est calcule sur les positifs.
    """
    threshold = float(np.max(neg) + 1e-9)
    fnr = float(np.mean(pos < threshold))
    return threshold, fnr


def train_wakeword_model(
    embeddings: np.ndarray,
    labels: np.ndarray,
    steps: int,
    l2: float = 1e-3,
    lr: float = 0.1,
    audio_len: int = 16000,
) -> tuple[LogisticWakewordModel, dict]:
    """
    Entraine une regression logistique sur des embeddings et retourne
    le modele + un rapport de metriques.

    Pourquoi: encapsuler la normalisation, l'entrainement et l'evaluation
    dans un seul flux reproductible.
    Comment: on normalise, on apprend w/b, on calcule AUC/EER et seuils.
    """
    # Statistiques de normalisation apprises sur le train.
    mean = embeddings.mean(axis=0).astype(np.float32)
    std = (embeddings.std(axis=0) + 1e-6).astype(np.float32)
    normalized = (embeddings - mean) / std

    # Apprentissage du classifieur lineaire sur donnees normalisees.
    weights, bias = train_logreg(normalized, labels, l2=l2, lr=lr, steps=steps)
    probs = _sigmoid(normalized @ weights + bias)

    # Separation des scores par classe pour les metriques.
    pos_scores = probs[labels == 1.0]
    neg_scores = probs[labels == 0.0]

    # Metriques globales + seuils utiles.
    auc, eer, eer_thr = auc_eer(pos_scores, neg_scores)
    zfp_thr, zfp_fnr = zero_fp_threshold(pos_scores, neg_scores)

    # Construction du modele avec tous les parametres necessaires.
    model = LogisticWakewordModel(
        weights=weights.astype(np.float32),
        bias=float(bias),
        mean=mean,
        std=std,
        threshold=float(zfp_thr),
        audio_len=int(audio_len),
    )

    # Rapport JSON simple pour suivi d'entrainement.
    metrics = {
        "counts": {
            "positives": int((labels == 1.0).sum()),
            "negatives": int((labels == 0.0).sum()),
            "total": int(labels.size),
            "train_examples": int(embeddings.shape[0]),
        },
        "embedder": {"audio_len": int(audio_len)},
        "metrics": {
            "auc": float(auc),
            "eer": float(eer),
            "eer_threshold": float(eer_thr),
            "zero_fp_threshold": float(zfp_thr),
            "fnr_at_zero_fp": float(zfp_fnr),
        },
        "note": "zero_fp_threshold = 0 FP sur l'échantillon de tests négatifs. Valider avec des données réelles variées.",
    }
    return model, metrics


def save_report(report: dict, path: Path) -> None:
    """
    Ecrit un rapport JSON lisible sur disque.

    Pourquoi: conserver les metriques pour comparaison d'experiences.
    Comment: json.dumps indent=2 avec UTF-8.
    """
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

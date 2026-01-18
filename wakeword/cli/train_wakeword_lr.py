"""
CLI d'entrainement d'un modele wakeword (regression logistique).

Pourquoi: exposer un flux complet "dataset -> modele -> rapport" en une commande.
Comment: preparation d'embeddings puis apprentissage et sauvegarde sur disque.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wakeword.dataset import EmbeddingDataset, build_embedding_dataset
from wakeword.embedder import WakewordEmbedder
from wakeword.training import save_report, train_wakeword_model


def build_parser() -> argparse.ArgumentParser:
    """
    Construit le parser d'arguments.

    Pourquoi: parametrer facilement l'entrainement depuis la CLI.
    Comment: options pour les dossiers, hyperparametres et sorties.
    """
    parser = argparse.ArgumentParser(description="Entraînement d'un modèle wakeword (régression logistique).")
    parser.add_argument("--positive", required=True, help="Dossier WAV positifs")
    parser.add_argument("--negative", required=True, help="Dossier WAV négatifs")
    parser.add_argument("--out", default="ovos_lr_model.npz", help="Fichier modèle (npz)")
    parser.add_argument("--report", default="ovos_lr_report.json", help="Rapport JSON de l'entraînement")
    parser.add_argument("--augment", type=int, default=5, help="Nb d'augmentations par fichier")
    parser.add_argument("--seed", type=int, default=123, help="Seed RNG pour reproductibilité")
    parser.add_argument("--threads", type=int, default=1, help="Threads ONNXRuntime")
    parser.add_argument("--steps", type=int, default=2500, help="Itérations de descente de gradient")
    parser.add_argument("--l2", type=float, default=1e-3, help="Régularisation L2")
    parser.add_argument("--lr", type=float, default=0.1, help="Taux d'apprentissage")
    return parser


def create_rng(seed: int) -> np.random.Generator:
    """
    Construit un generateur pseudo-aleatoire pour les augmentations.

    Pourquoi: rendre les runs reproductibles via un seed.
    Comment: np.random.default_rng avec une seed explicite.
    """
    return np.random.default_rng(seed)


def build_dataset(
    positive_dir: Path,
    negative_dir: Path,
    embedder: WakewordEmbedder,
    rng: np.random.Generator,
    augmentations: int,
) -> EmbeddingDataset:
    """
    Construit le dataset d'embeddings a partir des dossiers audio.

    Pourquoi: isoler la preparation des donnees de la logique CLI.
    Comment: delegation a wakeword.dataset.build_embedding_dataset.
    """
    return build_embedding_dataset(
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        embedder=embedder,
        rng=rng,
        augmentations=int(augmentations),
    )


def add_source_counts(report: dict, dataset: EmbeddingDataset) -> None:
    """
    Ajoute les compteurs de fichiers source au rapport.

    Pourquoi: faciliter l'audit et la traçabilite des donnees.
    Comment: update du bloc report["counts"].
    """
    report["counts"].update(
        {
            "pos_files": len(dataset.positive_files),
            "neg_files": len(dataset.negative_files),
        }
    )


def print_training_summary(report: dict, model_path: Path, report_path: Path) -> None:
    """
    Affiche un resume lisible de l'entrainement.

    Pourquoi: feedback immediat pour l'utilisateur CLI.
    Comment: formatage simple dans la console.
    """
    print("Modèle entraîné et sauvegardé.")
    print(f"- Modèle : {model_path}")
    print(f"- Rapport : {report_path}")
    print(f"- AUC : {report['metrics']['auc']:.3f}")
    print(f"- EER : {report['metrics']['eer']:.3f} (thr={report['metrics']['eer_threshold']:.3f})")
    print(
        f"- Seuil zéro FP : {report['metrics']['zero_fp_threshold']:.3f} | "
        f"FNR@0FP={report['metrics']['fnr_at_zero_fp']:.3f}"
    )


def main() -> None:
    """
    Point d'entree CLI d'entrainement.

    Pourquoi: orchestrer preparation du dataset et entrainement LR.
    Comment: instancie l'embedder, construit X/y, puis entraine et sauvegarde.
    """
    args = build_parser().parse_args()
    # RNG pour que les augmentations soient reproductibles.
    rng = create_rng(args.seed)

    # Embedder ONNX pour produire les vecteurs openWakeWord.
    embedder = WakewordEmbedder(threads=args.threads)
    dataset = build_dataset(
        positive_dir=Path(args.positive),
        negative_dir=Path(args.negative),
        embedder=embedder,
        rng=rng,
        augmentations=args.augment,
    )

    # Entrainement de la regression logistique.
    model, report = train_wakeword_model(
        embeddings=dataset.embeddings,
        labels=dataset.labels,
        steps=int(args.steps),
        l2=float(args.l2),
        lr=float(args.lr),
        audio_len=dataset.audio_len,
    )

    # Ajout d'un résumé sur les fichiers source.
    add_source_counts(report, dataset)

    # Sauvegardes: modele + rapport JSON.
    model_path = Path(args.out)
    report_path = Path(args.report)
    model.save(model_path)
    save_report(report, report_path)

    # Sortie utilisateur lisible.
    print_training_summary(report, model_path, report_path)


if __name__ == "__main__":
    main()

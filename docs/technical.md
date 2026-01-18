# Documentation technique

Ce document décrit les rôles de chaque fichier et les principaux flux internes (prétraitement audio, embedding ONNX, entraînement LR, détection temps réel).

## Vue d'ensemble
```
wakeword/
  audio.py          # utilitaires audio (I/O, normalisation, resampling, crop, augmentations)
  embedder.py       # wrapper ONNX openWakeWord
  dataset.py        # préparation du dataset (embeddings + labels)
  training.py       # régression logistique, métriques, sérialisation modèle
  inference.py      # boucle de détection temps réel
  recording.py      # helpers d'enregistrement d'échantillons
  cli/              # scripts CLI (record, train, run, download models, detect mic)
docs/               # documentation technique
```

## Modules

### wakeword/audio.py
- `list_wav_files(directory)` : renvoie la liste triée des WAV dans un dossier.
- `load_mono_audio(path)` : charge un WAV en float32 mono, supprime l'offset DC pour éviter les biais.
- `resample_linear(samples, source_sr, target_sr)` : rééchantillonnage linéaire léger (pas de dépendance externe).
- `rms_normalize(samples, target_rms)` : normalise le niveau RMS et applique un `tanh` pour limiter les pics.
- `center_crop_on_energy(samples, sample_rate, target_len)` : trouve le frame RMS le plus énergique et centre un crop de longueur fixe dessus.
- `augment_waveform(samples, rng, gain_range, noise_std, max_shift)` : applique gain aléatoire, bruit blanc et léger décalage circulaire pour enrichir l'entraînement.
- `downsample_48k_to_16k(samples)` : sous-échantillonnage x3 (suppose strictement 48 kHz en entrée).

### wakeword/embedder.py
- `resolve_oww_model_path(model_name)` : récupère le chemin d'un modèle ONNX openWakeWord inclus dans le package.
- `_crop_or_pad_axis(arr, axis, target)` : utilitaire interne de crop/pad centré.
- `_format_mel_to_nhwc(mel, expected_freq_bins)` : harmonise les sorties mel éventuelles vers le format `[1, T, F, 1]` attendu par l'embedder ONNX en détectant les axes temps/fréquence.
- `_prepare_waveform(audio_16k, target_len)` : pad/crop central pour que le graphe mel accepte la longueur exacte.
- `WakewordEmbedder` :
  - Initialise deux `InferenceSession` ONNXRuntime (`melspectrogram.onnx` puis `embedding_model.onnx`), expose les tailles attendues (`audio_len`, `t_req`, `f_req`).
  - `_run_mel(waveform)` : exécute la passe mel et applique formatage/crop sur les axes temps/fréquence.
  - `embed_waveform(audio_16k)` : produit un embedding L2-normalisé pour un signal mono 16 kHz.

### wakeword/dataset.py
- `EmbeddingDataset` : dataclass regroupant embeddings, labels, listes de fichiers et longueur audio utilisée.
- `_embed_file_with_augmentations(file_path, label, embedder, rng, augmentations, target_sr)` : génère l'embedding de base + N augmentations pour un fichier donné.
- `build_embedding_dataset(positive_dir, negative_dir, embedder, rng, augmentations, target_sr)` :
  - Charge les WAV, normalise, crop sur l'énergie, applique les augmentations.
  - Empile les embeddings positifs/négatifs dans des matrices `X` et `y`.
  - Retourne un `EmbeddingDataset` prêt pour l'entraînement.

### wakeword/training.py
- `_sigmoid(z)` : activation logistique.
- `LogisticWakewordModel` : encapsule poids/biais de la LR, stats de normalisation et seuil de déclenchement. Méthodes :
  - `_normalize(embedding)` : centre/réduit un embedding.
  - `score(embedding)` : probabilité wakeword pour un embedding individuel.
  - `predict_batch(embeddings)` : probabilités pour un lot.
  - `save(path)` / `load(path)` : sérialisation au format `npz` (poids `w`, biais `b`, moyennes `mu`, écart-types `sd`, seuil `threshold`, `audio_len`).
- `train_logreg(X, y, l2, lr, steps)` : descente de gradient simple avec régularisation L2.
- `auc_eer(pos, neg)` : calcule AUC, EER et seuil EER en balayant tous les seuils uniques.
- `zero_fp_threshold(pos, neg)` : plus petit seuil éliminant les FP sur l'échantillon négatif, retourne aussi le FNR associé.
- `train_wakeword_model(embeddings, labels, steps, l2, lr, audio_len)` :
  - Calcule moyennes/écarts-types, entraîne la LR, puis dérive les métriques (AUC, EER, seuil zéro FP).
  - Construit un `LogisticWakewordModel` et un dictionnaire de rapport (counts, métriques, note d'usage).
- `save_report(report, path)` : écrit le rapport JSON (UTF-8, indenté).

### wakeword/inference.py
- `SlidingWindow(length)` : buffer circulaire 1D, expose `push(samples)` et `get()`.
- `WakewordScorer(model, embedder)` : applique normalisation RMS, embedding ONNX puis LR pour retourner une probabilité.
- `StreamingWakewordDetector(scorer, cooldown_s, consecutive)` :
  - Maintient un ring buffer, un compteur de streak et un cooldown.
  - `push_block_48k(block)` : convertit un bloc 48 kHz en 16 kHz et l'insère.
  - `evaluate()` : retourne `(probabilité, déclenché)` ou `None` si le buffer n'est pas rempli.
- `run_realtime_detection(model, embedder, mic_index, sample_rate, hop_ms, cooldown_s, consecutive)` :
  - Boucle CLI bloquante qui ouvre un `sounddevice.InputStream`, alimente le détecteur et imprime un message dès qu'un trigger est validé.
  - Suppose un micro en 48 kHz (sous-échantillonnage x3).

### wakeword/recording.py
- `next_index(folder, prefix)` : trouve l'index suivant pour ne pas écraser les enregistrements.
- `record_sample(folder, prefix, mic_index, samplerate, duration_s, channels, dtype)` : enregistre un fichier WAV dans le dossier voulu.
- `interactive_recording_session(positive_dir, negative_dir, ...)` :
  - Boucle interactive (touches `p`, `n`, `q`) pour capturer rapidement des lots positifs et négatifs.
  - Utilise `record_sample` pour la capture réelle.

### wakeword/cli/*
- `detect_mic.py` : affiche les périphériques d'entrée disponibles avec leur index (`python -m wakeword.cli.detect_mic`).
- `install_models.py` : télécharge les modèles openWakeWord nécessaires à l'embedder ONNX (`python -m wakeword.cli.install_models`).
- `record_wakeword.py` : CLI interactive pour capturer des exemples (options mic, durée, répertoires) (`python -m wakeword.cli.record_wakeword`).
- `train_wakeword_lr.py` : lance la construction du dataset d'embeddings puis l'entraînement LR ; sauvegarde le modèle `npz` et un rapport JSON (`python -m wakeword.cli.train_wakeword_lr`).
- `run_wakeword_lr.py` : charge le modèle `npz`, instancie l'embedder ONNX et exécute la détection temps réel (`python -m wakeword.cli.run_wakeword_lr`).

## Flux principaux
- **Enregistrement** : `wakeword/cli/record_wakeword.py` → `recording.interactive_recording_session` → `record_sample`.
- **Préparation dataset** : `dataset.build_embedding_dataset` orchestre chargement WAV → resampling → RMS norm → crop énergie → augmentations → embedding ONNX.
- **Entraînement** : `training.train_wakeword_model` prend les embeddings/labels, normalise, entraîne la LR, calcule métriques et seuils, puis sérialise via `LogisticWakewordModel.save`.
- **Détection temps réel** : `wakeword/cli/run_wakeword_lr.py` charge le modèle → instancie `WakewordEmbedder` → boucle `run_realtime_detection` qui alimente `StreamingWakewordDetector` avec des blocs micro (48 kHz) et imprime un trigger quand le streak dépasse le seuil/cooldown.

## Format du modèle (npz)
- `w` : poids LR (float32, dimension = embedding_dim)
- `b` : biais (float32 scalaire)
- `mu` : moyennes d'entraînement par dimension d'embedding
- `sd` : écarts-types (avec epsilon pour éviter les divisions par zéro)
- `threshold` : seuil calibré pour 0 faux positif sur l'échantillon négatif
- `audio_len` : longueur de fenêtre audio 16 kHz attendue par l'embedder

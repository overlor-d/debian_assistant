# Wakeword léger (openWakeWord + LR)

Petit utilitaire pour enregistrer des exemples, entraîner une régression logistique sur les embeddings openWakeWord et tester le wakeword en temps réel.

## Prérequis
- Python 3.10+
- PortAudio (paquets `portaudio19-dev` sur Debian/Ubuntu)
- FFmpeg/sox conseillés pour diagnostiquer les WAV
- Dépendances Python : `pip install -r requirements.txt`

Le script `install.sh` installe les paquets système de base puis crée un venv minimal.


## Démarrage rapide
1) Créer/activer le venv (install.sh le crée dans ce repo)  
   `python -m venv env && source env/bin/activate`  
   *(ou exécuter `scripts/install.sh` avec source)*  
2) Installer les dépendances Python  
   `pip install -r requirements.txt`
3) Télécharger les modèles openWakeWord  
   `python -m wakeword.cli.install_models`
4) Lister les micros disponibles  
   `python -m wakeword.cli.detect_mic`
5) Enregistrer des exemples (positifs/négatifs)  
   `python -m wakeword.cli.record_wakeword --mic <index> --base-dir wakeword_data/ovos`
6) Entraîner le modèle  
   `python -m wakeword.cli.train_wakeword_lr --positive wakeword_data/ovos/positive --negative wakeword_data/ovos/negative`
7) Tester en temps réel  
   `python -m wakeword.cli.run_wakeword_lr --mic <index>`

Astuce : commencer avec ~50 exemples positifs et négatifs, puis enrichir avec des négatifs "difficiles" (TV, autres voix...).

## Structure du projet
- `wakeword/` : package principal (audio, embedder ONNX, préparation du dataset, entraînement, détection, enregistrement).
- `wakeword/cli/` : scripts CLI pour enregistrer, entraîner, tester, télécharger les modèles et lister les micros (appelés via `python -m wakeword.cli.<script>`).
- `wakeword_data/` : dossier conseillé pour stocker vos WAV (non versionné).
- `docs/technical.md` : documentation technique détaillée sur les modules/fonctions.

## Données attendues
Les jeux d'entraînement sont attendus en deux dossiers distincts (positif/négatif), par exemple :
```
wakeword_data/
  ovos/
    positive/*.wav
    negative/*.wav
```

## Aide et dépannage
- Vérifiez que le micro enregistre bien à 48 kHz (le script temps réel sous-échantillonne en x3).
- Si aucun trigger n'apparaît, baissez `--consecutive` ou réduisez `--cooldown_s` pour déboguer.
- Les seuils sont calibrés pour 0 faux positif sur le jeu négatif d'entraînement : validez toujours sur des données réelles.

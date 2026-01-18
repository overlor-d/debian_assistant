SHELL := /bin/bash

VENV_DIR ?= env
DATA_DIR ?= wakeword_data
DATA_ARCHIVE ?= audio_data.tar.gz

.PHONY: install clean upload

# Nettoie les artefacts de build et l'env, sans toucher aux données audio.
clean:
	@find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	@find . -name '*.pyc' -delete
	@rm -f *.npz *.json $(DATA_ARCHIVE)

# Archive les données audio pour partage/sauvegarde.
upload:
	@if [ ! -d "$(DATA_DIR)" ]; then echo "DATA_DIR manquant: $(DATA_DIR)"; exit 1; fi
	@tar -czf $(DATA_ARCHIVE) $(DATA_DIR)
	@echo "Archive créée: $(DATA_ARCHIVE)"

#!/bin/bash

set -xe

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source "$SCRIPT_DIR"/.env
source "$SCRIPT_DIR"/.venv/bin/activate

echo "Downloading Docling Models"
docling-tools models download-hf-repo ds4sd/CodeFormulaV2
docling-tools models download-hf-repo ds4sd/DocumentFigureClassifier
docling-tools models download-hf-repo ds4sd/docling-layout-heron
docling-tools models download-hf-repo ds4sd/docling-models

echo "Downloading Huggingface Models"
hf download "$HF_EMBEDDING_MODEL" # Embedding model for qdrant db

echo "Downloading Ollama Models"
ollama pull "$CUSTOM_PICTURE_DESC_MODEL"        # For Visual Language Model for Picture Description
ollama pull "$CUSTOM_TEXT_ANALYSER_MODEL"       # For Visual Language Model for Text Analysis
ollama pull "$RAG_LLM_MODEL"                    # For Q/A
ollama pull "$RERANKING_OLLAMA_EMBEDDING_MODEL" # For Reranking

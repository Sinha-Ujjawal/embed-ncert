#!/bin/bash

set -xe

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source "$SCRIPT_DIR"/.env

if [[ "$DOCLING_ARTIFACTS_PATH" == "" ]]; then
    echo "Env variable DOCLING_ARTIFACTS_PATH must be defined!. Check $SCRIPT_DIR/.env file"
    exit 1
fi

download_hf_repo() {
    local model_repo=$1
    "$SCRIPT_DIR"/.venv/bin/docling-tools models download-hf-repo "$model_repo" -o "$DOCLING_ARTIFACTS_PATH"/
}

# Downloading Docling Models
download_hf_repo   ds4sd/CodeFormulaV2              &
download_hf_repo   ds4sd/DocumentFigureClassifier   &
download_hf_repo   ds4sd/docling-layout-heron       &
download_hf_repo   ds4sd/docling-layout-heron-101   &
download_hf_repo   ds4sd/docling-models             &

wait

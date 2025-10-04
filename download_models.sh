#!/bin/bash

set -xe

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source "$SCRIPT_DIR"/.env

if [[ "$DOCLING_ARTIFACTS_PATH" == "" ]]; then
    echo "Env variable DOCLING_ARTIFACTS_PATH must be defined!. Check $SCRIPT_DIR/.env file"
    exit 1
fi

# Assuming DOCLING_ARTIFACTS_PATH is present

"$SCRIPT_DIR"/.venv/bin/docling-tools models download -o $DOCLING_ARTIFACTS_PATH/

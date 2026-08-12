#!/usr/bin/env bash
# Single build command for Render (and any other host): install deps, then
# rebuild the data + model artifacts from the raw CSVs so the deployed app
# never depends on committed .pkl/.csv files matching the current code.
set -euo pipefail

pip install -r requirements.txt
python -m src.preprocessing
python -m src.features
python -m src.model

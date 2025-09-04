#!/bin/bash
set -e

# Install the spaCy model
python -m pip install --no-cache-dir -r requirements.txt
python -m spacy download en_core_web_sm --no-cache-dir

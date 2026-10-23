#!/usr/bin/env bash
# Descarga el dataset de vocabulario por nivel (CEFR-J / Open Language Profiles).
set -e
cd "$(dirname "$0")"
mkdir -p data
GH=https://raw.githubusercontent.com/openlanguageprofiles/olp-en-cefrj/master
curl -sSL -o data/cefrj-vocabulary.csv $GH/cefrj-vocabulary-profile-1.5.csv        # A1-B2
curl -sSL -o data/octanove-c1c2.csv   $GH/octanove-vocabulary-profile-c1c2-1.0.csv  # C1-C2
echo "Listo:"; ls -la data

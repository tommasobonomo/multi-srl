#!/bin/bash

export PYTHONPATH=${PWD}

echo "Convert CoNLL2009 data with trankit dependencies"

# English
python scripts/preprocessing/convert_conll2009_dep.py \
    --input_dir data/original/conll2009/en \
    --output_dir data/original/conll2009/en_ud25 \
    --language english \
    --strict_language \
    --only_dependencies
    
# Catalan
python scripts/preprocessing/convert_conll2009_dep.py \
    --input_dir data/original/conll2009/ca \
    --output_dir data/original/conll2009/ca_ud25 \
    --language catalan \
    --strict_language \
    --only_dependencies

# Chinese
python scripts/preprocessing/convert_conll2009_dep.py \
    --input_dir data/original/conll2009/zh \
    --output_dir data/original/conll2009/zh_ud25 \
    --language chinese \
    --strict_language \
    --only_dependencies

# Czech
python scripts/preprocessing/convert_conll2009_dep.py \
    --input_dir data/original/conll2009/cs \
    --output_dir data/original/conll2009/cs_ud25 \
    --language czech \
    --strict_language \
    --only_dependencies

# German
python scripts/preprocessing/convert_conll2009_dep.py \
    --input_dir data/original/conll2009/de \
    --output_dir data/original/conll2009/de_ud25 \
    --language german \
    --strict_language \
    --only_dependencies

# Spanish
python scripts/preprocessing/convert_conll2009_dep.py \
    --input_dir data/original/conll2009/es \
    --output_dir data/original/conll2009/es_ud25 \
    --language spanish \
    --strict_language \
    --only_dependencies
    
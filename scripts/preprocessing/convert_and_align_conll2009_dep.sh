#!/bin/bash

export PYTHONPATH=${PWD}

echo "Convert CoNLL2009 data with trankit dependencies, *aligning* roles"

# English
python scripts/preprocessing/convert_conll2009_dep.py \
    --input_dir data/original/conll2009/en \
    --output_dir data/trankit/conll2009/en \
    --language english \
    --strict_language
    
# Catalan
python scripts/preprocessing/convert_conll2009_dep.py \
    --input_dir data/original/conll2009/ca \
    --output_dir data/trankit/conll2009/ca \
    --language catalan \
    --strict_language

# Chinese
python scripts/preprocessing/convert_conll2009_dep.py \
    --input_dir data/original/conll2009/zh \
    --output_dir data/trankit/conll2009/zh \
    --language chinese \
    --strict_language

# Czech
python scripts/preprocessing/convert_conll2009_dep.py \
    --input_dir data/original/conll2009/cs \
    --output_dir data/trankit/conll2009/cs \
    --language czech \
    --strict_language

# German
python scripts/preprocessing/convert_conll2009_dep.py \
    --input_dir data/original/conll2009/de \
    --output_dir data/trankit/conll2009/de \
    --language german \
    --strict_language

# Spanish
python scripts/preprocessing/convert_conll2009_dep.py \
    --input_dir data/original/conll2009/es \
    --output_dir data/trankit/conll2009/es \
    --language spanish \
    --strict_language
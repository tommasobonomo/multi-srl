#!/bin/bash

export PYTHONPATH=${PWD}

python scripts/preprocessing/build_dependency_vocabs.py \
    --input_path data/original/conll2009/ca/CoNLL2009_train.txt \
    --output_path resources/conll_ca_dependency_vocab.json

python scripts/preprocessing/build_dependency_vocabs.py \
    --input_path data/original/conll2009/cs/CoNLL2009_train.txt \
    --output_path resources/conll_cs_dependency_vocab.json

python scripts/preprocessing/build_dependency_vocabs.py \
    --input_path data/original/conll2009/de/CoNLL2009_train.txt \
    --output_path resources/conll_de_dependency_vocab.json

python scripts/preprocessing/build_dependency_vocabs.py \
    --input_path data/original/conll2009/es/CoNLL2009_train.txt \
    --output_path resources/conll_es_dependency_vocab.json

python scripts/preprocessing/build_dependency_vocabs.py \
    --input_path data/original/conll2009/zh/CoNLL2009_train.txt \
    --output_path resources/conll_zh_dependency_vocab.json

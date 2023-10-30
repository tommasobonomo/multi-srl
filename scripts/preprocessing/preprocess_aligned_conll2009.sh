#!/bin/bash

# This script preprocesses the CoNLL-2009 dataset.

export PYTHONPATH=${PWD}

# English
python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/en/CoNLL2009_train.txt \
    --output data/preprocessed_trankit/conll2009/en/CoNLL2009_train.json \
    --add_predicate_pos \
    --keep_lemmas \
    --keep_dep_info \
    --use_original_dep_labels

python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/en/CoNLL2009_dev.txt \
    --output data/preprocessed_trankit/conll2009/en/CoNLL2009_dev.json \
    --add_predicate_pos \
    --keep_lemmas \
    --keep_dep_info \
    --use_original_dep_labels

python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/en/CoNLL2009_test.txt \
    --output data/preprocessed_trankit/conll2009/en/CoNLL2009_test.json \
    --add_predicate_pos \
    --keep_lemmas \
    --keep_dep_info \
    --use_original_dep_labels

python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/en/CoNLL2009_test_ood.txt \
    --output data/preprocessed_trankit/conll2009/en/CoNLL2009_test_ood.json \
    --add_predicate_pos \
    --keep_lemmas \
    --keep_dep_info \
    --use_original_dep_labels

# German
python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/de/CoNLL2009_train.txt \
    --output data/preprocessed_trankit/conll2009/de/CoNLL2009_train.json \
    --keep_lemmas \
    --keep_dep_info

python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/de/CoNLL2009_dev.txt \
    --output data/preprocessed_trankit/conll2009/de/CoNLL2009_dev.json \
    --keep_lemmas \
    --keep_dep_info

python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/de/CoNLL2009_test.txt \
    --output data/preprocessed_trankit/conll2009/de/CoNLL2009_test.json \
    --keep_lemmas \
    --keep_dep_info

python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/de/CoNLL2009_test_ood.txt \
    --output data/preprocessed_trankit/conll2009/de/CoNLL2009_test_ood.json \
    --keep_lemmas \
    --keep_dep_info

# Czech
python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/cs/CoNLL2009_train.txt \
    --output data/preprocessed_trankit/conll2009/cs/CoNLL2009_train.json \
    --keep_lemmas \
    --czech \
    --keep_dep_info

python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/cs/CoNLL2009_dev.txt \
    --output data/preprocessed_trankit/conll2009/cs/CoNLL2009_dev.json \
    --keep_lemmas \
    --czech \
    --keep_dep_info

python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/cs/CoNLL2009_test.txt \
    --output data/preprocessed_trankit/conll2009/cs/CoNLL2009_test.json \
    --keep_lemmas \
    --czech \
    --keep_dep_info

python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/cs/CoNLL2009_test_ood.txt \
    --output data/preprocessed_trankit/conll2009/cs/CoNLL2009_test_ood.json \
    --keep_lemmas \
    --czech \
    --keep_dep_info

# Spanish
python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/es/CoNLL2009_train.txt \
    --output data/preprocessed_trankit/conll2009/es/CoNLL2009_train.json \
    --keep_lemmas \
    --keep_dep_info

python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/es/CoNLL2009_dev.txt \
    --output data/preprocessed_trankit/conll2009/es/CoNLL2009_dev.json \
    --keep_lemmas \
    --keep_dep_info

python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/es/CoNLL2009_test.txt \
    --output data/preprocessed_trankit/conll2009/es/CoNLL2009_test.json \
    --keep_lemmas \
    --keep_dep_info

# Catalan
python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/ca/CoNLL2009_train.txt \
    --output data/preprocessed_trankit/conll2009/ca/CoNLL2009_train.json \
    --keep_lemmas \
    --keep_dep_info

python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/ca/CoNLL2009_dev.txt \
    --output data/preprocessed_trankit/conll2009/ca/CoNLL2009_dev.json \
    --keep_lemmas \
    --keep_dep_info

python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/ca/CoNLL2009_test.txt \
    --output data/preprocessed_trankit/conll2009/ca/CoNLL2009_test.json \
    --keep_lemmas \
    --keep_dep_info

# Chinese
python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/zh/CoNLL2009_train.txt \
    --output data/preprocessed_trankit/conll2009/zh/CoNLL2009_train.json \
    --keep_lemmas \
    --keep_dep_info

python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/zh/CoNLL2009_dev.txt \
    --output data/preprocessed_trankit/conll2009/zh/CoNLL2009_dev.json \
    --keep_lemmas \
    --keep_dep_info

python scripts/preprocessing/preprocess_conll2009.py \
    --input data/trankit/conll2009/zh/CoNLL2009_test.txt \
    --output data/preprocessed_trankit/conll2009/zh/CoNLL2009_test.json \
    --keep_lemmas \
    --keep_dep_info
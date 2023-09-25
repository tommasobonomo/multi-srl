#!/bin/bash

# This script preprocesses the CoNLL-2009 dataset using the original CoNLL-2008 dependency labels.

# Preprocess original CoNLL-2009 dataset
python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/en_ud25/CoNLL2009_train.txt \
    --output data/preprocessed/conll2009/en/CoNLL2009_train.json \
    --add_predicate_pos \
    --keep_lemmas \
    --keep_dep_info \
    --use_original_dep_labels

python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/en_ud25/CoNLL2009_dev.txt \
    --output data/preprocessed/conll2009/en/CoNLL2009_dev.json \
    --add_predicate_pos \
    --keep_lemmas \
    --keep_dep_info \
    --use_original_dep_labels

python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/en_ud25/CoNLL2009_test.txt \
    --output data/preprocessed/conll2009/en/CoNLL2009_test.json \
    --add_predicate_pos \
    --keep_lemmas \
    --keep_dep_info \
    --use_original_dep_labels

python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/en_ud25/CoNLL2009_test-ood.txt \
    --output data/preprocessed/conll2009/en/CoNLL2009_test_ood.json \
    --add_predicate_pos \
    --keep_lemmas \
    --keep_dep_info \
    --use_original_dep_labels


# Build vocabulary
python scripts/preprocessing/compute_vocabulary.py \
    --input data/preprocessed/conll2009/en/CoNLL2009_train.json \
    --output data/preprocessed/conll2009/en/vocabulary.json
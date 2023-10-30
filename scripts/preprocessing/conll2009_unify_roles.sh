#!/bin/bash

echo "Unifying all CoNLL09 annotations into single preprocessed version"

PYTHONPATH=${PWD}

for lang in ca cs de en es zh
do

    python scripts/preprocessing/conll2009_unify_roles.py \
        --conll08_input_path data/preprocessed/conll2009/${lang}/CoNLL2009_train.json \
        --ud25_input_path data/preprocessed_trankit/conll2009/${lang}/CoNLL2009_train.json \
        --output_path data/preprocessed_unified/conll2009/${lang}/CoNLL2009_train.json
    
    python scripts/preprocessing/conll2009_unify_roles.py \
        --conll08_input_path data/preprocessed/conll2009/${lang}/CoNLL2009_dev.json \
        --ud25_input_path data/preprocessed_trankit/conll2009/${lang}/CoNLL2009_dev.json \
        --output_path data/preprocessed_unified/conll2009/${lang}/CoNLL2009_dev.json
    
    python scripts/preprocessing/conll2009_unify_roles.py \
        --conll08_input_path data/preprocessed/conll2009/${lang}/CoNLL2009_test.json \
        --ud25_input_path data/preprocessed_trankit/conll2009/${lang}/CoNLL2009_test.json \
        --output_path data/preprocessed_unified/conll2009/${lang}/CoNLL2009_test.json
done    
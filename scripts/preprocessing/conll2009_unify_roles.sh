#!/bin/bash

python scripts/preprocessing/conll2009_unify_roles.py \
    --conll08_input_path data/preprocessed/conll2009/en/CoNLL2009_train.json \
    --ud25_input_path data/preprocessed_trankit/conll2009/en/CoNLL2009_train.json \
    --output_path data/preprocessed_unified/conll2009/en/CoNLL2009_train.json

python scripts/preprocessing/conll2009_unify_roles.py \
    --conll08_input_path data/preprocessed/conll2009/en/CoNLL2009_dev.json \
    --ud25_input_path data/preprocessed_trankit/conll2009/en/CoNLL2009_dev.json \
    --output_path data/preprocessed_unified/conll2009/en/CoNLL2009_dev.json

python scripts/preprocessing/conll2009_unify_roles.py \
    --conll08_input_path data/preprocessed/conll2009/en/CoNLL2009_test.json \
    --ud25_input_path data/preprocessed_trankit/conll2009/en/CoNLL2009_test.json \
    --output_path data/preprocessed_unified/conll2009/en/CoNLL2009_test.json
    
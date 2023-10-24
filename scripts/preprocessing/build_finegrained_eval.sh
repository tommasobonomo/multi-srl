#!/bin/bash

PYTHONPATH=${PWD}

echo "Building fine-grained evaluation data..."

for lang in ca cs de en es zh
do
    python scripts/preprocessing/build_finegrained_eval.py \
        --og_conll_file data/preprocessed/conll2009/${lang}/CoNLL2009_test.json \
        --ud_aligned_file data/preprocessed_trankit/conll2009/${lang}/CoNLL2009_test.json \
        --output_file data/preprocessed/conll2009/${lang}/CoNLL2009_finegrained_test.json
done

echo "Done."
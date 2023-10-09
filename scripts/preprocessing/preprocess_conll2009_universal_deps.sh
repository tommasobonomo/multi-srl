#!/bin/bash

# This script preprocesses the CoNLL-2009 dataset using Universal Dependencies.

for lang in ca cs de en es zh
do
    echo Preprocessing language ${lang}

    if [ "$lang" = "cs" ]
    then
        czech_option="--czech"  
    else
        czech_option=""
    fi


    # Preprocess original CoNLL-2009 dataset in given language
    python scripts/preprocessing/preprocess_conll2009.py \
        --input data/original/conll2009/${lang}_ud25/CoNLL2009_train.txt \
        --output data/preprocessed/conll2009/${lang}_ud25/CoNLL2009_train.json \
        --add_predicate_pos \
        --keep_lemmas \
        --keep_dep_info \
        ${czech_option}

    python scripts/preprocessing/preprocess_conll2009.py \
        --input data/original/conll2009/${lang}_ud25/CoNLL2009_dev.txt \
        --output data/preprocessed/conll2009/${lang}_ud25/CoNLL2009_dev.json \
        --add_predicate_pos \
        --keep_lemmas \
        --keep_dep_info \
        ${czech_option}

    python scripts/preprocessing/preprocess_conll2009.py \
        --input data/original/conll2009/${lang}_ud25/CoNLL2009_test.txt \
        --output data/preprocessed/conll2009/${lang}_ud25/CoNLL2009_test.json \
        --add_predicate_pos \
        --keep_lemmas \
        --keep_dep_info \
        ${czech_option}

    if [ -e data/original/conll2009/${lang}_ud25/CoNLL2009_test-ood.txt ]
    then
        echo "Found out-of-domain test set for language ${lang}, preprocessing it"

        python scripts/preprocessing/preprocess_conll2009.py \
            --input data/original/conll2009/${lang}_ud25/CoNLL2009_test-ood.txt \
            --output data/preprocessed/conll2009/${lang}_ud25/CoNLL2009_test_ood.json \
            --add_predicate_pos \
            --keep_lemmas \
            --keep_dep_info \
            ${czech_option}
    else
        echo "No out-of-domain test set found for language ${lang}, will not be preprocessed"
    fi


    # Build vocabulary for given language
    python scripts/preprocessing/compute_vocabulary.py \
        --input data/preprocessed/conll2009/${lang}_ud25/CoNLL2009_train.json \
        --output data/preprocessed/conll2009/${lang}_ud25/vocabulary.json
done
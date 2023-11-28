#!/bin/bash

export PYTHONPATH=${PWD}

echo "Training all base models for all languages on CoNLL 2009, with the unified approach"
echo "Total of 5 models will be trained"

for lang in ca cs de en es zh
do
    echo "Training in language ${lang}"

    python scripts/training/trainer.py fit \
        --config configurations/conll2009/base.yaml \
        --config configurations/conll2009/${lang}/unified-xlm-roberta/xlm-roberta-base-ft.yaml

    rm -rf /dev/shm/*

done

#!/bin/bash

export PYTHONPATH=$(pwd)

# Evaluate all checkpoints

for lang in ca cs de en es zh
do 
    echo "Evaluating language ${lang}"
    python scripts/evaluation/evaluate_checkpoint.py \
        --input_model lightning_logs/conll09-${lang}-unified-xlm-roberta-base/version_0/ \
        --language ${lang}
done
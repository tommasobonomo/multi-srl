#!/bin/bash

export PYTHONPATH=$(pwd)

# Evaluate all checkpoints

# Catalan
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-ca-xlm-roberta-base/version_1 \
    --language ca
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-ca-dep-conll08-xlm-roberta-base/version_3 \
    --language ca
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-ca-dep-silver-conll08-xlm-roberta-base/version_2 \
    --language ca
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-ca-dep-universal-xlm-roberta-base/version_8 \
    --language ca

# Czech
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-cs-xlm-roberta-base/version_1 \
    --language cs
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-cs-dep-conll08-xlm-roberta-base/version_2 \
    --language cs
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-cs-dep-silver-conll08-xlm-roberta-base/version_2 \
    --language cs
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-cs-dep-universal-xlm-roberta-base/version_1 \
    --language cs

# German
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-de-xlm-roberta-base/version_1 \
    --language de
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-de-dep-conll08-xlm-roberta-base/version_2 \
    --language de
python scripts/evaluation/evaluate_checkpoint.py \ 
    --input_model lightning_logs/conll09-de-dep-silver-conll08-xlm-roberta-base/version_2 \
    --language de
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-de-dep-universal-xlm-roberta-base/version_1 \
    --language de

# English
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-en-xlm-roberta-base/version_1 \
    --language en
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-en-dep-conll08-xlm-roberta-base/version_0 \
    --language en
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-en-dep-silver-conll08-xlm-roberta-base/version_0 \
    --language en
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-en-dep-universal-xlm-roberta-base/version_63 \
    --language en

# Spanish
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-es-xlm-roberta-base/version_2 \
    --language es
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-es-dep-conll08-xlm-roberta-base/version_3 \
    --language es
python scripts/evaluation/evaluate_checkpoint.py \ 
    --input_model lightning_logs/conll09-es-dep-silver-conll08-xlm-roberta-base/version_3 \
    --language es
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-es-dep-universal-xlm-roberta-base/version_3 \
    --language es

# Chinese
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-zh-xlm-roberta-base/version_2 \
    --language zh
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-zh-dep-conll08-xlm-roberta-base/version_3 \
    --language zh
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-zh-dep-silver-conll08-xlm-roberta-base/version_3 \
    --language zh
python scripts/evaluation/evaluate_checkpoint.py \
    --input_model lightning_logs/conll09-zh-dep-universal-xlm-roberta-base/version_2 \
    --language zh

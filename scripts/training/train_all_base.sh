#!/bin/bash

echo "Training all base models for all languages on CoNLL 2009"
echo "Total of 24 models will be trained"

for lang in ca cs de en es zh
do
    echo "Training in language ${lang}"

    python scripts/training/trainer.py fit \
        --config configurations/conll2009/base.yaml \
        --config configurations/conll2009/${lang}/conll08-dep-xlm-roberta/xlm-roberta-base-ft.yaml

    python scripts/training/trainer.py fit \
        --config configurations/conll2009/base.yaml \
        --config configurations/conll2009/${lang}/conll08-dep-xlm-roberta/xlm-roberta-base-ft-silver.yaml

    python scripts/training/trainer.py fit \
        --config configurations/conll2009/base.yaml \
        --config configurations/conll2009/${lang}/universal-dep-xlm-roberta/xlm-roberta-base-ft.yaml

    python scripts/training/trainer.py fit \
        --config configurations/conll2009/base.yaml \
        --config configurations/conll2009/${lang}/xlm-roberta/xlm-roberta-base-ft.yaml
done

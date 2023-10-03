#!/bin/bash

export PYTHONPATH=$(pwd)

python scripts/training/trainer.py fit \
--config configurations/conll2009/base.yaml \
--config configurations/conll2009/en/unified-roberta/roberta-large-ft.yaml 

python scripts/training/trainer.py fit \
--config configurations/conll2009/base.yaml \
--config configurations/conll2009/en/roberta/roberta-large-ft.yaml 
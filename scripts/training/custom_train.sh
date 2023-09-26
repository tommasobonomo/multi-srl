#!/bin/bash

export PYTHONPATH=/home/tommaso/phd/multi-srl

python scripts/training/trainer.py fit \
--config configurations/conll2009/base.yaml \
--config configurations/conll2009/en/conll08-dep-roberta/roberta-base-ft.yaml

python scripts/training/trainer.py fit \
--config configurations/conll2009/base.yaml \
--config configurations/conll2009/en/universal-dep-roberta/roberta-base-ft.yaml

#!/bin/bash

# PID=2156884
# while ps -p $PID >/dev/null; do sleep 120; done ;

export PYTHONPATH=/home/tommaso/phd/semantic_head/multi-srl

python scripts/training/trainer.py fit \
--config configurations/conll2009/base.yaml \
--config configurations/conll2009/en/roberta/sem-roberta-large-ft.yaml

python scripts/training/trainer.py fit \
--config configurations/conll2009/base.yaml \
--config configurations/conll2009/en/roberta/syn-roberta-large-ft.yaml

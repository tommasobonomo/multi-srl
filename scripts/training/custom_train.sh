#!/bin/bash

export PYTHONPATH=$(pwd)

python scripts/training/trainer.py fit \
--config configurations/conll2009/base.yaml \
--config configurations/conll2009/en/roberta/roberta-base-ft-10pct.yaml

python scripts/training/trainer.py fit \
--config configurations/conll2009/base.yaml \
--config configurations/conll2009/en/conll08-dep-roberta/roberta-base-ft-10pct.yaml

python scripts/training/trainer.py fit \
--config configurations/conll2009/base.yaml \
--config configurations/conll2009/en/conll08-dep-roberta/roberta-base-ft-silver-10pct.yaml

python scripts/training/trainer.py fit \
--config configurations/conll2009/base.yaml \
--config configurations/conll2009/en/universal-dep-roberta/roberta-base-ft-10pct.yaml
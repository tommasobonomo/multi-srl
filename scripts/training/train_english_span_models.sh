#!/bin/bash

python3 srl/train.py --name bert-base-cased-span --language_model bert-base-cased --train_path data/json/en/CoNLL2012_train.json --dev_path data/json/en/CoNLL2012_dev.json
python3 srl/train.py --name bert-large-cased-span --language_model bert-large-cased --train_path data/json/en/CoNLL2012_train.json --dev_path data/json/en/CoNLL2012_dev.json
python3 srl/train.py --name bert-base-cased-span --language_model bert-base-cased --train_path data/json/en/CoNLL2012_train.json --dev_path data/json/en/CoNLL2012_dev.json --language_model_fine_tuning --language_model_learning_rate 1e-5 --language_model_min_learning_rate 1e-9 --batch_size 16 --accumulate_grad_batches 8
python3 srl/train.py --name bert-large-cased-span --language_model bert-large-cased --train_path data/json/en/CoNLL2012_train.json --dev_path data/json/en/CoNLL2012_dev.json --language_model_fine_tuning --language_model_learning_rate 1e-5 --language_model_min_learning_rate 1e-9 --batch_size 4 --accumulate_grad_batches 32
# English
python scripts/preprocessing/preprocess_conll2009.py \
    --input ../data/conll2009/trankit/CoNLL2009_train.txt \
    --output ../data/conll2009/preprocessed_trankit/CoNLL2009_train.json \
    --add_predicate_pos \
    --keep_lemmas

python scripts/preprocessing/preprocess_conll2009.py \
    --input ../data/conll2009/trankit/CoNLL2009_dev.txt \
    --output ../data/conll2009/preprocessed_trankit/CoNLL2009_dev.json \
    --add_predicate_pos \
    --keep_lemmas

python scripts/preprocessing/preprocess_conll2009.py \
    --input ../data/conll2009/trankit/CoNLL2009_test.txt \
    --output ../data/conll2009/preprocessed_trankit/CoNLL2009_test.json \
    --add_predicate_pos \
    --keep_lemmas

python scripts/preprocessing/preprocess_conll2009.py \
    --input ../data/conll2009/trankit/CoNLL2009_test-ood.txt \
    --output ../data/conll2009/preprocessed_trankit/CoNLL2009_test_ood.json \
    --add_predicate_pos \
    --keep_lemmas
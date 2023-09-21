# English
python3 scripts/preprocessing/preprocess_united.py \
    --input ../data/united/en/dependency/train.conllu \
    --output ../data/united/preprocessed/en/dependency/united_train.json

# python3 scripts/preprocessing/preprocess_united.py \
#     --input ../data/united/en/dependency/train.extra.conllu \
#     --output ../data/united/preprocessed/en/dependency/united_train.extra.json

python3 scripts/preprocessing/preprocess_united.py \
    --input ../data/united/en/dependency/dev.conllu \
    --output ../data/united/preprocessed/en/dependency/united_dev.json

python3 scripts/preprocessing/preprocess_united.py \
    --input ../data/united/en/dependency/test.conllu \
    --output ../data/united/preprocessed/en/dependency/united_test.json

# Vocabulary
python scripts/preprocessing/compute_vocabulary.py \
    --input ../data/united/preprocessed/en/dependency/united_train.json \
    --output ../data/united/preprocessed/en/dependency/vocabulary.json
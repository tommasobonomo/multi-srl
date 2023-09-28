import json
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from transformers import AutoTokenizer

from srl.data.srl_dataset import SrlDataset


class SrlDataModule(pl.LightningDataModule):
    def __init__(
        self,
        vocabulary_path: str,
        train_path: str = None,
        dev_path: str = None,
        test_path: str = None,
        pred_path: str = None,
        language_model_name: str = "bert-base-cased",
        batch_size: int = 32,
        num_workers: int = 8,
        padding_label_id: int = -1,
        dependency_labels_vocab_path: Optional[str] = None,
    ):
        super().__init__()
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.pred_path = pred_path
        self.language_model_name = language_model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.padding_label_id = padding_label_id
        self._load_vocabulary(vocabulary_path)
        self._load_tokenizer(language_model_name)
        self._load_dependency_labels_vocab(dependency_labels_vocab_path)

    def setup(self, stage: str = None):
        if stage in ("fit",):
            self.train_data = SrlDataset(
                path_to_data=self.train_path,
                dependency_labels_vocab_path=self.dependency_labels_vocab_path,
            )
            self.dev_data = SrlDataset(
                path_to_data=self.dev_path,
                dependency_labels_vocab_path=self.dependency_labels_vocab_path,
            )

        if stage in ("validate",):
            self.dev_data = SrlDataset(
                path_to_data=self.dev_path,
                dependency_labels_vocab_path=self.dependency_labels_vocab_path,
            )

        if stage in ("test",):
            self.test_data = SrlDataset(
                path_to_data=self.test_path,
                dependency_labels_vocab_path=self.dependency_labels_vocab_path,
            )

        if stage in ("predict",):
            self.pred_data = SrlDataset(
                path_to_data=self.pred_path,
                dependency_labels_vocab_path=self.dependency_labels_vocab_path,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_sentences,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_sentences,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_sentences,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_sentences,
        )

    def encode_sentence(self, sentence):
        """
        Given a sentence object, returns an input sample.
        """
        encoded_sentence = {}
        sentence_length = len(sentence["words"])

        if "predicate_indices" in sentence:
            # List of 0s and 1s to indicate the predicates in the input sentence.
            predicates = [0] * sentence_length
            sense_candidates = []
            for predicate_index in sentence["predicate_indices"]:
                predicates[predicate_index] = 1
                lemma = sentence["lemmas"][predicate_index]
                if lemma in self.lemma2candidates:
                    sense_candidates.append(self.lemma2candidates[lemma])
                else:
                    sense_candidates.append([self.null_sense_id])
            encoded_sentence["predicates"] = predicates
            encoded_sentence["sense_candidates"] = sense_candidates

        elif "annotations" in sentence:
            predicates = [0] * sentence_length
            sense_ids = []
            role_ids = []
            sense_candidates = []

            predicate_indices = sorted(list(sentence["annotations"].keys()))
            for predicate_index in predicate_indices:
                annotation = sentence["annotations"][predicate_index]
                predicates[predicate_index] = 1
                lemma = sentence["lemmas"][predicate_index]
                if lemma in self.lemma2candidates:
                    sense_candidates.append(self.lemma2candidates[lemma])
                else:
                    sense_candidates.append([self.null_sense_id])

                sense = annotation["predicate"]
                sense_id = (
                    self.sense2id[sense]
                    if sense in self.sense2id
                    else self.unknown_sense_id
                )
                sense_ids.append(sense_id)

                _role_ids = [self.padding_label_id] * sentence_length
                for role_index, role in enumerate(annotation["roles"]):
                    _role_ids[role_index] = (
                        self.role2id[role]
                        if role in self.role2id
                        else self.unknown_role_id
                    )

                role_ids.append(_role_ids)

            encoded_sentence["predicates"] = predicates
            encoded_sentence["sense_ids"] = sense_ids
            encoded_sentence["role_ids"] = role_ids
            encoded_sentence["sense_candidates"] = sense_candidates

        return encoded_sentence

    def _load_vocabulary(self, vocabulary_path: str):
        with open(vocabulary_path, "r") as f:
            vocabulary = json.load(f)

        self.sense2id = {k: v for k, v in vocabulary["senses"].items()}
        self.id2sense = {k: v for v, k in self.sense2id.items()}
        self.num_senses = len(self.sense2id)
        self.null_sense_id = self.sense2id["_"]
        self.unknown_sense_id = self.sense2id["<UNK>"]

        self.role2id = {k: v for k, v in vocabulary["roles"].items()}
        self.id2role = {k: v for v, k in self.role2id.items()}
        self.num_roles = len(self.role2id)
        self.null_role_id = self.sense2id["_"]
        self.unknown_role_id = self.role2id["<UNK>"]

        self.lemma2candidates = {
            k: [self.sense2id[s] for s in senses]
            for k, senses in vocabulary["candidates"].items()
        }

    def _load_tokenizer(
        self,
        language_model_name: str,
    ):
        if "roberta" in language_model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                language_model_name, add_prefix_space=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)

    def _load_dependency_labels_vocab(
        self, dependency_labels_vocab_path: Optional[str]
    ):
        self.dependency_labels_vocab_path = dependency_labels_vocab_path
        if dependency_labels_vocab_path is not None:
            with open(dependency_labels_vocab_path, "r") as f:
                dependency_labels = json.load(f)
            self.id2dependency_label = {int(k): v for k, v in dependency_labels.items()}
            self.dependency_label2id = {
                k: v for v, k in self.id2dependency_label.items()
            }
            self.num_dependency_labels = len(self.dependency_label2id)
        else:
            self.id2dependency_label, self.dependency_label2id = {}, {}
            self.num_dependency_labels = 0

    def _collate_sentences(self, sentences):
        batched_inputs = {
            "sentence_ids": [],
            "sentence_lengths": [],
            "lm_inputs": None,
            "subword_indices": [],
        }

        batched_targets = {}

        words = []
        dep_heads = []
        dep_labels = []
        max_sentence_length = 0

        for sentence in sentences:
            words.append(sentence["words"])
            dep_heads.append(sentence["dep_heads"])
            dep_labels.append(sentence["dep_labels"])
            sentence_length = len(sentence["words"])
            max_sentence_length = max(sentence_length, max_sentence_length)

            batched_inputs["sentence_ids"].append(sentence["sentence_id"])
            batched_inputs["sentence_lengths"].append(sentence_length)

            encoded_sentence = self.encode_sentence(sentence)

            if "predicates" in encoded_sentence:
                if "predicates" not in batched_targets:
                    batched_targets["predicates"] = []
                    batched_inputs["sense_candidates"] = []
                batched_targets["predicates"].append(
                    torch.as_tensor(encoded_sentence["predicates"])
                )
                batched_inputs["sense_candidates"].extend(
                    [
                        torch.as_tensor(c, dtype=torch.int64)
                        for c in encoded_sentence["sense_candidates"]
                    ]
                )

            if "sense_ids" in encoded_sentence:
                if "sense_ids" not in batched_targets:
                    batched_targets["sense_ids"] = []
                    batched_targets["role_ids"] = []
                batched_targets["sense_ids"].extend(encoded_sentence["sense_ids"])
                batched_targets["role_ids"].extend(
                    torch.as_tensor(encoded_sentence["role_ids"])
                )

        if "deberta" not in self.language_model_name:
            lm_inputs = self.tokenizer(
                words,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        else:
            lm_inputs = self.tokenizer(
                words, is_split_into_words=True, padding=True, return_tensors="pt"
            )
        batched_inputs["lm_inputs"] = lm_inputs

        # Build the deptree inputs, only if the dependency labels vocab is provided.
        if self.num_dependency_labels > 0:
            sentence_dependency_trees = []
            for sent_dep_heads, sent_dep_labels in zip(dep_heads, dep_labels):
                raw_edge_list = []
                raw_edge_attribute = []
                for i, head in enumerate(sent_dep_heads):
                    if head != -1:
                        raw_edge_list.append([i, head])
                        raw_edge_list.append([head, i])
                        raw_edge_attribute.append(sent_dep_labels[i])
                        raw_edge_attribute.append(sent_dep_labels[i])
                edge_list = torch.as_tensor(raw_edge_list, dtype=torch.long)
                edge_attribute = torch.as_tensor(
                    [self.dependency_label2id[label] for label in raw_edge_attribute],
                    dtype=torch.long,
                ).unsqueeze(1)
                if edge_list.ndim == 1:
                    edge_list.resize_(0, 2)
                    edge_attribute.resize_(0, 1)
                edge_index = edge_list.T.contiguous()

                sentence_dependency_trees.append(
                    Data(
                        edge_index=edge_index,
                        edge_attr=edge_attribute,
                        num_nodes=len(sent_dep_heads),
                    )
                )
            batched_inputs["dependency_trees"] = Batch.from_data_list(
                sentence_dependency_trees
            )

        current_predicate = 0
        for sentence_index in range(len(sentences)):
            subword_indices = [
                id + 1 if id is not None else 0
                for id in lm_inputs.word_ids(batch_index=sentence_index)
            ]
            true_sentence_length = max(subword_indices)
            batched_inputs["sentence_lengths"][sentence_index] = true_sentence_length
            last_token_index = subword_indices[1:].index(0) + 1
            subword_indices[last_token_index] = true_sentence_length + 1

            for token_index in range(last_token_index, len(subword_indices)):
                subword_indices[token_index] = min(
                    max_sentence_length + 1, true_sentence_length + 2
                )

            batched_inputs["subword_indices"].append(subword_indices)

            for predicate_index, predicate in enumerate(
                batched_targets["predicates"][sentence_index]
            ):
                if predicate == 1:
                    if predicate_index >= true_sentence_length:
                        if "sense_ids" in batched_targets:
                            batched_targets["sense_ids"] = (
                                batched_targets["sense_ids"][:current_predicate]
                                + batched_targets["sense_ids"][current_predicate + 1 :]
                            )
                            batched_targets["role_ids"] = (
                                batched_targets["role_ids"][:current_predicate]
                                + batched_targets["role_ids"][current_predicate + 1 :]
                            )
                        batched_inputs["sense_candidates"] = (
                            batched_inputs["sense_candidates"][:current_predicate]
                            + batched_inputs["sense_candidates"][
                                current_predicate + 1 :
                            ]
                        )
                    else:
                        current_predicate += 1

        new_max_sentence_length = max(batched_inputs["sentence_lengths"]) + 1
        batched_inputs["subword_indices"] = torch.as_tensor(
            batched_inputs["subword_indices"]
        )
        batched_inputs["sentence_lengths"] = torch.as_tensor(
            batched_inputs["sentence_lengths"]
        )

        if "sense_ids" in batched_targets:
            batched_targets["sense_ids"] = torch.as_tensor(batched_targets["sense_ids"])

        if "predicates" in batched_targets:
            batched_targets["predicates"] = pad_sequence(
                batched_targets["predicates"],
                batch_first=True,
                padding_value=self.padding_label_id,
            )[:, :new_max_sentence_length]
            batched_inputs["predicates"] = batched_targets["predicates"]
            batched_inputs["sense_candidates"] = pad_sequence(
                batched_inputs["sense_candidates"],
                batch_first=True,
                padding_value=self.null_sense_id,
            )

        if "role_ids" in batched_targets:
            batched_targets["role_ids"] = pad_sequence(
                batched_targets["role_ids"],
                batch_first=True,
                padding_value=self.padding_label_id,
            )[:, :new_max_sentence_length]

        return batched_inputs, batched_targets

    @staticmethod
    def _pad_bidimensional_sequences(sequences, sequence_length, padding_value=0):
        padded_sequences = torch.full(
            (len(sequences), sequence_length, sequence_length),
            padding_value,
            dtype=torch.long,
        )
        for i, sequence in enumerate(sequences):
            for j, subsequence in enumerate(sequence):
                padded_sequences[i][j][: len(subsequence)] = torch.as_tensor(
                    subsequence, dtype=torch.long
                )
        return padded_sequences


# if __name__ == "__main__":
#     dm = SrlDataModule(
#         vocabulary_path="data/preprocessed/conll2009/en_ud25/vocabulary.json",
#         train_path="data/preprocessed/conll2009/en_ud25/CoNLL2009_train.json",
#         dev_path="data/preprocessed/conll2009/en_ud25/CoNLL2009_dev.json",
#         dependency_labels_vocab_path="resources/universal_dependency_vocab.json",
#         num_workers=0,
#     )
#     dm.setup("validate")
#     for batch in dm.val_dataloader():
#         pass

import json
from typing import List

from torch.utils.data import Dataset


class SrlDataset(Dataset):
    def __init__(
        self,
        path_to_data: str = None,
        sentences: List = None,
        dependency_labels_vocab_path: str = None,
    ):
        super().__init__()

        if dependency_labels_vocab_path is not None:
            dependency_type = (
                "conll" if "conll" in dependency_labels_vocab_path else "universal"
            )
        else:
            dependency_type = None

        if path_to_data is not None:
            self.sentences = SrlDataset.load_sentences_from_file(
                path_to_data, dependency_type
            )
        else:
            self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

    @staticmethod
    def load_sentences_from_file(path: str, dependency_type: str = None) -> List[dict]:
        sentences = []

        with open(path) as json_file:
            for i, sentence in json.load(json_file).items():
                if (
                    "predicate_indices" not in sentence
                    and "annotations" not in sentence
                ):
                    continue

                if (
                    "predicate_indices" in sentence
                    and not sentence["predicate_indices"]
                ):
                    continue

                if "annotations" in sentence and not sentence["annotations"]:
                    continue

                if len(sentence["words"]) > 128:
                    continue

                sample = {
                    "sentence_id": i,
                    "words": SrlDataset.process_words(sentence["words"]),
                    "lemmas": sentence["lemmas"],
                }
                if "dep_heads" in sentence and "dep_labels" in sentence:
                    sample["dep_heads"] = [int(head) for head in sentence["dep_heads"]]
                    sample["dep_labels"] = [
                        SrlDataset.clean_deprel(deprel, dependency_type)
                        for deprel in sentence["dep_labels"]
                    ]

                if "annotations" in sentence:
                    sample["annotations"] = {
                        int(idx): a for idx, a in sentence["annotations"].items()
                    }
                elif "predicate_indices" in sentence:
                    sample["predicate_indices"] = sentence["predicate_indices"]

                sentences.append(sample)

        return sentences

    @staticmethod
    def load_sentences(data: list, dependency_type: str = None) -> List[dict]:
        sentences = []

        for i, sentence in enumerate(data):
            sentence_id = i
            words = SrlDataset.process_words([w["text"] for w in sentence["tokens"]])
            lemmas = [w["lemma"] for w in sentence["tokens"]]
            pos_tags = [w["upos"] for w in sentence["tokens"]]

            sample = {
                "sentence_id": sentence_id,
                "words": words,
                "lemmas": lemmas,
                "dep_heads": [int(w["head"]) for w in sentence["tokens"]],
                "dep_labels": [
                    SrlDataset.clean_deprel(w["deprel"], dependency_type)
                    for w in sentence["tokens"]
                ],
            }
            sentences.append(sample)

        return sentences

    @staticmethod
    def process_words(words: List[str]) -> List[str]:
        processed_words = []
        for word in words:
            processed_word = SrlDataset.clean_word(word)
            processed_words.append(processed_word)
        return processed_words

    @staticmethod
    def clean_word(word: str) -> str:
        if word == "n't":
            return "not"
        if word == "wo":
            return "will"
        if word == "'ll":
            return "will"
        if word == "'m":
            return "am"
        if word == "``":
            return '"'
        if word == "''":
            return '"'
        if word == "/.":
            return "."
        if word == "/-":
            return "..."
        if word == "-LRB-" or word == "-LSB-" or word == "-LCB-":
            return "("
        if word == "-RRB-" or word == "-RSB-" or word == "-RCB-":
            return ")"

        if "\\/" in word:
            word = word.replace("\\/", "/")

        return word

    @staticmethod
    def clean_deprel(relation: str, dependency_type: str = None) -> str:
        # keep only the first part of the relation
        relation = relation.lower()
        if dependency_type == "conll":
            if "-" in relation:
                # It's a non-atomic relation
                atomic_part, second_part = relation.split("-", 1)
                if "prd" in relation:
                    # It's a predicate, just return the predicate relation
                    return "prd"
                elif "oprd" in relation:
                    # It's an OPRD, just return the OPRD relation
                    return "oprd"
                elif atomic_part == "gap":
                    # Return the non-gap part of the relation
                    return second_part
                elif second_part == "gap":
                    # Return the non-gap part of the relation
                    return atomic_part
                else:
                    # It's a compound adverbial relation, return the first one
                    return atomic_part
            else:
                # It's an atomic relation, just return it
                return relation
        elif dependency_type == "universal":
            if ":" in relation:
                # The relation contains a sub-type, we discard it and return the main type
                return relation.split(":", 1)[0]
            else:
                # It's a normal type relation, just return it
                return relation
        else:
            # TODO: This will probably break stuff in demo and predict but will have to be fixed later
            return relation

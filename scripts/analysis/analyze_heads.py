import argparse
import json
import os

import requests
from tqdm import tqdm


def read_data(input_path: str):
    with open(input_path, "r") as f:
        data = json.load(f)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)

    args = parser.parse_args()

    data = read_data(args.input)
    num_total_roles = 0
    num_different_roles = 0
    num_total_predicates = 0
    num_predicates_with_different_roles = 0
    original_content_words = 0
    modified_content_words = 0
    all_original_synsets = set()
    all_modified_synsets = set()
    all_original_tokens = set()
    all_modified_tokens = set()

    for sentence_id, sentence_data in tqdm(data.items()):
        words = sentence_data["words"]
        sentence_words = " ".join(words)
        annotations = sentence_data["annotations"]
        print("=" * 80)
        print("Sentence ID: {}".format(sentence_id))
        print(sentence_words)
        print()

        # Get language from environment variable
        lang = os.environ.get("LANGUAGES")

        # Do a POST request to the server.
        # curl -X 'POST' \
        #     'http://127.0.0.1/api/model' \
        #     -H 'accept: application/json' \
        #     -H 'Content-Type: application/json' \
        #     -d '[
        #         {"text":"The quick brown fox jumps over the lazy dog.", "lang":"EN"},
        #     ]'
        response = requests.post(
            "http://localhost/api/model",
            headers={"Content-Type": "application/json"},
            json=[{"text": sentence_words, "lang": lang}],
        )

        # Get the response data as a python object.
        data = response.json()
        tokens = data[0]["tokens"]
        synsets = []
        for token in tokens:
            synsets.append(token["bnSynsetId"])

        for predicate_idx, predicate_annotations in annotations.items():
            num_total_predicates += 1
            original_roles = predicate_annotations["roles"]
            modified_roles = predicate_annotations["modified_roles"]
            assert len(original_roles) == len(
                modified_roles
            ), "Number of roles should be the same for original and modified sentences. Sentence ID: {}".format(
                sentence_id
            )

            original_arguments = {}
            original_synsets = {}
            modified_arguments = {}
            modified_synsets = {}

            for role_id, (original_role, modified_role) in enumerate(
                zip(original_roles, modified_roles)
            ):
                is_content_word = synsets[role_id] != "O"
                if original_role != "_":
                    num_total_roles += 1
                if original_role != modified_role:
                    if original_role != "_":
                        original_arguments[original_role] = words[role_id]
                        original_synsets[original_role] = synsets[role_id]
                        all_original_tokens.add(words[role_id].lower())
                        if is_content_word:
                            original_content_words += 1
                            all_original_synsets.add(synsets[role_id])
                    if modified_role != "_":
                        modified_arguments[modified_role] = words[role_id]
                        modified_synsets[modified_role] = synsets[role_id]
                        all_modified_tokens.add(words[role_id].lower())
                        if is_content_word:
                            modified_content_words += 1
                            all_modified_synsets.add(synsets[role_id])

            different_roles = list(
                set(original_arguments.keys()).union(set(modified_arguments.keys()))
            )
            different_roles = sorted(different_roles)
            if len(different_roles) > 0:
                num_different_roles += len(different_roles)
                print(
                    "Predicate ID: {}, Predicate word: {}".format(
                        predicate_idx, words[int(predicate_idx)]
                    )
                )

                for role in different_roles:
                    print(
                        "  {}: {} [{}] -> {} [{}]".format(
                            role,
                            original_arguments.get(role, "_"),
                            original_synsets.get(role, "_"),
                            modified_arguments.get(role, "_"),
                            modified_synsets.get(role, "_"),
                        )
                    )

                print()
                num_predicates_with_different_roles += 1

    print(
        "Number of modified roles: {}/{} [{:0.2f}]".format(
            num_different_roles,
            num_total_roles,
            100 * num_different_roles / num_total_roles,
        )
    )
    print(
        "Number of predicates with modified roles: {}/{} [{:0.2f}]".format(
            num_predicates_with_different_roles,
            num_total_predicates,
            100 * num_predicates_with_different_roles / num_total_predicates,
        )
    )
    print("Number of original content words: {}".format(original_content_words))
    print("Number of modified content words: {}".format(modified_content_words))
    print("Number of original tokens: {}".format(len(all_original_tokens)))
    print("Number of modified tokens: {}".format(len(all_modified_tokens)))
    print("Number of original synsets: {}".format(len(all_original_synsets)))
    print("Number of modified synsets: {}".format(len(all_modified_synsets)))

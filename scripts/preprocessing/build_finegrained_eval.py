import json
from pathlib import Path
from typing import Dict

from tap import Tap
from wasabi import msg


class ScriptArgs(Tap):

    """
    Script that builds a fine-grained evaluation file from a preprocessed CoNLL09 `.json` file.
    Fine-grained means we build an evaluation set from all the roles that have been moved to be aligned with UD trees.
    """

    og_conll_file: Path  # Path to a preprocessed CoNLL09 `.json` file, usually the original test CoNLL09 file
    ud_aligned_file: Path  # Path to a preprocessed CoNLL09 `.json` file aligned with UD, usually the trankit test file
    output_file: Path  # Path to the output file that should be written

    @staticmethod
    def check_file(file: Path):
        if not (file.exists() and file.suffix == ".json"):
            raise RuntimeError(f"Invalid input file: {file}")

    def process_args(self) -> None:
        self.check_file(self.og_conll_file)
        self.check_file(self.ud_aligned_file)

        # Ensure output file parent directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Warn if output file already exists
        if self.output_file.exists():
            msg.warn(f"Output file {self.output_file} already exists. Overwriting.")


def run(args: ScriptArgs) -> None:
    msg.divider("Building fine-grained evaluation file")

    # Load the original CoNLL09 file
    with open(args.og_conll_file, "r") as f:
        og_conll = json.load(f)

    # Load the UD aligned CoNLL09 file
    with open(args.ud_aligned_file, "r") as f:
        ud_aligned = json.load(f)

    msg.good(
        f"Loaded {len(og_conll)} sentences from {args.og_conll_file}, {len(ud_aligned)} sentences from {args.ud_aligned_file}"
    )

    # Loop over all sentences in the original CoNLL09 file
    eval_set = {}
    num_aligned_roles = 0
    for key, sent in og_conll.items():
        annotations = sent["annotations"]
        if len(annotations) == 0:
            continue
        else:
            if key in ud_aligned:
                ud_annotations = ud_aligned[key]["annotations"]
            else:
                ud_annotations = {}

            # Loop over all predicate annotations and add them if at least one role has been moved to be aligned with UD
            new_annotations = {}
            for pred_idx, annotation in annotations.items():
                if pred_idx not in ud_annotations:
                    continue
                else:
                    og_roles = annotation["roles"]
                    ud_roles = ud_annotations[pred_idx]["roles"]
                    new_roles = []
                    for role, ud_role in zip(og_roles, ud_roles):
                        if role == ud_role == "_":
                            new_roles.append("_")
                        else:
                            if role == ud_role:
                                new_roles.append("_")
                            else:
                                new_roles.append(role)
                                num_aligned_roles += 1

                    if any(role != "_" for role in new_roles):
                        new_annotations[pred_idx] = {
                            "predicate": annotation["predicate"],
                            "roles": new_roles,
                        }

            if len(new_annotations) > 0:
                sent["annotations"] = new_annotations
                eval_set[key] = sent

    msg.good(
        f"Built evaluation set with {len(eval_set)} sentences and {num_aligned_roles} aligned roles"
    )

    # Write the evaluation set to file
    with open(args.output_file, "w") as f:
        json.dump(eval_set, f, indent=2)

    msg.good(f"Wrote evaluation set to {args.output_file}")


if __name__ == "__main__":
    args = ScriptArgs().parse_args()
    run(args)

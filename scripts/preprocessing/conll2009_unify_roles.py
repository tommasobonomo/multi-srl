import json
from pathlib import Path

from tap import Tap
from wasabi import msg


class ScriptArgs(Tap):
    """
    This script takes two preprocessed .json files with CoNLL2008 and UD2.5 dependency data for roles
    and adds the UD2.5 dependency data to the CoNLL2008 dependency data.
    It outputs the unified preprocessed dependency data in a .json file with the extra field `modified_roles`.
    """

    conll08_input_path: Path  # Path to the input file with CoNLL2008 dependency data for roles
    ud25_input_path: Path  # Path to the input file with UD2.5 dependency data for roles
    output_path: Path  # Path to the output file with both CoNLL2008 and UD2.5 dependency data for roles

    def process_args(self) -> None:
        if not (
            self.conll08_input_path.exists()
            and self.conll08_input_path.is_file()
            and self.conll08_input_path.suffix == ".json"
        ):
            raise ValueError(
                "The input file with CoNLL2008 dependency data for roles is not specified."
            )
        if not (
            self.ud25_input_path.exists()
            and self.ud25_input_path.is_file()
            and self.ud25_input_path.suffix == ".json"
        ):
            raise ValueError(
                "The input file with UD2.5 dependency data for roles is not specified."
            )

        if not self.output_path.suffix == ".json":
            raise ValueError(
                "The output file with both CoNLL2008 and UD2.5 dependency data for roles must be a .json file."
            )

        if not (self.output_path.parent.exists() and self.output_path.parent.is_dir()):
            self.output_path.parent.mkdir(parents=True, exist_ok=True)


def run(args: ScriptArgs) -> None:
    msg.divider("Unifying CoNLL2008 and UD2.5 dependency data in preprocessed data")

    with open(args.conll08_input_path, mode="r", encoding="utf-8") as conll08_file:
        conll08_data = json.load(conll08_file)
    with open(args.ud25_input_path, mode="r", encoding="utf-8") as ud25_file:
        ud25_data = json.load(ud25_file)

    msg.good("Loaded CoNLL2008 and UD2.5 dependency data for roles")
    msg.info("CoNLL2008 path:", args.conll08_input_path.as_posix())
    msg.info("UD2.5 path:", args.ud25_input_path.as_posix())

    assert set(conll08_data.keys()) == set(
        ud25_data.keys()
    ), "The sentence_ids in the two input files must be the same."

    with msg.loading("Unifying CoNLL2008 and UD2.5 dependency data for roles..."):
        unified_data = {}
        for sentence_id in conll08_data.keys():
            conll08_sentence = conll08_data[sentence_id]
            ud25_sentence = ud25_data[sentence_id]
            unified_annotations = {
                sent_idx: {
                    "predicate": conll08_annotation["predicate"],
                    "roles": conll08_annotation["roles"],
                    "modified_roles": ud25_sentence["annotations"][sent_idx]["roles"],
                }
                for sent_idx, conll08_annotation in conll08_sentence[
                    "annotations"
                ].items()
            }
            unified_data[sentence_id] = {
                "annotations": unified_annotations,
                "lemmas": conll08_sentence["lemmas"],
                "words": conll08_sentence["words"],
            }

    msg.good("Unified CoNLL2008 and UD2.5 dependency data for roles")

    with open(args.output_path, mode="w+", encoding="utf-8") as output_file:
        json.dump(unified_data, output_file, indent=2)

    msg.good(
        f"Saved unified CoNLL2008 and UD2.5 dependency data for roles to {args.output_path.as_posix()}"
    )


if __name__ == "__main__":
    run(ScriptArgs().parse_args())

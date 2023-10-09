from pathlib import Path
from typing import Literal

import spacy
from tap import Tap
from tqdm import tqdm
from trankit import Pipeline
from wasabi import msg

import src.utils.conll
import src.utils.united
from src.semantic_heads import english_to_universal_semantic_head


class ScriptArgs(Tap):
    input_dir: Path  # Path to directory of `.collu` files that should be converted
    output_dir: Path  # Path to directory where output should be saved
    pipeline_type: Literal["spacy", "trankit"] = "trankit"  # Type of pipeline to use
    language: str = "english"  # Language of input files
    strict_language: bool = (
        False  # If True, will fail if `language` is not supported by trankit
    )
    input_type: Literal["conll09", "united"] = "conll09"  # Type of input files
    only_dependencies: bool = False  # Only convert dependencies, i.e. add UD2.5 predictions with pipeline for PDEPREL and PHEAD

    def process_args(self) -> None:
        if not (self.input_dir.exists() and self.input_dir.is_dir()):
            raise RuntimeError(
                f"Path {self.input_dir.as_posix()} does not exist or is not a directory"
            )

        if not (self.output_dir.exists() and self.output_dir.is_dir()):
            self.output_dir.mkdir(parents=True, exist_ok=True)


def run(args: ScriptArgs):
    msg.divider(f"Tagging CoNLL09-{args.language}")

    if args.input_type == "conll09":
        parse_raw_sentence = src.utils.conll.parse_raw_sentence
        write_as_conll = src.utils.conll.write_as_conll
        msg.info("Using CoNLL09 input files")
    elif args.input_type == "united":
        parse_raw_sentence = src.utils.united.parse_raw_sentence
        write_as_conll = src.utils.united.write_as_conll
        msg.info("Using UniteD input files")
    else:
        raise NotImplementedError(f"Input type {args.input_type} is not implemented")

    if args.pipeline_type == "spacy":
        pipeline_name = "en_udv25_englishewt_trf"
        nlp = spacy.load(pipeline_name, disable=["experimental_char_ner_tokenizer"])
        msg.good(f"Loaded spaCy pipeline for UD 2.5, `{pipeline_name}`.")
    elif args.pipeline_type == "trankit":
        try:
            nlp = Pipeline(args.language, embedding="xlm-roberta-large")
        except AssertionError as e:
            msg.fail(f"Could not load trankit pipeline: {e}. Falling back to english")
            nlp = Pipeline("english", embedding="xlm-roberta-large")
        msg.good("Loaded trankit pipeline.")
    else:
        nlp = None

    input_files = [
        file
        for file in args.input_dir.iterdir()
        if file.is_file() and file.suffix != ".json"
    ]
    for file in tqdm(input_files, desc=f"Parsing files in {args.input_dir.as_posix()}"):
        text = file.read_text()

        output_file = args.output_dir / file.name
        output_handler = open(output_file, "w+")

        sentences = text.split("\n\n")

        for sentence in tqdm(sentences, desc=f"Parsing {file.name}", leave=False):
            if sentence == "":
                continue

            # Parse CoNLL09 sentence
            parsed_sentence = parse_raw_sentence(sentence, nlp)

            # Recover frames, then compute semantic heads if not only dependencies
            frames: list[dict] = []
            for token in parsed_sentence:
                if token.frame_name is not None:
                    frames.append(
                        {
                            "frame_idx": token.idx,
                            "frame_name": token.frame_name,
                            "frame_roles": token.frame_roles,
                        }
                    )
            if not args.only_dependencies:
                for frame in frames:
                    for role in frame["frame_roles"]:
                        semantic_head_idx = english_to_universal_semantic_head(
                            role.role_idx,
                            parsed_sentence,
                        )
                        role.semantic_head_idx = semantic_head_idx

            # Save to file
            raw_tokens = [
                raw_token
                for raw_token in sentence.split("\n")
                if not raw_token.startswith("#")
            ]
            for token, string_token in zip(parsed_sentence, raw_tokens):
                assert (
                    not token.sem_dep.is_any_none()
                ), f"Token {token} is missing semantic dependency information"

                raw_token = string_token.split()

                correct_role_tokens = []
                for frame in frames:
                    if args.input_type == "united":
                        # UniteD marks predicates with B-V in the appropriate frame column
                        if frame["frame_idx"] == token.idx:
                            correct_role_tokens.append("B-V")
                            continue

                    # Check first semantic idx of roles
                    semantic_role_token = next(
                        (
                            role
                            for role in frame["frame_roles"]
                            if role.semantic_head_idx is not None
                            and role.semantic_head_idx == token.idx
                        ),
                        None,
                    )
                    if semantic_role_token is not None:
                        correct_role_tokens.append(semantic_role_token.role_name)
                    else:
                        # Fallback to syntactic idx
                        syntactic_role_token = next(
                            (
                                role
                                for role in frame["frame_roles"]
                                if role.role_idx == token.idx
                            ),
                            None,
                        )
                        if (
                            syntactic_role_token is not None
                            and syntactic_role_token.semantic_head_idx is None
                        ):
                            # Add syntactic token only if it has no semantic head
                            correct_role_tokens.append(syntactic_role_token.role_name)
                        else:
                            correct_role_tokens.append("_")

                # Write exact same string as input, but add modified values
                out_text = write_as_conll(token, raw_token, correct_role_tokens)

                output_handler.write(out_text)

            output_handler.write("\n")

        output_handler.close()


if __name__ == "__main__":
    args = ScriptArgs().parse_args()
    run(args)

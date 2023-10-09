from pathlib import Path
from typing import List, Union

import spacy
from spacy.language import Language
from spacy.tokens import Doc
from trankit import Pipeline

from src.token_types import DepRelation, FrameRole, Sentence, TokenFeatures


def iob_parsing(roles: List[str]) -> List[dict]:
    out_roles: List[dict] = []
    in_span = False
    accumulator = {}
    for idx, role in enumerate(roles):
        if role == "_" or role == "B-V":
            if in_span:
                # Close span
                accumulator["end_idx"] = idx
                out_roles.append(accumulator)
                accumulator = {}
            in_span = False
        elif role.startswith("B"):
            in_span = True
            accumulator = {"role_name": role[2:], "start_idx": idx}
        elif role.startswith("I"):
            continue
    return out_roles


def parse_role_spans(raw_sentence: str) -> dict:
    raw_tokens = [
        raw_token.split("\t")
        for raw_token in raw_sentence.split("\n")
        if not raw_token.startswith("#")
    ]
    num_frames = len(raw_tokens[0]) - 4
    frame_and_roles = [raw_token[num_frames:] for raw_token in raw_tokens]
    raw_frame_names, *frame_roles = zip(*frame_and_roles)
    frame_names = [frame_name for frame_name in raw_frame_names if frame_name != "_"]

    parsed_frames_and_roles = {}
    for frame_name, raw_frame_roles in zip(frame_names, frame_roles):
        parsed_frames_and_roles[frame_name] = iob_parsing(list(raw_frame_roles))

    return parsed_frames_and_roles


def parse_raw_sentence(
    raw_sentence: str, pipeline: Union[Language, Pipeline, None]
) -> Sentence:
    """
    Parse raw UniteD sentence to a list of TokenFeatures.
    If a spaCy or trankit pipeline is passed, will be used for UD2.5 DEP-tree parsing.
    Will instantiate a spaCy pipeline for SD dep-tree parsing to build the syntactic tree.
    """
    if pipeline is None:
        raise RuntimeError(
            "No pipeline provided, and UniteD needs a pipeline to be provided."
        )

    parsed_sentence: List[TokenFeatures] = []
    raw_tokens = [
        raw_token
        for raw_token in raw_sentence.split("\n")
        if not raw_token.startswith("#")
    ]
    text_tokens = [raw_token.split()[1] for raw_token in raw_tokens]
    spacy_pipeline, trankit_pipeline = None, None
    if isinstance(pipeline, Language):
        spacy_pipeline = pipeline
        doc = spacy_pipeline(Doc(spacy_pipeline.vocab, text_tokens))
    elif isinstance(pipeline, Pipeline):
        trankit_pipeline = pipeline
        doc = trankit_pipeline(text_tokens, is_sent=True)

    synt_pipeline = spacy.load("en_core_web_trf", disable=["ner"])
    synt_doc = synt_pipeline(Doc(synt_pipeline.vocab, text_tokens))

    for idx, raw_token in enumerate(raw_tokens):
        token = raw_token.split()

        if spacy_pipeline is not None:
            spacy_token = doc[idx]
            sem_dep = spacy_token.dep_
            sem_dep_head_idx = spacy_token.head.i if spacy_token.head.i != idx else -1
            sem_dep_child_idx = [tok.i for tok in spacy_token.children]
        elif trankit_pipeline is not None:
            trakit_token = doc["tokens"][idx]  # type: ignore
            sem_dep = trakit_token["deprel"]
            sem_dep_head_idx = trakit_token["head"] - 1
            sem_dep_child_idx = [
                tok["id"] - 1 for tok in doc["tokens"] if tok["head"] - 1 == idx  # type: ignore
            ]

        # Use the synt_doc for syntactic dependencies
        synt_token = synt_doc[idx]
        synt_dep = synt_token.dep_
        synt_dep_head_idx = synt_token.head.i if synt_token.head.i != idx else -1
        synt_dep_child_idx = [tok.i for tok in synt_token.children]

        parsed_sentence.append(
            TokenFeatures(
                idx=int(token[0]),
                text=token[1],
                lemma=token[2],
                pos="_",
                feat="_",
                synt_dep=DepRelation(synt_dep, synt_dep_head_idx, synt_dep_child_idx),
                sem_dep=DepRelation(sem_dep, sem_dep_head_idx, sem_dep_child_idx),
                frame_name=None,
                frame_roles=[],
            )
        )

    # Second pass for SRL parsing
    num_srl_frames = len(raw_tokens[0].split()) - 4
    frames: List[dict] = [
        {"frame_idx": None, "frame_name": None, "frame_roles": []}
        for _ in range(num_srl_frames)
    ]
    parsed_frames: set[int] = set()
    for i in range(num_srl_frames):
        frame = frames[i]
        for raw_token in raw_tokens:
            token = raw_token.split()
            # Check if token is predicate
            if (
                token[3] != "_"
                and frame["frame_idx"] is None
                and (int(token[0])) not in parsed_frames
            ):
                frame["frame_idx"] = int(token[0])
                frame["frame_name"] = token[3]
                parsed_frames.add(int(token[0]))

            # Check if token is role
            if len(token) > 4 + i and token[4 + i] != "_" and token[4 + i] != "B-V":
                frame["frame_roles"].append(
                    FrameRole(
                        role_idx=int(token[0]),
                        role_name=token[4 + i],
                        semantic_head_idx=None,
                    )
                )

        parsed_token = parsed_sentence[frame["frame_idx"]]
        parsed_token.frame_name = frame["frame_name"]
        parsed_token.frame_roles = frame["frame_roles"]

    return parsed_sentence


def write_as_conll(
    token: TokenFeatures, raw_token: List[str], correct_role_tokens: List[str]
):
    return (
        f"{raw_token[0]}\t"  # ID
        + f"{raw_token[1]}\t"  # FORM
        + f"{raw_token[2]}\t"  # LEMMA
        + f"{token.frame_name}\t"  # PRED
        + "\t".join(correct_role_tokens)  # APREDS
        + "\n"
    )


def write_predictions_to_conll(predictions: dict, output_path: Path, gold_path: Path):
    with open(gold_path) as f_gold, open(output_path, "w") as f_pred:
        sentence_id = 0
        sentence_parts = []

        for line in f_gold:
            line = line.strip()

            if line.startswith("#"):
                # Write comments as is
                f_pred.write(line + "\n")
                continue

            if not line:
                sentence_roles = []

                if sentence_id in predictions:
                    predicate_indices = sorted(predictions[sentence_id].keys())

                    for predicate_idx in predicate_indices:
                        predicate_predictions = predictions[sentence_id][predicate_idx]
                        sentence_parts[predicate_idx][3] = predicate_predictions[
                            "predicate"
                        ]
                        predicate_predictions["roles"][predicate_idx] = "B-V"
                        sentence_roles.append(predicate_predictions["roles"])

                if sentence_roles:
                    sentence_roles = list(zip(*sentence_roles))
                    for line, line_roles in zip(sentence_parts, sentence_roles):
                        line.extend(line_roles)

                for line in sentence_parts:
                    f_pred.write("\t".join(line) + "\n")
                f_pred.write("\n")

                sentence_id += 1
                sentence_parts = []
                continue

            parts = line.split("\t")[:3] + ["_"]
            sentence_parts.append(parts)

    return output_path

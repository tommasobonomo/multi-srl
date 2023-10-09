import json
from pathlib import Path
from typing import Dict, List, Union

from spacy.language import Language
from spacy.tokens import Doc
from tqdm import tqdm
from trankit import Pipeline
from wasabi import msg

from src.token_types import DepRelation, FrameRole, Sentence, TokenFeatures


def parse_conll_file(
    file: Path, pipeline: Union[Language, Pipeline, None]
) -> List[Sentence]:
    """
    Parses CoNLL'09 files to sentences with head and stuff.
    If a spaCy or a trankit pipeline is passed, will use it for `semantic' DEP-tree parsing
    Otherwise will consider PHEAD and PDEPREL as semantic dep-tree annotations.
    """

    with open(file) as f:
        raw_file = f.read()

    parsed_sentences: List[List[TokenFeatures]] = []

    # Get sentences
    sentences = raw_file.strip().split("\n\n")
    for raw_sentence in tqdm(sentences, desc=f"Parsing {file.as_posix()}", leave=False):
        parsed_sentence = parse_raw_sentence(raw_sentence, pipeline)
        parsed_sentences.append(parsed_sentence)

    return parsed_sentences


def parse_raw_sentence(
    raw_sentence: str, pipeline: Union[Language, Pipeline, None]
) -> Sentence:
    """
    Parses raw sentence with CoNLL'09 features to a list of TokenFeatures.
    If a spaCy pipeline is passed, will use it for DEP-tree parsing and
    will be saved alongside annotated DEP features.
    Otherwise will consider PHEAD and PDEPREL as semantic dep-tree annotations.
    The semantic head algorithm is not applied here.
    """
    parsed_sentence: List[TokenFeatures] = []
    raw_tokens = [
        raw_token
        for raw_token in raw_sentence.split("\n")
        if not raw_token.startswith("#")
    ]
    spacy_pipeline, trankit_pipeline = None, None
    if pipeline is not None:
        text_tokens = [raw_token.split()[1] for raw_token in raw_tokens]
        if isinstance(pipeline, Language):
            spacy_pipeline = pipeline
            doc = spacy_pipeline(Doc(spacy_pipeline.vocab, text_tokens))
        elif isinstance(pipeline, Pipeline):
            trankit_pipeline = pipeline
            doc = trankit_pipeline(text_tokens, is_sent=True)

    for idx, raw_token in enumerate(raw_tokens):
        token = raw_token.split()

        # Dependency parsing tags from spaCy UD => semantic dependencies
        if spacy_pipeline is not None:
            spacy_token = doc[idx]
            sem_dep = spacy_token.dep_
            sem_dep_head_idx = spacy_token.head.i if spacy_token.head.i != idx else -1
            sem_dep_child_idx = [tok.i for tok in spacy_token.children]
        elif trankit_pipeline is not None:
            trakit_token = doc["tokens"][idx]  # type: ignore
            if "deprel" in trakit_token:
                sem_dep = trakit_token["deprel"]
            else:
                msg.warn("No deprel field in trankit token! Using empty dep relation.")
                sem_dep = ""
            sem_dep_head_idx = trakit_token["head"] - 1
            sem_dep_child_idx = [
                tok["id"] - 1 for tok in doc["tokens"] if tok["head"] - 1 == idx  # type: ignore
            ]
        else:
            # Take the fields PHEAD and PDEPREL as semantic dependencies
            sem_dep, sem_dep_head_idx, sem_dep_child_idx = (
                token[11],
                int(token[9]) - 1,
                [
                    int(tok.split()[0]) - 1
                    for tok in raw_tokens
                    if int(tok.split()[9]) - 1 == idx
                ],
            )

        # Also save syntactic dependencies
        synt_dep = token[10]
        synt_dep_head_idx = int(token[8]) - 1
        synt_dep_child_idx = [
            int(tok.split()[0]) - 1
            for tok in raw_tokens
            if int(tok.split()[8]) - 1 == idx
        ]

        parsed_sentence.append(
            TokenFeatures(
                idx=int(token[0]) - 1,
                text=token[1],
                lemma=token[2],
                pos=token[4],
                feat=token[6],
                synt_dep=DepRelation(synt_dep, synt_dep_head_idx, synt_dep_child_idx),
                sem_dep=DepRelation(sem_dep, sem_dep_head_idx, sem_dep_child_idx),
                frame_name=None,
                frame_roles=[],
            )
        )

    # Second pass for SRL parsing
    num_srl_frames = len(raw_tokens[0].split()) - 14
    frames: List[dict] = [
        {"frame_idx": None, "frame_name": None, "frame_roles": []}
        for _ in range(num_srl_frames)
    ]
    parsed_frames: set[int] = set()
    for i in range(len(frames)):
        frame = frames[i]
        for raw_token in raw_tokens:
            token = raw_token.split()
            if (
                token[13] != "_"
                and frame["frame_idx"] is None
                and (int(token[0]) - 1) not in parsed_frames
            ):
                frame["frame_idx"] = int(token[0]) - 1
                frame["frame_name"] = token[13]
                parsed_frames.add(frame["frame_idx"])  # type: ignore

            if token[14 + i] != "_":
                frame["frame_roles"].append(
                    FrameRole(
                        role_idx=int(token[0]) - 1,
                        role_name=token[14 + i],
                        semantic_head_idx=None,
                    )
                )

        parsed_token = parsed_sentence[frame["frame_idx"]]
        parsed_token.frame_name = frame["frame_name"]
        parsed_token.frame_roles = frame["frame_roles"]

    return parsed_sentence


def load_predictions(predictions_path: Path) -> Dict[int, Dict[int, dict]]:
    with open(predictions_path) as f:
        predictions = json.load(f)

    predictions = {
        int(sentence_idx): {
            int(predicate_idx): predicate_v
            for predicate_idx, predicate_v in sentence_v.items()
        }
        for sentence_idx, sentence_v in predictions.items()
    }

    return predictions


def write_as_conll(
    token: TokenFeatures, raw_token: List[str], correct_role_tokens: List[str]
):
    return (
        f"{raw_token[0]}\t"  # ID
        + f"{raw_token[1]}\t"  # FORM
        + f"{raw_token[2]}\t"  # LEMMA
        + f"{raw_token[3]}\t"  # PLEMMA
        + f"{raw_token[4]}\t"  # POS
        + f"{raw_token[5]}\t"  # PPOS
        + f"{raw_token[6]}\t"  # FEAT
        + f"{raw_token[7]}\t"  # PFEAT
        + f"{raw_token[8]}\t"  # HEAD
        + f"{token.sem_dep.head_idx + 1 if token.sem_dep.head_idx is not None else '_'}\t"  # PHEAD
        + f"{raw_token[10]}\t"  # DEPREL
        + f"{token.sem_dep.relation if token.sem_dep.relation is not None else '_'}\t"  # PDEPREL
        + f"{raw_token[12]}\t"  # FILLPRED
        + f"{raw_token[13]}\t"  # PRED
        + "\t".join(correct_role_tokens)  # APREDS
        + "\n"
    )


def write_predictions_to_conll(
    predictions: dict, output_path: Path, gold_path: Path, czech: bool = False
):
    with open(gold_path) as f_gold, open(output_path, "w") as f_pred:
        sentence_id = 0
        sentence_parts = []

        for line in f_gold:
            line = line.strip()

            if not line:
                sentence_roles = []

                if sentence_id in predictions:
                    predicate_indices = sorted(predictions[sentence_id].keys())

                    for predicate_idx in predicate_indices:
                        predicate_predictions = predictions[sentence_id][predicate_idx]
                        sentence_parts[predicate_idx][12] = "Y"
                        predicted_predicate = predicate_predictions[
                            "predicate"
                        ].replace("-v.", ".")
                        if czech and predicted_predicate == "[no-sense]":
                            predicted_predicate = sentence_parts[predicate_idx][3]
                        sentence_parts[predicate_idx][13] = (
                            predicted_predicate
                            if predicted_predicate != "<UNK>"
                            and predicted_predicate != "_"
                            else "unknown.01"
                        )
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

            parts = line.split("\t")[:12] + ["_", "_"]
            sentence_parts.append(parts)

    return output_path

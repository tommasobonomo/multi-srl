import json
from pathlib import Path
from typing import List, Optional

from amr_utils.alignments import AMR_Alignment
from amr_utils.amr import AMR
from amr_utils.amr_readers import AMR_Reader
from tap import Tap
from tqdm import tqdm
from trankit import Pipeline
from wasabi import msg

from src.srl.data.srl_data_module import SrlDataModule
from src.srl.data.srl_dataset import SrlDataset
from src.srl.model.srl_parser import SrlParser
from src.token_types import DepRelation, FrameRole, Sentence, TokenFeatures


class ScriptArgs(Tap):
    input_amr_dir: Path  # The path to a directory containing AMR-annotated file
    vocabulary_path: Path  # The path to the vocabulary file for multi-srl
    checkpoint_path: Path  # The path to the checkpoint file for multi-srl
    recompute: bool = False  # Whether to recompute the predictions
    persist_dir: Path = (
        Path("data") / "amr_data" / "predictions"
    )  # The path to persist the results to
    use_unified_srl: bool = False  # Whether to use the unified SRL model

    def process_args(self) -> None:
        if not (self.input_amr_dir.exists() and self.input_amr_dir.is_dir()):
            msg.fail(
                f"Input file {self.input_amr_dir} does not exist or is not a directory.",
                exits=1,
            )

        if not (self.persist_dir.exists() and self.persist_dir.is_dir()):
            msg.warn("Persist directory does not exist. Creating it...")
            self.persist_dir.mkdir(parents=True)


def load_model(checkpoint_path: Path, vocabulary_path: Path) -> SrlParser:
    model = SrlParser.load_from_checkpoint(
        checkpoint_path.as_posix(), vocabulary_path=vocabulary_path
    )
    model.eval()
    model.use_sense_candidates = False
    model.predictions_path = "amr_predictions.json"

    return model


def load_datamodule(model: SrlParser, vocabulary_path: Path) -> SrlDataModule:
    datamodule = SrlDataModule(
        vocabulary_path=vocabulary_path.as_posix(),
        language_model_name=model.hparams.language_model_name,  # type: ignore
        num_workers=2,
    )
    return datamodule


def parse_amr_sentence(
    amr_sentences: List[str],
    pipeline: Pipeline,
    srl_module: SrlParser,
    srl_datamodule: SrlDataModule,
    use_modified: bool = False,
) -> List[Sentence]:
    """
    Takes an AMR-annotated sentence and returns: a list of TokenFeatures with trankit
    dep parse and multi-srl annotations; the AMR graph as a Node object; and a mapping
    representing the alignment from sentence indexes to AMR node ids.
    """
    # Get metadata and AMR graph lines
    all_metadata = [
        [line.strip() for line in amr_lines.split("\n") if line.startswith("#")]
        for amr_lines in amr_sentences
    ]

    # Get tokens from metadata
    token_lines = [
        next(line for line in metadata if line.startswith("# ::tok"))
        for metadata in all_metadata
        if len(metadata) > 0
    ]
    sentence_tokens = [
        [
            TokenFeatures(idx=idx, text=token)
            for idx, token in enumerate(token_line.split()[2:])
        ]
        for token_line in token_lines
    ]

    # Get trankit output
    trankit_outs = [
        pipeline([token.text for token in sent], is_sent=True)
        for sent in tqdm(sentence_tokens, desc="Running trankit", leave=False)
    ]
    # Get SRL annotations
    srl_datamodule.pred_data = SrlDataset.load_sentences(trankit_outs)  # type: ignore
    predictions = {}
    for batch in tqdm(
        srl_datamodule.predict_dataloader(), desc="Running SRL", leave=False
    ):
        batch_predictions = srl_module.predict_step(
            batch,
            0,
            write_to_file=False,
            use_preidentified_predicates=False,
            use_modified=use_modified,
        )
        predictions.update(batch_predictions)  # type: ignore

    # Add SRL and dep annotations to tokens
    for sent_idx, (trankit_out, srl_pred) in enumerate(
        zip(trankit_outs, predictions.values())
    ):
        for idx, trankit_token in enumerate(trankit_out["tokens"]):
            token = sentence_tokens[sent_idx][idx]
            token.sem_dep = DepRelation(
                relation=trankit_token["deprel"], head_idx=trankit_token["head"] - 1
            )
            token.lemma = trankit_token["lemma"]
            token.pos = trankit_token["upos"]
        for idx, annotation in srl_pred.items():
            if idx >= len(sentence_tokens[sent_idx]):
                continue
            token = sentence_tokens[sent_idx][idx]
            token.frame_name = annotation["predicate"]
            token.frame_roles = [
                FrameRole(role_idx=role_idx, role_name=role)
                for role_idx, role in enumerate(annotation["roles"])
                if role != "_"
            ]

    return sentence_tokens


def token2node(
    token_idx: int, amr_graph: AMR, alignments: List[AMR_Alignment]
) -> Optional[str]:
    token_alignment = amr_graph.get_alignment(alignments, token_id=token_idx)
    if len(token_alignment.nodes) > 0:
        return token_alignment.nodes[0]
    elif len(token_alignment.edges) > 0:
        return token_alignment.edges[0][-1]
    else:
        return None


def are_directly_connected(source_node: str, target_node: str) -> bool:
    return (
        source_node in target_node
        and target_node.startswith(source_node)
        and len(target_node.split(".")) == (len(source_node.split(".")) + 1)
    )


def run(args: ScriptArgs) -> None:
    msg.divider("Alignment Evaluation")

    # Load the model and datamodule
    model = load_model(args.checkpoint_path, args.vocabulary_path)
    datamodule = load_datamodule(model, args.vocabulary_path)
    msg.good("multi-srl model and datamodule loaded.")
    pipeline = Pipeline(lang="english", gpu=True, embedding="xlm-roberta-large")
    msg.good("trankit pipeline loaded.")

    # Read in the AMR file
    total_true_positives = 0
    total_num_predictions = 0
    for input_amr_file in args.input_amr_dir.glob("*.txt"):
        true_positives = 0
        num_predictions = 0

        # Compute predictions if necessary
        predictions_path = args.persist_dir / input_amr_file.with_suffix(".json").name
        if args.recompute or not predictions_path.exists():
            msg.info(f"Computing predictions for {input_amr_file.name}...")

            amr_sentences = input_amr_file.read_text().split("\n\n")

            # Parse the AMR sentences with SRL and DEP annotations
            sentence_tokens = parse_amr_sentence(
                amr_sentences[1:], pipeline, model, datamodule, args.use_unified_srl
            )

            # Persist predicted sentences
            with predictions_path.open("w+") as f:
                json.dump(
                    [[tok.to_dict() for tok in sent] for sent in sentence_tokens],
                    f,
                    indent=2,
                )

            msg.good(
                f"Predictions computed and persisted to {predictions_path.as_posix()}"
            )
        else:
            msg.info(f"Loading predictions from {predictions_path.as_posix()}...")

            with predictions_path.open("r") as f:
                raw_predictions = json.load(f)

            sentence_tokens = [
                [TokenFeatures.from_dict(tok) for tok in sent]
                for sent in raw_predictions
            ]

            msg.good("Predictions loaded.")

        # Read in the AMR file with the AMR graph
        amr_reader = AMR_Reader()
        amr_graphs, alignments = amr_reader.load(
            input_amr_file.as_posix(), output_alignments=True
        )
        msg.good("AMR graphs loaded.")

        assert len(sentence_tokens) == len(amr_graphs), (
            f"Length mismatch: {len(sentence_tokens)} sentences, "
            f"{len(amr_graphs)} graphs."
        )

        # Iterate over the sentences and compare the alignments
        for sentence, amr_graph in zip(sentence_tokens, amr_graphs):
            # Loop over frames
            for token in sentence:
                if token.frame_name is None:
                    continue
                frame_node = token2node(token.idx, amr_graph, alignments)
                if frame_node is None:
                    # This frame is not included in the AMR graph, so we skip it
                    continue
                # Loop over roles
                for role in token.frame_roles:
                    role_node = token2node(role.role_idx, amr_graph, alignments)
                    # role_node can be None if it is not included in the AMR graph
                    # We still increase the number of predictions in this case
                    if role_node is not None:
                        # Check if the frame and role are connected
                        if are_directly_connected(frame_node, role_node):  # type: ignore
                            true_positives += 1
                        # shortest_path = graph_utils.get_shortest_path(
                        #     amr_graph, frame_node, role_node
                        # )
                        # if shortest_path is not None and len(shortest_path) == 2:
                        #     # There's a direct edge between the frame and role
                        #     true_positives += 1
                    if frame_node != role_node:
                        num_predictions += 1

        msg.info(f"File: {input_amr_file.name}")
        msg.info(f"True positives: {true_positives}")
        msg.info(f"Predictions: {num_predictions}")
        msg.info(f"Precision: {true_positives / num_predictions}")

        total_true_positives += true_positives
        total_num_predictions += num_predictions

    msg.divider("Total")
    msg.info(f"Total true positives: {total_true_positives}")
    msg.info(f"Total predictions: {total_num_predictions}")
    msg.info(f"Total precision: {total_true_positives / total_num_predictions}")


if __name__ == "__main__":
    args = ScriptArgs().parse_args()
    run(args)

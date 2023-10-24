import json
from pathlib import Path

from tap import Tap
from wasabi import msg


class ScriptArgs(Tap):
    predictions_file: Path  # Path to a predicted CoNLL09 `.json` file we want to evaluate
    gold_file: Path  # Path to a gold CoNLL09 `.json` file we want to evaluate against

    def process_args(self) -> None:
        assert (
            self.predictions_file.exists() and self.predictions_file.suffix == ".json"
        ), f"Invalid predictions file: {self.predictions_file}"
        assert (
            self.gold_file.exists() and self.gold_file.suffix == ".json"
        ), f"Invalid gold file: {self.gold_file}"


def run(args: ScriptArgs) -> None:
    msg.divider("Evaluating fine-grained predictions")

    # Load the predicted CoNLL09 file
    with open(args.predictions_file, "r") as f:
        predictions = json.load(f)

    # Load the gold CoNLL09 file
    with open(args.gold_file, "r") as f:
        gold = json.load(f)

    msg.good(
        f"Loaded {len(predictions)} sentences from {args.predictions_file}, {len(gold)} sentences from {args.gold_file}"
    )

    # Loop over all gold roles
    tp = 0
    total_roles = 0
    for key, sent in gold.items():
        for pred_idx, annotation in sent["annotations"].items():
            gold_roles = annotation["roles"]
            total_roles += sum(1 for role in gold_roles if role != "_")
            if key not in predictions or pred_idx not in predictions[key]:
                continue
            pred_roles = predictions[key][pred_idx]["roles"]
            for gold_role, pred_role in zip(gold_roles, pred_roles):
                if gold_role != "_" and gold_role == pred_role:
                    tp += 1

    # Compute and print true positive rate
    tpr = tp / total_roles
    msg.good(
        f"TPR: {tpr:.4f} ({tp} / {total_roles})\n"
        + f"TP: {tp}\n"
        + f"Total roles: {total_roles}\n"
    )


if __name__ == "__main__":
    args = ScriptArgs().parse_args()
    run(args)

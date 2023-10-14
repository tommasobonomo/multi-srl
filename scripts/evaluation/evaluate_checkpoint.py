import shutil
import subprocess
from pathlib import Path

from tap import Tap
from wasabi import msg


class ScriptArgs(Tap):
    input_model: Path  # Path to lightning logs dir of model to evaluate
    language: str  # Language to evaluate
    result_sheet: Path = Path(
        "results.csv"
    )  # Path to the .csv file to write results to
    checkpoint: str = "best"  # Which checkpoint to evaluate, or the path to the checkpoint to evaluate

    def process_args(self) -> None:
        assert (
            self.input_model.exists() and self.input_model.is_dir()
        ), f"input_model must be a directory, got {self.input_model}"

        if not self.result_sheet.exists():
            msg.warn(f"result_sheet {self.result_sheet} does not exist, creating it")
            self.result_sheet.parent.mkdir(exist_ok=True, parents=True)
            self.result_sheet.touch()

        assert self.language in (
            "ca",
            "cs",
            "de",
            "en",
            "es",
            "zh",
        ), f"language must be one of 'ca', 'cs', 'de', 'en', 'es', 'zh', got {self.language}"

        checkpoint_dir = self.input_model / "checkpoints"
        if self.checkpoint == "best":
            all_checkpoints = checkpoint_dir.glob("*.ckpt")
            try:
                best_checkpoint = max(
                    all_checkpoints,
                    key=lambda x: int(x.name.split("=")[-1].split(".")[0]),
                )
            except ValueError as e:
                msg.fail(f"No checkpoints found in {checkpoint_dir}")
                raise e
        else:
            best_checkpoint = checkpoint_dir / self.checkpoint
            assert (
                best_checkpoint.exists() and best_checkpoint.is_file()
            ), f"checkpoint {best_checkpoint} does not exist"

        self._checkpoint_path = best_checkpoint


def run(args: ScriptArgs):
    msg.divider(f"Evaluating checkpoint {args.checkpoint}")

    # Run prediction script
    result = subprocess.run(
        [
            "python",
            "scripts/training/trainer.py",
            "test",
            "--config",
            f"{args.input_model}/config.yaml",
            "--ckpt_path",
            f"{args._checkpoint_path}",
        ]
    )
    result.check_returncode()
    # Move predictions to model dir
    output_path = Path("lightning_logs/test_predictions.json")
    predictions_path = args.input_model / "test_predictions.json"
    shutil.move(str(output_path), args.input_model / "test_predictions.json")

    # Run evaluation script, capture output
    czech_arg = ["--czech"] if args.language == "cs" else []
    raw_out = subprocess.run(
        [
            "python",
            "scripts/evaluation/evaluate_on_conll2009.py",
            "--gold",
            f"data/original/conll2009/{args.language}/CoNLL2009_test.txt",
            "--predictions",
            f"{predictions_path}",
        ]
        + czech_arg,
        capture_output=True,
    )
    raw_out.check_returncode()

    # Parse output
    out = raw_out.stdout.decode("utf-8")
    sections = out.split("\n\n")
    semantic_section = sections[1]
    semantic_lines = semantic_section.split("\n")
    precision = float(semantic_lines[1].strip().split(" ")[-2])
    recall = float(semantic_lines[2].strip().split(" ")[-2])
    f1 = float(semantic_lines[3].strip().split(" ")[-1])

    # Write results to csv
    with args.result_sheet.open("a") as f:
        f.write(
            f"{args.input_model},{args.language},{args.checkpoint},{precision},{recall},{f1}\n"
        )

    msg.good(
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f} (saved to {args.result_sheet})"
    )


if __name__ == "__main__":
    args = ScriptArgs().parse_args()
    run(args)

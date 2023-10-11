import json
from pathlib import Path

from tap import Tap
from wasabi import msg


class ScriptArgs(Tap):
    input_path: Path  # Path to train file from which to build the dependency vocabulary
    output_path: Path  # Path to output resource file


def run(args: ScriptArgs) -> None:
    msg.divider(f"Building dependency vocabulary for {args.input_path}")

    sentences = args.input_path.read_text().split("\n\n")

    dependencies = set(["unk"])
    for sentence in sentences:
        for line in sentence.split("\n"):
            if line.startswith("#") or line.strip() == "":
                continue
            dependencies.add(line.split("\t")[10].lower())
    msg.good(f"Found {len(dependencies)} unique dependencies")

    with open(args.output_path, "w+") as f:
        json.dump({i: dep for i, dep in enumerate(dependencies)}, f, indent=2)


if __name__ == "__main__":
    run(ScriptArgs().parse_args())

import json
import random
from pathlib import Path

from tap import Tap
from wasabi import msg


class ScriptArgs(Tap):
    """
    Downsamples an input JSON file generated with the repo's preprocessing
    scripts to a given percentage.
    """

    input_file: Path  # Path to input file that should be downsampled
    output_file: Path  # Path to output file generated
    sample_pct: float = 0.1  # Percentage of data to sample
    random_seed: int = 5246  # Random seed for reproducibility


def run(args: ScriptArgs) -> None:
    msg.divider("Downsampling script")

    msg.info(f"Input file: {args.input_file}")
    msg.info(f"Sample percentage: {args.sample_pct}")

    with open(args.input_file) as f:
        input_data = json.load(f)

    msg.good("Loaded input data")

    all_samples = set(input_data.keys())
    random.seed(args.random_seed)
    sample = random.sample(all_samples, int(len(all_samples) * args.sample_pct))

    msg.good(f"Sampled {len(sample)} samples")

    output_data = {k: input_data[k] for k in sample}
    with open(args.output_file, "w+") as f:
        json.dump(output_data, f)

    msg.good(f"Saved output data to {args.output_file}")


if __name__ == "__main__":
    args = ScriptArgs().parse_args()
    run(args)

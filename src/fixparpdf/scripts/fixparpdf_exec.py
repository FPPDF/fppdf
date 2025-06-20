#!/usr/bin/env python3

from argparse import ArgumentTypeError, ArgumentParser
from pathlib import Path
import subprocess as sp


def _existing_path(value):
    value_path = Path(value)
    if not value_path.exists():
        return ArgumentTypeError(f"Could not find {value_path}")
    return value_path


def main():
    parser = ArgumentParser()

    parser.add_argument("runcard", type=_existing_path, help="Runcard defining the run")
    parser.add_argument(
        "--config", type=_existing_path, help="Configuration file to override default parameters"
    )
    args = parser.parse_args()

    # For the time being just call fixpar_nnpdf
    fixpar_script = Path(__file__).parent / "fixpar_nnpdf.py"
    sp.run(["python3", fixpar_script.as_posix(), "--config", args.runcard.as_posix()])


if __name__ == "__main__":
    main()

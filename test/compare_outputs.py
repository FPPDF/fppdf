#!/usr/bin/env python3
"""
Compare the test output .dat files against the references.
It uses only the numerical values in the .dat files and ignores all the rest.
"""
import argparse
from pathlib import Path
import sys


def _extract_numbers(dat_path):
    """Extract all numbers from a dat file."""
    """Extract all numeric tokens from a file."""
    ns = []
    for line in dat_path.read_text().splitlines():
        for t in line.split():
            try:
                ns.append(float(t))
            except ValueError:
                pass
    return ns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ref_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--tol", type=float, default=1e-6)
    args = p.parse_args()

    errors = []
    for ref_f in sorted(args.ref_dir.rglob("*.dat")):
        out_f = args.out_dir / ref_f.relative_to(args.ref_dir)
        if not out_f.exists():
            errors.append(f"Missing: {ref_f.relative_to(args.ref_dir)}")
            continue
        rn, on = _extract_numbers(ref_f), _extract_numbers(out_f)
        if len(rn) != len(on):
            errors.append(
                f"count mismatch: {ref_f.relative_to(args.ref_dir)} ({len(rn)} vs {len(on)})"
            )
            continue
        for i, (r, o) in enumerate(zip(rn, on)):
            if abs(o - r) > args.tol * max(abs(r), 1.0):
                errors.append(f"{ref_f.relative_to(args.ref_dir)}[{i}]: {r} vs {o}")

    if errors:
        print(f"FAIL ({len(errors)} errors):")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()

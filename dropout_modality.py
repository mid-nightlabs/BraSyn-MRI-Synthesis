"""
Create a pseudo validation/test set by randomly dropping one modality per case.
Matches the BraSyn challenge validation and test setup (arXiv:2305.09011).
Usage: python dropout_modality.py <full_data_folder> <output_folder>
"""

import argparse
import os
import random
import shutil
from pathlib import Path

from dataset import MODALITIES


def main():
    parser = argparse.ArgumentParser(description="BraSyn: create pseudo val/test set with one modality dropped per case")
    parser.add_argument("source", type=str, help="Folder containing full case subdirs (each with 4 modalities)")
    parser.add_argument("dest", type=str, help="Output folder for pseudo validation set")
    parser.add_argument("--seed", type=int, default=123456)
    args = parser.parse_args()
    random.seed(args.seed)
    source = Path(args.source)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    case_dirs = sorted([d for d in source.iterdir() if d.is_dir()])
    for case_dir in case_dirs:
        files = list(case_dir.glob("*.nii.gz")) + list(case_dir.glob("*.nii"))
        has_all = all(
            any(f.name.replace(".nii.gz", "").replace(".nii", "").endswith(f"-{m}") for f in files)
            for m in MODALITIES
        )
        if not has_all:
            continue
        drop = random.choice(MODALITIES)
        out_case = dest / case_dir.name
        out_case.mkdir(parents=True, exist_ok=True)
        for f in files:
            stem = f.name.replace(".nii.gz", "").replace(".nii", "")
            if stem.endswith(f"-{drop}"):
                continue
            shutil.copy2(f, out_case / f.name)
        print(f"Dropped {drop} for case {case_dir.name}")


if __name__ == "__main__":
    main()

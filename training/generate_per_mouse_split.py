"""
Generate an nnU-Net splits_final.json using a per-animal (per-mouse) group
split rather than nnU-Net's default per-image random split.

Why: consecutive images from the same mouse are acquired only ~300 ms apart
and are near-duplicates. A naive per-image split can place near-duplicate
images on both sides of the train/validation boundary, artificially
inflating the validation Dice score (data leakage). This script guarantees
that all images from a given mouse fall entirely on one side.

Expects label files named "<mouseID>_<imgID>.nii.gz" in labels_dir, e.g.
"souris01_img05.nii.gz".

Usage:
    python generate_per_mouse_split.py --labels_dir path/to/labelsTr \
                                        --output path/to/splits_final.json \
                                        --n_folds 5 --seed 42
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict


def generate_splits(labels_dir: str, output_path: str, n_folds: int = 5, seed: int = 42) -> None:
    all_cases = [f.replace(".nii.gz", "") for f in os.listdir(labels_dir) if f.endswith(".nii.gz")]

    mouse_to_cases = defaultdict(list)
    for case in all_cases:
        match = re.match(r"([A-Za-z0-9]+)_", case)
        if match:
            mouse_to_cases[match.group(1)].append(case)
        else:
            print(f"WARNING: unexpected filename pattern, skipped -> {case}")

    mice = sorted(mouse_to_cases.keys())
    print(f"{len(mice)} animal(s) found: {mice}")
    print(f"{len(all_cases)} image(s) total")

    random.seed(seed)
    mice_shuffled = mice.copy()
    random.shuffle(mice_shuffled)

    folds_mice = [mice_shuffled[i::n_folds] for i in range(n_folds)]

    splits = []
    for fold_idx in range(n_folds):
        val_mice = folds_mice[fold_idx]
        train_mice = [m for m in mice if m not in val_mice]

        val_cases = sorted(c for m in val_mice for c in mouse_to_cases[m])
        train_cases = sorted(c for m in train_mice for c in mouse_to_cases[m])

        splits.append({"train": train_cases, "val": val_cases})

        print(
            f"Fold {fold_idx}: {len(train_mice)} train animals / {len(val_mice)} val animals "
            f"({len(train_cases)} train images / {len(val_cases)} val images)"
        )

    with open(output_path, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"\nsplits_final.json written -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels_dir", required=True, help="nnU-Net labelsTr folder")
    parser.add_argument("--output", required=True, help="Path to write splits_final.json")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_splits(args.labels_dir, args.output, args.n_folds, args.seed)

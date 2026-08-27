"""
Sanity check for nnU-Net label masks: flags any file in labelsTr that is not
a proper binary mask (values other than {0, 1}), e.g. a grayscale image
accidentally exported in place of a segmentation mask.

Usage:
    python check_label_integrity.py --labels_dir path/to/labelsTr
"""

import argparse
import os

import numpy as np
import SimpleITK as sitk


def check_labels(labels_dir: str) -> None:
    flagged = 0
    for fname in sorted(os.listdir(labels_dir)):
        if not fname.endswith(".nii.gz"):
            continue
        path = os.path.join(labels_dir, fname)
        arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
        unique_vals = np.unique(arr)

        if not set(unique_vals.tolist()).issubset({0, 1}):
            preview = unique_vals[:10]
            suffix = "..." if len(unique_vals) > 10 else ""
            print(f"{fname} -> unexpected values: {preview}{suffix} ({len(unique_vals)} unique values)")
            flagged += 1

    if flagged == 0:
        print("All label files are valid binary masks (0/1).")
    else:
        print(f"\n{flagged} file(s) flagged for review.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels_dir", required=True, help="nnU-Net labelsTr folder")
    args = parser.parse_args()

    check_labels(args.labels_dir)

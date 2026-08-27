"""
Post-processing for nnU-Net spleen segmentation predictions.

The model occasionally predicts a small isolated blob elsewhere in the
image (a false positive disconnected from the main spleen region), in
addition to correctly segmenting the spleen itself. This script removes
any spurious component by keeping only the largest connected component
in each predicted mask.

Usage:
    python keep_largest_component.py --input_dir path/to/predictions \
                                      --output_dir path/to/predictions_clean
"""

import argparse
import os

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def keep_largest_component(input_path: str, output_path: str) -> int:
    """Keep only the largest connected component of a binary mask.

    Returns the number of removed (spurious) components.
    """
    img = sitk.ReadImage(input_path)
    arr = sitk.GetArrayFromImage(img)

    labeled, num_features = ndimage.label(arr)

    if num_features > 1:
        sizes = ndimage.sum(arr, labeled, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        arr_clean = (labeled == largest_label).astype(arr.dtype)
        removed = num_features - 1
    else:
        arr_clean = arr
        removed = 0

    img_clean = sitk.GetImageFromArray(arr_clean)
    img_clean.CopyInformation(img)
    sitk.WriteImage(img_clean, output_path)
    return removed


def clean_predictions_folder(input_dir: str, output_dir: str) -> None:
    """Apply keep_largest_component to every .nii.gz file in a folder."""
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".nii.gz")]
    print(f"{len(files)} file(s) to process in {input_dir}\n")

    total_cleaned = 0
    for fname in sorted(files):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        removed = keep_largest_component(in_path, out_path)

        if removed > 0:
            print(f"{fname}: removed {removed} spurious component(s)")
            total_cleaned += 1
        else:
            print(f"{fname}: nothing to clean (single component)")

    print(f"\nDone. {total_cleaned}/{len(files)} file(s) had spurious components.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Folder of raw nnU-Net predictions (.nii.gz)")
    parser.add_argument("--output_dir", required=True, help="Folder to write cleaned masks to")
    args = parser.parse_args()

    clean_predictions_folder(args.input_dir, args.output_dir)

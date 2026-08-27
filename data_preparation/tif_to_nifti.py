"""
Convert 2D .tif/.tiff ultrasound images to the pseudo-3D NIfTI (.nii.gz)
format expected by nnU-Net's 2D configuration.

Why a "pseudo-3D" volume: nnU-Net's 2D pipeline still expects a 3-element
spacing tuple (x, y, z) internally. A plain 2D .tif converted naively to
NIfTI only carries 2 spacing dimensions and will fail at inference with
"IndexError: list index out of range" during preprocessing. This script
adds a singleton z-dimension (shape (1, H, W)) and a matching 3-element
spacing to avoid that error.

Usage:
    python tif_to_nifti.py --input_dir path/to/tif_images \
                            --output_dir path/to/nnunet_input \
                            --pixel_spacing_xy 0.2645833194255829
"""

import argparse
import os

import numpy as np
import SimpleITK as sitk


def convert_folder(input_dir: str, output_dir: str, pixel_spacing_xy: float) -> None:
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith((".tif", ".tiff"))]
    print(f"{len(files)} .tif file(s) found in {input_dir}\n")

    for fname in sorted(files):
        img = sitk.ReadImage(os.path.join(input_dir, fname))
        arr = sitk.GetArrayFromImage(img)  # (H, W) or (H, W, 3/4) if color

        # Convert RGB(A) to grayscale if needed
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = np.mean(arr[..., :3], axis=-1).astype(arr.dtype)

        # Add a singleton z-dimension: (H, W) -> (1, H, W)
        arr_3d = arr[np.newaxis, ...]

        img_3d = sitk.GetImageFromArray(arr_3d)
        img_3d.SetSpacing((pixel_spacing_xy, pixel_spacing_xy, 1.0))  # (x, y, z)

        base_name = os.path.splitext(fname)[0]
        out_name = f"{base_name}_0000.nii.gz"
        sitk.WriteImage(img_3d, os.path.join(output_dir, out_name))
        print(f"{fname} -> {out_name}  (shape: {arr_3d.shape})")

    print("\nConversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Folder of raw .tif/.tiff images")
    parser.add_argument("--output_dir", required=True, help="Folder to write nnU-Net-ready .nii.gz files to")
    parser.add_argument(
        "--pixel_spacing_xy",
        type=float,
        required=True,
        help="In-plane pixel spacing in mm (must match the spacing used at training time, "
             "found in nnUNet_preprocessed/DatasetXXX/nnUNetPlans.json)",
    )
    args = parser.parse_args()

    convert_folder(args.input_dir, args.output_dir, args.pixel_spacing_xy)

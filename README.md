# Automatic Mouse Spleen Segmentation on B-Mode Ultrasound (nnU-Net)

An end-to-end, reproducible pipeline for training and using an [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) model to automatically segment the mouse spleen on B-mode ultrasound images (Vevo F2 system), replacing manual contouring in 3D Slicer.

Developed at PhyMedExp (IPAM/Biocampus, INSERM, CNRS, Université de Montpellier).

This repository covers **segmentation only** — from raw ultrasound exports to a trained model and its use on new images. A companion repository covering the downstream shear wave elastography (SWE) heterogeneity analysis will be released separately.

---

## Table of contents

1. [Overview](#1-overview)
2. [Requirements](#2-requirements)
3. [Installation](#3-installation)
4. [Data preparation](#4-data-preparation)
5. [Building a leakage-free train/validation split](#5-building-a-leakage-free-trainvalidation-split)
6. [Training](#6-training)
7. [Inference on new images](#7-inference-on-new-images)
8. [Post-processing predictions](#8-post-processing-predictions)
9. [Reading the training curves](#9-reading-the-training-curves)
10. [Using the model on a different computer](#10-using-the-model-on-a-different-computer)
11. [Troubleshooting](#11-troubleshooting)
12. [Known limitations](#12-known-limitations)
13. [Pretrained model weights](#13-pretrained-model-weights)
14. [Citation](#14-citation)
15. [License](#15-license)

---

## 1. Overview

```
raw Vevo F2 export
      │
      ▼
1. Grayscale conversion (3D Slicer)          convert raw export to a grayscale volume
      │
      ▼
2. Manual reference masks (3D Slicer)        ground truth for training (not scripted — see §4.2)
      │
      ▼
3. data_preparation/check_label_integrity.py verify manual masks are valid binary labels
      │
      ▼
4. training/generate_per_mouse_split.py      build a leakage-free train/val split (grouped by animal)
      │
      ▼
5. nnUNetv2_plan_and_preprocess              nnU-Net dataset verification & auto-configuration
      │
      ▼
6. nnUNetv2_train                            train the segmentation model
      │
      ▼
7. nnUNetv2_predict                          predict spleen masks on new images
      │
      ▼
8. postprocessing/keep_largest_component.py  remove spurious isolated false-positive blobs
      │
      ▼
   final spleen mask (.nii.gz)
```

## 2. Requirements

- Python 3.10 or 3.11 (nnU-Net v2 is not reliably compatible with 3.12+ at the time of writing; if a newer Python version is already installed system-wide, install 3.10/3.11 alongside it rather than replacing it — see [Troubleshooting](#11-troubleshooting))
- An NVIDIA GPU with CUDA support is **strongly recommended for training** (a full 1000-epoch training run took ≈ 210 s/epoch, i.e. several days, on a dedicated NVIDIA GPU; training on CPU only is not practically feasible)
- A GPU is **not required for inference** (prediction on new images) — CPU works, just slower
- Windows, Linux, or macOS (this pipeline was developed and tested on Windows)

## 3. Installation

```bash
# create and activate a virtual environment (use the correct Python version explicitly
# if multiple are installed, e.g. `py -3.11` on Windows)
python -m venv nnunet_env

# Windows (cmd):
nnunet_env\Scripts\activate.bat
# Windows (PowerShell):
nnunet_env\Scripts\Activate.ps1
# Linux/macOS:
source nnunet_env/bin/activate

pip install -r requirements.txt
```

**Install PyTorch with CUDA support separately**, matching your GPU driver's CUDA version (check with `nvidia-smi`). Do **not** run a plain `pip install torch` — it installs a CPU-only build. Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), select your OS / CUDA version, and run the generated command, e.g.:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU support is active:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

This should print `True` followed by your GPU's name.

### Set nnU-Net's environment variables

nnU-Net requires three environment variables pointing to your working folders:

```bash
# Windows (PowerShell), per-session:
$env:nnUNet_raw = "C:\path\to\nnUNet_raw"
$env:nnUNet_preprocessed = "C:\path\to\nnUNet_preprocessed"
$env:nnUNet_results = "C:\path\to\nnUNet_results"

# Linux/macOS, per-session:
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

For a permanent setup on Windows: **Start → "Edit the system environment variables" → Environment Variables… → New…** under "User variables", and create the three variables above. Restart your terminal/IDE afterwards for the change to take effect.

## 4. Data preparation

### 4.1 Convert raw exports to grayscale

Raw Vevo F2 exports were converted to grayscale volumes in **3D Slicer** prior to manual segmentation. This step is performed manually in Slicer (not scripted in this repository).

### 4.2 Generate manual reference masks

Ground-truth spleen masks were traced manually in **3D Slicer** (Segment Editor module), then exported as label maps (**not** as a plain Volume — this distinction matters, see [Troubleshooting](#11-troubleshooting)). This step is not automated; it is the source of the "ground truth" the model learns from.

### 4.3 Organize the nnU-Net dataset folder

Following the [nnU-Net data format specification](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md):

```
nnUNet_raw/
    Dataset001_Spleen/
        imagesTr/
            mouse01_img01_0000.nii.gz
            mouse01_img02_0000.nii.gz
            ...
        labelsTr/
            mouse01_img01.nii.gz
            mouse01_img02.nii.gz
            ...
        dataset.json
```

- Image files: `<caseID>_0000.nii.gz` (the `_0000` suffix denotes channel 0 — required by nnU-Net even for single-channel data)
- Label files: `<caseID>.nii.gz`, binary masks (0 = background, 1 = spleen)
- `<caseID>` naming convention used in this study: `<mouseID>_<imageID>`, e.g. `souris01_img05`

### 4.4 Verify label integrity

Before training, check that every label file is a genuine binary mask — a grayscale image accidentally exported in place of a segmentation mask will otherwise silently corrupt training:

```bash
python data_preparation/check_label_integrity.py --labels_dir nnUNet_raw/Dataset001_Spleen/labelsTr
```

This flags any file containing values other than {0, 1}. (In this study, this check caught one corrupted file that contained raw grayscale image data instead of a mask.)

## 5. Building a leakage-free train/validation split

**Why this matters:** if your images include multiple near-duplicate frames per animal (e.g. consecutive frames from the same acquisition, taken milliseconds apart — common in cine-loop or EKV-style acquisitions), nnU-Net's **default per-image random split** can place near-duplicate images on both sides of a given fold's train/validation boundary. The model is then partly validated on images it has effectively already seen, which **artificially inflates the validation Dice score** (data leakage). This was empirically observed in this project: the naive per-image split produced a validation Dice ≈ 0.93 within the first few epochs, well before the model could plausibly have converged — a signature of leakage.

The fix is to group images **by animal of origin** rather than by individual image, so that all images from a given mouse fall entirely on the training side or entirely on the validation side of each fold:

```bash
python training/generate_per_mouse_split.py \
    --labels_dir nnUNet_raw/Dataset001_Spleen/labelsTr \
    --output nnUNet_preprocessed/Dataset001_Spleen/splits_final.json \
    --n_folds 5 --seed 42
```

Run `nnUNetv2_plan_and_preprocess` (see below) **before** generating this file (it needs `nnUNet_preprocessed/Dataset001_Spleen` to exist), and make sure `generate_per_mouse_split.py` is re-run (overwriting `splits_final.json`) if you re-preprocess the dataset, so nnU-Net picks up the corrected split rather than regenerating its own default one on the next training run.

## 6. Training

### 6.1 Verify and preprocess the dataset

```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

(`-d 1` refers to `Dataset001_Spleen`; adjust the number to match your dataset.) This validates the dataset, then auto-configures the network architecture, patch size, batch size, and preprocessing parameters from the dataset's "fingerprint" — no manual hyperparameter tuning is required. It writes an initial (per-image, non-grouped) `splits_final.json`; regenerate it with the per-mouse script above afterwards.

### 6.2 Train

```bash
nnUNetv2_train 1 2d 0
```

- `1`: dataset ID
- `2d`: configuration (2D images — the appropriate choice for single-slice B-mode data; nnU-Net also supports `3d_fullres` / `3d_cascade_fullres` for volumetric data)
- `0`: fold number (of the 5 defined in `splits_final.json`)

Training runs for 1000 epochs by default. In this study, using an NVIDIA GPU, this took ≈ 210 seconds/epoch (several days of wall-clock time). A checkpoint is saved regularly (`checkpoint_latest.pth`), and results (including a `progress.png` training curve) are written to:

```
nnUNet_results/Dataset001_Spleen/nnUNetTrainer__nnUNetPlans__2d/fold_0/
```

### 6.3 Resuming after an interruption

If training is interrupted (power loss, reboot, sleep, etc.), **do not** re-run the plain command above — it restarts from epoch 0 and overwrites the existing checkpoint. Resume with:

```bash
nnUNetv2_train 1 2d 0 --c
```

(`--c`, not `-c` — a single dash is not a recognized argument and will fail.) After resuming, check that the printed epoch number and learning rate continue from where training left off (rather than epoch 0 / learning rate reset to its initial value 0.01) to confirm the resume worked correctly.

**Tip:** disable your system's sleep/hibernate settings and pause automatic OS updates for the duration of training, since either can silently interrupt a multi-day run.

## 7. Inference on new images

### 7.1 Prepare input images

Input images must be in the same format used for training (`.nii.gz`, `_0000` suffix). If starting from `.tif`/`.tiff` images instead, convert them first:

```bash
python data_preparation/conversion.py \
    --input_dir new_tif_images/ \
    --output_dir inference_input/ \
    --pixel_spacing_xy 0.2645833194255829
```
The conversion script:

converts RGB images to grayscale;
adds a singleton z-dimension;
sets the in-plane pixel spacing used during training;
saves the images with the required _0000.nii.gz suffix.

For example:

```
inference_input/
├── souris4660_img01_0000.nii.gz
├── souris4660_img02_0000.nii.gz
└── ...
```

(Use the same in-plane pixel spacing as at training time — found in `nnUNet_preprocessed/Dataset001_Spleen/nnUNetPlans.json`. A mismatch here does not raise an error but will silently resample images incorrectly.)

### 7.2 Configure nnU-Net

Before prediction, nnU-Net must know where the trained model is stored.

Set the nnUNet_results environment variable to the directory containing the trained model:

Windows PowerShell:
```PowerShell
$env:nnUNet_results = "C:\path\to\nnUNet_results"
```
For inference, nnUNet_raw and nnUNet_preprocessed are not required if the trained model is already available in nnUNet_results.

The expected model structure is:
```
nnUNet_results/
└── Dataset001_Rate/
    └── nnUNetTrainer__nnUNetPlans__2d/
        └── fold_0/
            └── checkpoint_final.pth
```

Important: avoid paths containing accented characters or other special characters when possible, especially on Windows. 

### 7.3 Predict

Run: 

```PowerShell
nnUNetv2_predict \
    -i inference_input/ \
    -o predictions/ \
    -d 1 \
    -c 2d \
    -f 0
```
where:

- ```-i``` = input NIfTI images;
- ```-o``` = output directory for predicted masks;
- ```-d 1``` = dataset ID;
- ```-c 2d``` = 2D configuration;
- ```-f 0``` = fold 0.

The predicted masks are saved as:

```
predictions/
├── souris4660_img01.nii.gz
├── souris4660_img02.nii.gz
└── ...
```

## 8. Post-processing predictions

The trained model occasionally predicts a small, spatially disconnected false-positive blob elsewhere in the image (e.g. a nearby vessel with similar echogenicity to the spleen), in addition to correctly segmenting the spleen itself. This is removed by keeping only the largest connected component in each predicted mask:

```bash
python postprocessing/keep_largest_component.py \
    --input_dir predictions/ \
    --output_dir predictions_clean/
```

This step has no effect on masks that are already a single connected region, and only removes spurious secondary components — it does not alter the main predicted spleen contour.

## 9. Reading the training curves

`progress.png` (auto-generated per fold) plots, per epoch: training loss, validation loss, pseudo-Dice score (raw + moving average), epoch duration, and learning rate.

- **Training loss** should decrease steadily and plateau.
- **Validation loss** should decrease then plateau, staying close to the training loss — a validation loss that diverges upward while training loss keeps falling indicates overfitting.
- **Pseudo-Dice (moving average)** should increase toward 1; a segmentation of a well-defined organ is generally considered usable above ≈ 0.85–0.90.

In this study, the final model reached a validation Dice (moving average) of ≈ 0.92–0.93, plateauing around epoch 400–500, with no signs of overfitting through the full 1000 epochs.

## 10. Using the model on a different computer

**A trained model is portable across machines; a Python virtual environment is not.** Never copy a `venv`/`nnunet_env` folder between computers — it hardcodes paths to the Python installation it was created on and will fail with errors like `did not find executable at ...` on another machine. Always recreate the virtual environment from scratch (§3) on each new machine.

To reuse an already-trained model elsewhere:

1. Set up a fresh environment on the new machine (§2–3). A GPU is not required for inference.
2. Copy the results folder `nnUNet_results/Dataset001_Spleen/nnUNetTrainer__nnUNetPlans__2d/` (including `fold_0/`) to the new machine.
3. Set the `nnUNet_results` environment variable to point to its new location (`nnUNet_raw` / `nnUNet_preprocessed` are only needed if you intend to retrain, not just to predict).
4. Run `nnUNetv2_predict` as in §7.

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `nnunet_env\Scripts\Activate.ps1` fails, "running scripts is disabled" | PowerShell execution policy blocks the activation script | Run PowerShell **as administrator**: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, or activate via Command Prompt instead (`nnunet_env\Scripts\activate.bat`) |
| `did not find executable at '...pythonXX...'` when activating a copied venv | Virtual environments are not portable between machines | Delete the copied venv folder and recreate it from scratch on the new machine (§3) |
| `RuntimeError: Some segmentation images contained unexpected labels` listing dozens/hundreds of distinct values | A label file is a grayscale image (or otherwise not a proper 0/1 binary mask) — e.g. the grayscale B-mode volume itself accidentally placed in `labelsTr` instead of the traced mask | Run `check_label_integrity.py` (§4.4) to find the offending file(s); re-export the correct binary mask from 3D Slicer as a Labelmap (not a Volume) |
| Suspiciously high validation Dice (e.g. > 0.9) within the first few epochs | Train/validation split leaking near-duplicate images between the two sets | Regenerate `splits_final.json` with `generate_per_mouse_split.py` (§5) |
| `nnUNetv2_train ... -c` → `error: unrecognized arguments: -c` | Wrong flag | Use `--c` (double dash), not `-c` |
| Training silently restarts from epoch 0 with learning rate back at 0.01 after a resume attempt | Resume flag missing or wrong, or the prior checkpoint was already overwritten | Always resume with `nnUNetv2_train <d> <config> <fold> --c`; verify the resumed epoch/learning rate in the printed log before assuming the resume worked |
| `IndexError: list index out of range` during `nnUNetv2_predict` preprocessing on images converted from `.tif` | A plain 2D `.tif` → NIfTI conversion only carries a 2-element spacing tuple; nnU-Net's 2D pipeline expects 3 | Use `conversion.py` (§7.1), which adds a singleton z-dimension and a 3-element spacing |
| Predicted masks contain a small isolated blob elsewhere in the image | Model false positive on a structure with similar echogenicity to the spleen | Apply `keep_largest_component.py` (§8) |
| Python version conflicts (e.g. system default is too new for nnU-Net) | Multiple Python versions needed on the same machine | Install Python 3.10/3.11 alongside the existing version rather than replacing it; use `py -3.11 -m venv nnunet_env` (Windows) to target the correct one explicitly |

## 12. Known limitations

- The model shows occasional under-segmentation (missed spleen regions) on animal groups underrepresented in the training cohort. Addressing this requires expanding training data coverage for those groups, not a pipeline/code change.
- The per-mouse split strategy assumes multiple images per animal are near-duplicates; if your acquisition protocol produces genuinely independent images per animal, a standard per-image split may be adequate instead.
- The grayscale conversion step (§4.1) is performed manually in 3D Slicer and is not scripted in this repository.

## 13. Pretrained model weights

Model weights are not stored in this Git repository (too large for standard Git). They are hosted separately: **[link to be added — GitHub Release or Zenodo]**.

Once downloaded, follow §10 above to use them.

## 14. Citation

If you use this pipeline, please cite:

> [Publication in preparation — citation to be added]

This work builds on nnU-Net:

> Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203–211. [github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

## 15. License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

PhyMedExp animal housing staff and the Imagerie Préclinique Appliquée de Montpellier (IPAM, Biocampus).

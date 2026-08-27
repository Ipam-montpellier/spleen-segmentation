# How to publish this repository on GitHub

## 1. Fill in the placeholders

Before publishing:
- Fill in the citation in `README.md` once the publication is available (or add a preprint DOI).
- Replace `Dataset001_Spleen` placeholders in the README with your actual dataset name/number if you want the commands to be copy-pasteable as-is.

## 2. GitHub organization

Repository lives under the **Ipam-montpellier** organization on GitHub.

## 3. Create the repository

On GitHub, from the `Ipam-montpellier` organization page: **New repository** → name it `spleen-segmentation` (or your preferred name) → set to **Public** → do NOT initialize with a README/license (we already have them) → **Create repository**.

## 4. Push this folder to GitHub

From inside this folder, in a terminal (or VS Code's integrated terminal):

```bash
git init
git add .
git commit -m "Initial commit: automatic spleen segmentation pipeline (nnU-Net)"
git branch -M main
git remote add origin https://github.com/Ipam-montpellier/spleen-segmentation.git
git push -u origin main
```

If Git is not installed, download it from [git-scm.com](https://git-scm.com/). If you don't yet have push access to `Ipam-montpellier`, ask the organization's administrator to add you as a member with write access first.

## 5. Host the trained model weights separately

Git is not designed for large binary files like `.pth` checkpoints. Two good options:

### Option A — GitHub Releases (simplest)
1. On your repo page: **Releases** → **Create a new release**
2. Tag it (e.g. `v1.0-model`), give it a title, and attach the checkpoint file(s) (`checkpoint_final.pth`, `plans.json`, `dataset.json`)
3. Copy the download link and paste it into the "Pretrained model weights" section of `README.md`

### Option B — Zenodo (recommended if you want a citable DOI)
1. Create an account at [zenodo.org](https://zenodo.org/), connect it to your GitHub account
2. Upload the model weights as a new "Software" or "Dataset" entry
3. Zenodo issues a permanent DOI you can cite in the publication and link from the README

## 6. Optional polish

- Add a `CONTRIBUTING.md` if you expect external contributions.
- Add 1-2 small example input/output images under `docs/examples/` so users can test the pipeline without needing your full dataset.
- Once the paper is published, update the citation block in the README with the final reference.

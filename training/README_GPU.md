# Training the detector on a borrowed GPU

The detector (RF-DETR-Nano, 384 x 384, about 30.5 M parameters) is fine-tuned **only on a borrowed or teammate GPU**, never on a project laptop (CPU only; measured 6.7 s per iteration at batch 4 on the Lead's laptop, which extrapolates to about 22 hours for 470 images x 100 epochs. That is an estimate from `gpu_preflight`, see below). Every script here is a Python entry point with explicit paths, so the same commands run on any machine.

Which GPU will be used is **not decided** (a teammate's RTX 4090 or 4070 Ti, or Kaggle). Pick path A or B below.

**Conventions in this guide**

- **[NEEDS LEAD'S YES]** marks every step that downloads anything (IMPLEMENTATION_PLAN.md R7). Do not run it without a yes in chat. Steps without the mark download nothing.
- Commands are written for a shell in the repo root. On Windows PowerShell use the same `python -m ...` lines; only the activation line differs.
- The pretrained weights are **never** downloaded by the scripts unless you pass `--allow-download`.

---

## What you need to copy to the GPU machine

| What | From (Lead's laptop) | Size | Why |
| --- | --- | --- | --- |
| the repo (clone `develop`, or copy the folder) | `D:\Z_PS2_2026\BAS` | small | the scripts, `config/experiment.json`, `contracts.py`, the lockfile |
| `data/dataset/` | written by `python -m training.build_dataset` | about 100 MB of JPEGs | the dataset (`train/`, `valid/`, `test/`, each with `_annotations.coco.json`) |
| `reports/dataset.json` | committed in the repo | small | its `dataset_stamp` goes into the training summary |
| RF-DETR-Nano pretrained weights `rf-detr-nano.pth` | `%USERPROFILE%\.roboflow\models\` (Linux: `~/.roboflow/models/`) | 366 MB | the COCO-pretrained starting point; MD5-checked by rfdetr |
| YOLO11n weights `yolo11n.pt` (only for the fallback) | `%USERPROFILE%\.cache\ultralytics_benchmark\` | 5.6 MB | the benchmarked fallback only |

`data/` and `weights/` are git-ignored, so a plain clone does **not** contain the dataset or the weights: copy them (USB stick, shared drive, `scp`).

---

## A. A teammate's CUDA machine (Windows or Linux)

**A1. Get the code.** Clone `develop` (or copy the repo folder). **[NEEDS LEAD'S YES]** if it is a `git clone` from GitHub.

```bash
git clone <repo-url> BAS && cd BAS && git checkout develop
```

**A2. Install `uv`** if the machine does not have it (<https://docs.astral.sh/uv/>). **[NEEDS LEAD'S YES]** (downloads the installer).

**A3. Copy the data and the weights cache** (no download):

- copy `data/dataset/` into the repo as `data/dataset/`;
- copy `rf-detr-nano.pth` into `~/.roboflow/models/` (Windows: `%USERPROFILE%\.roboflow\models\`), or keep it anywhere and pass `--weights PATH` to every command below;
- for the YOLO fallback also copy `yolo11n.pt`, and pass `--weights PATH`.

**A4. Create the environment from the lockfile.** **[NEEDS LEAD'S YES]** (downloads the packages).

```bash
uv sync --group train
```

For the YOLO11n fallback add the tools group, which holds Ultralytics (AGPL-3.0): `uv sync --group train --group tools`.

> **Known gap, seen on the Lead's laptop, not yet fixed.** rfdetr's training stack lives in its `[train]` extras (`pytorch_lightning`, `torchmetrics`, `pycocotools`, ...). The lockfile's `train` group lists `rfdetr` **without** that extra, so `uv sync --group train` does not install them and `RFDETRNano().train()` stops with an `ImportError`. `python -m training.finetune --model rfdetr ...` detects this and prints which packages are missing (exit code 4). The fix is for **P2 to change the `train` group to `rfdetr[train]` and re-lock** (only P2 edits `uv.lock`; ISSUES.md 2026-10-03 P1.5 prep). Until then installing the extra by hand is possible but is **outside the lockfile**: write the exact command and the resulting versions into the training summary notes and into ISSUES.md, and get the Lead's yes first.

**A5. Install the CUDA build of torch.** The lockfile pins torch 2.14.0 from PyPI. On Windows that wheel is a **CPU build** (the Lead's laptop has `2.14.0+cpu`), which silently ignores the GPU; on Linux the PyPI wheel usually bundles CUDA, but I have not verified that, so let the preflight in A6 tell you which build you got and do this step only if it says CPU build. Open the official selector at <https://pytorch.org/get-started/locally/>, choose your OS, `Pip`, `Python`, and the CUDA version that matches the machine's driver (`nvidia-smi` shows the highest supported CUDA version), and run **the command it prints** inside the project environment. **[NEEDS LEAD'S YES]** (downloads about 2 GB).

This step is **outside the lockfile, and only P2 edits `uv.lock`**. So: do not commit anything for it, and record the **exact command you ran and the resulting `torch` version** (`python -c "import torch; print(torch.__version__, torch.version.cuda)"`) in `ISSUES.md` and in the `notes` of the training summary. `train_summary.json` already records `torch`, `torch_build` and the device name automatically.

I did not run this command; no CUDA machine was available when this guide was written.

**A6. Preflight.** Run it with `--require-cuda`: it exits with code 2 and a message if CUDA is not usable (usually a CPU-only torch from A4 that A5 did not replace).

```bash
python -m training.gpu_preflight --model rfdetr --batch-size 4 --n-train 470 --epochs 100 --require-cuda
```

It prints the GPU name and VRAM, CUDA availability, whether torch is a CPU or CUDA build, the torch and rfdetr versions, then times 5 forward+backward iterations of the real model on synthetic input and prints seconds per iteration, plus an **estimate** of seconds per epoch and total. The estimate leaves out validation, data loading, the optimizer and the loss, and the backward uses a surrogate loss; treat it as a lower bound for planning. Use the real `n_train` from `reports/dataset.json`. If the weights are not found it stops and says where it looked (exit code 3).

**A7. Fine-tune.**

```bash
python -m training.finetune --model rfdetr --dataset-dir data/dataset --output-dir out/rfdetr_run1 --seed 0 2>&1 | tee out_rfdetr_run1.log
```

(On Windows PowerShell use `| Tee-Object out_rfdetr_run1.log` instead of `tee`.) Defaults come from the number of train images: epochs 200 below 500 images, 100 for 500 to 1,999, 50 for 2,000 to 9,999 (the upper end of the Stage 6 ranges, early stopping ends the run sooner); learning rate 5e-5 below 1,000 images, else 1e-4; batch 4 x gradient accumulation 4 = effective 16. Override with `--epochs`, `--lr`, `--batch-size`, `--grad-accum`. A free or shared machine can stop early: rfdetr writes `last.ckpt` every epoch, so re-run the **same command with `--resume`** to continue. The script refuses to start (exit code 2) if the dataset class names differ from `config/experiment.json` in order, if a split folder, annotation file or image is missing, or if a run id is in two splits.

A 5-minute smoke test of the whole pipeline on synthetic data, before the real run: `python -m training.finetune --model rfdetr --output-dir out/smoke --dry-run`.

**A8. Copy back** `out/rfdetr_run1/detector.pth`, `out/rfdetr_run1/train_summary.json` and the log file to the Lead's laptop (for example into `weights/` and `reports/`; do not overwrite `weights/MANIFEST.json`, P1.5 writes that).

**A9. Verify the checksum** on the Lead's laptop; it must equal `detector_sha256` in `train_summary.json`:

```bash
python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" weights/detector.pth
```

**A10. Evaluate** (on the Lead's laptop, CPU is fine): `python -m training.eval_detector --model rfdetr --weights weights/detector.pth --dataset-dir data/dataset --split valid --out reports/detector_eval.json`. `--split test` is refused until `config/acceptance.yaml` exists.

---

## B. Kaggle (free GPU notebook)

**B1. Private dataset upload.** **[NEEDS LEAD'S YES]** (uploads project data to a third party; keep the dataset **private**). On kaggle.com choose Create, then New Dataset; upload the repo's `training/`, `config/`, `contracts.py`, `reports/dataset.json` and `data/dataset/` (zip them first; the zip is about 100 MB), plus `rf-detr-nano.pth` (366 MB). Set visibility to Private.

**B2. Create a GPU notebook.** **[NEEDS LEAD'S YES]** (account sign-in is the user's). New Notebook, Settings, Accelerator, **GPU T4** (T4 x2 also works; the script uses one GPU). Add the dataset from B1. Turn Internet **off** if the packages are already in the dataset, or on if you must install (that is a download and needs the yes).

**B3. The 5-minute smoke test first.** Kaggle's default PyTorch build is reported (March 2026) to lack kernels for the P100 (<https://github.com/Kaggle/docker-python/issues/1546>); the report on the T4 is conflicting and **its status today is unverified**. In the first cell:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m training.gpu_preflight --model rfdetr --batch-size 4 --n-train 470 --epochs 100 --require-cuda --weights /kaggle/input/<dataset>/rf-detr-nano.pth
```

If the preflight fails with a CUDA kernel error ("no kernel image is available"), stop: ask for a different accelerator or machine rather than debugging the build. Nothing on Kaggle has been run for this guide.

**B4. The same commands as path A.** In a notebook cell, `cd` into the copied repo, install per A4 and A5 (the notebook image may already carry a CUDA torch; check it with the line above and install nothing if it works), then run `training.finetune` exactly as in A7, with `--dataset-dir` pointing at the copied `data/dataset` and `--weights` at the uploaded checkpoint. Write outputs under `/kaggle/working/` (for example `--output-dir /kaggle/working/rfdetr_run1`), because only that folder is kept. A session can end early (weekly GPU quota, a 9-hour notebook limit; both unverified here), so use `--resume` to continue.

**B5. Download the outputs** `detector.pth`, `train_summary.json` and the log from the notebook's Output tab to the Lead's laptop, then verify the checksum as in A9. **[NEEDS LEAD'S YES]** (a download).

---

## YOLO11n fallback (either path)

Chosen once at P1.5, only if RF-DETR fails the acceptance thresholds or cannot be trained (AGENTS.md rule 15; YOLO11n is AGPL-3.0, so log the licence decision). Same steps with `--model yolo11n`: the preflight, then `python -m training.finetune --model yolo11n --dataset-dir data/dataset --output-dir out/yolo_run1 --weights <path to yolo11n.pt>`, which converts the COCO labels with `training.coco_to_yolo`, trains with Ultralytics (same seed, AdamW at the same lr, batch 16) and copies `best.pt` to `detector_yolo11n.pt`. Ultralytics sends anonymous usage analytics unless the machine's Ultralytics settings have `sync=False`; on a CUDA machine its automatic-mixed-precision check also tries to fetch a small `yolo11n.pt` and only warns if it cannot.

---

## Troubleshooting (things actually seen)

- **`--require-cuda: CUDA is not available (the installed torch is a CPU build ...)`.** The installed torch is a CPU build, as on the Lead's laptop (`2.14.0+cpu`). Do step A5; re-run the preflight.
- **`RF-DETR training dependencies are missing` / `rfdetr's training stack is not installed (missing: pytorch_lightning, torchmetrics, pycocotools)`.** The lockfile's `train` group lacks rfdetr's `[train]` extras (see the A4 note). Needs P2 and the Lead's yes.
- **`pretrained weights not found at ...`** (exit code 3). The weights cache was not copied. Copy `rf-detr-nano.pth` to `~/.roboflow/models/` or pass `--weights PATH`. Nothing is downloaded unless `--allow-download` is passed (RF-DETR only; needs the Lead's yes).
- **`refusing to train: category names [...] != experiment classes [...]`** (exit code 2). The dataset was built with another class order, or `config/experiment.json` changed after `build_dataset`. Rebuild the dataset with `python -m training.build_dataset`; never reorder categories by hand, because class ids 0 to 4 must stay in `experiment.classes` order (an off-by-one swaps red and yellow).
- **`run X appears in two splits`.** The dataset was assembled by hand. Rebuild it with `build_dataset`.
- **COCO ids.** rfdetr 1.11.0 accepts our 0-based category ids as they are (ids 0 to 4 become label indices 0 to 4, nothing is dropped); no placeholder category is needed. Pinned by `tests/unit/training/test_rfdetr_coco_ids.py`.
- **Out of memory on a small GPU.** Lower `--batch-size` and raise `--grad-accum` to keep the effective batch at 16 (for example `--batch-size 2 --grad-accum 8`).
- **`FileNotFoundError: --resume: no checkpoint` .** `--resume` needs `last.ckpt` or `checkpoint_N.ckpt` in the same `--output-dir` (YOLO: `yolo_train/weights/last.pt`).
- **Class ids in predictions.** The RF-DETR head has one spare output slot beyond the classes; a trained model never uses it, and `eval_detector` drops any detection outside `0..n-1` defensively.

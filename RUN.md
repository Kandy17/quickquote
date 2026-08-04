# QuickQuote segmentation model — how to run Phase 1

This folder is the complete Phase 1 pipeline. It could not run inside the
Cowork sandbox (no PyTorch, dataset host blocked), so you run it on your
machine or a Colab GPU. Nothing here contains a made-up number: every metric
comes out of files this pipeline writes.

## Option A — Claude Code (recommended)

Open this folder in Claude Code and paste:

> Run the QuickQuote Phase 1 pipeline in this folder per RUN.md: install
> requirements, download and prepare the data, run the smoke test, then the
> real training run, then evaluate and generate samples. Do not edit any
> metric by hand — metrics.json and training_log.csv are the only sources of
> truth. If something fails, fix the pipeline and note what failed and what
> changed in LIMITATIONS.md under "What broke during the build".

## Option B — manually

```bash
cd quickquote-model
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/download_data.py    # ~1.5 GB, resumable if interrupted
python src/prepare_data.py     # tiles + 3-class remap + image-level split
python src/train.py --smoke    # ~5 min sanity check: must reach "Done."
python src/train.py            # real run (see time estimates below)
python src/evaluate.py         # writes outputs/metrics.json
python src/infer.py --n 8      # writes outputs/samples/*.png + polygons
```

Colab GPU: upload this folder, `!pip install -r requirements.txt`, run the
same commands with `!python ...`. A T4 finishes training in well under an hour.
CPU-only laptop: expect several hours; lower `MAX_TILES` in `config.py`
(e.g. train 1500 / val 400 / test 400) if you need it faster — note the
reduction, it belongs in the write-up.

## What must exist on disk before Phase 2

- `outputs/training_log.csv` — per-epoch train/val loss + val mIoU
- `outputs/train_summary.json`
- `outputs/metrics.json` — per-class IoU/Dice on the held-out test set
- `outputs/samples/sample_*.png` — real screenshots for the portfolio
- `outputs/samples/polygons_*.json`
- `LIMITATIONS.md` — filled in with only what you actually observed

## Rules carried over from the master prompt

- If a number is not in `metrics.json` or `training_log.csv`, it does not
  appear in the README or the site. Period.
- Log failures as they happen (crashes, bad first results, fixes) — the
  "one real thing that didn't work" checkpoint question comes from here.
- Don't pre-fill LIMITATIONS.md with typical-sounding limitations.

## Bring back for Phase 2

Return with the `outputs/` folder and filled `LIMITATIONS.md` (paste contents
or reconnect the folder in Cowork). Also bring your answer on the pricing
engine: implemented today, or planned? The README will say whichever is true.

# QuickQuote — property-feature segmentation

I'm building QuickQuote, a B2B SaaS concept where a homeowner pulls up a
satellite view of their property, the relevant region (roof, lawn, driveway)
is highlighted automatically, they pick a service for that region, and they
get a quote. This repo is the computer vision core: a semantic segmentation
model that finds roofs and vegetation in aerial imagery and converts them
into polygon boundaries a frontend can highlight and measure.

## Why segmentation

Quotes scale with area. Object detection gives axis-aligned bounding boxes,
which overstate the area of anything that isn't a rectangle and can't be
rendered as a tight region highlight. Per-pixel segmentation follows the
actual roof/lawn outline, so the polygon is both the UI highlight and the
area estimate. OpenCV `findContours` bridges the two: class map in, polygons
out.

## Pipeline

```
LandCover.ai orthophotos
  → prepare_data.py   tile to 256×256, remap to 3 classes, 70/15/15 split
                      (split at source-image level to avoid spatial leakage)
  → train.py          U-Net, ImageNet-pretrained ResNet34 encoder,
                      Dice + cross-entropy loss, Adam @ 1e-4,
                      early stop on val-loss plateau
  → evaluate.py       per-class IoU / Dice on held-out test set → metrics.json
  → infer.py          class map → findContours → simplified polygons
                      + overlay visualizations
```

Design decisions I'd defend in a review:

- **Image-level split.** Tiles from one orthophoto share texture, season and
  lighting; splitting at the tile level leaks near-duplicates into val/test
  and inflates every metric. The 33 source images are split 70/15/15 first,
  then tiled.
- **3-class scope** (background / roof / vegetation). LandCover.ai also
  labels water and roads; v1 maps them to background. A working narrow model
  beats a broken wide one.
- **Dice + CE loss, IoU/Dice reporting.** Most pixels are background, so
  pixel accuracy rewards a model that finds nothing. Dice optimizes region
  overlap per class; cross-entropy stabilizes optimization.
- **Pretrained encoder.** With a few thousand tiles, ImageNet features are
  adapted (all layers trainable at lr 1e-4) rather than learned from scratch.

## Results

**Status: training run pending.** This repo's rule is that no metric appears
here unless it exists in `outputs/metrics.json`, written by `evaluate.py`
against the held-out test set. That run hasn't happened yet, so there are no
numbers to show. When it lands, this section gets per-class IoU/Dice, the
training curve, and sample predictions from `outputs/samples/`.

## Project status

| Component | Status |
|---|---|
| Segmentation pipeline (data → train → eval → polygons) | Code complete, training run pending |
| Polygon extraction (class map → contours → simplified polygons) | Code complete, pending trained weights |
| Frontend (satellite view, region highlight, service selection) | Planned |
| Comp-based pricing engine (location + neighbor spend) | Planned |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/download_data.py    # LandCover.ai v1, ~1.5 GB, resumable
python src/prepare_data.py     # tiles + split
python src/train.py --smoke    # 2-epoch pipeline sanity check
python src/train.py            # full training run
python src/evaluate.py         # writes outputs/metrics.json
python src/infer.py --n 8      # writes outputs/samples/
```

GPU recommended (a Colab T4 trains in well under an hour); CPU works but
takes hours. Knobs live in `config.py`.

## Repo map

```
config.py              every tunable, each with the reasoning next to it
src/download_data.py   resumable dataset download + extraction
src/prepare_data.py    tiling, class remap, leakage-safe split
src/dataset.py         Dataset + albumentations pipelines
src/train.py           training loop, CSV logging, checkpoints, early stop
src/evaluate.py        held-out IoU/Dice → metrics.json (sole source of truth)
src/infer.py           inference → contours → polygons + overlays
LIMITATIONS.md         only limitations actually observed, with evidence
CONCEPTS.md            the design reasoning, written to be explainable
```

## Limitations

Tracked in [LIMITATIONS.md](LIMITATIONS.md), populated only from observed
behavior. One is known by design already: training imagery is Polish aerial
orthophotography (25/50 cm), while production would consume maps-API
satellite tiles — that domain gap is real and untested.

## License

MIT — see [LICENSE](LICENSE).

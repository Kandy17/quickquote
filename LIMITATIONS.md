# Observed limitations

Everything below was observed in the training run of 2026-08-04 (T4 GPU,
25 epochs, best checkpoint epoch 17). Each item cites the file or image it
came from. Nothing here is a generic limitation copied from elsewhere.

## 1. Roofs are under-segmented — the model's main failure mode

`outputs/metrics.json`, confusion matrix: **425,666 true roof pixels (20.5%
of all roof pixels in the test set) were predicted as background.** Only
5,453 roof pixels were confused with vegetation. So the error isn't
roof-vs-vegetation ambiguity — it's roofs being missed entirely.

Per-class breakdown on the 800 held-out test tiles:

| Class | IoU | Dice | Recall | Precision |
|---|---|---|---|---|
| background | 0.8494 | 0.9186 | 0.9436 | 0.8949 |
| roof | 0.7043 | 0.8265 | 0.7928 | 0.8632 |
| vegetation | 0.8095 | 0.8947 | 0.8641 | 0.9277 |

Roof recall (0.793) is well below roof precision (0.863): when the model
calls something a roof it's usually right, but it misses about a fifth of
actual roof area. For QuickQuote that biases quotes **low**, which matters
— an under-measured roof is an under-priced job.

## 2. Large flat industrial roofs get missed; the speck filter then yields zero polygons

`outputs/samples/sample_02.png` is the clearest case. The tile is an
industrial/rail area of long flat-roofed structures whose colour and texture
closely match the adjacent paved surfaces. The model predicted only a single
small red patch, and because that patch fell under the
`MIN_CONTOUR_AREA_PX = 100` threshold in `config.py`, **contour extraction
produced 0 polygons** — a tile that is mostly building yields nothing to
highlight or quote.

Two distinct problems stacked here: (a) flat roofs that look like pavement
are genuinely hard for the model, and (b) the speck filter converts a weak
detection into no detection at all, with no signal that it happened.

By contrast `outputs/samples/sample_05.png` — pitched residential roofs with
clear colour separation from surroundings — produces 6 clean polygons whose
boundaries track the actual roof outlines. Pitched suburban roofs, the
QuickQuote target case, work substantially better than industrial ones.

## 3. Mild overfitting from ~epoch 10

`outputs/training_log.csv`: train loss falls monotonically 1.042 → 0.530
across 25 epochs, while validation loss flattens into a 0.68–0.72 band from
epoch 10 and never improves on epoch 17's 0.6784. Validation mIoU likewise
plateaus around 0.82–0.83. Training early-stopped at epoch 25 after 8 epochs
without improvement. The gap between the two curves is the overfitting; the
ReduceLROnPlateau drops (1e-4 → 5e-5 at epoch 15 → 2.5e-5 at epoch 22)
bought small gains but did not close it. More training data or stronger
regularization is the next thing to try, not more epochs.

## 4. Vegetation is also under-segmented, though less severely

13.6% of true vegetation pixels (2,838,369) were predicted as background.
Same directional bias as roofs, smaller in relative terms.

## Known gaps by design (true without needing a run)

- Trained on LandCover.ai orthophotos of Poland (25/50 cm aerial imagery).
  Production QuickQuote would consume a maps-API satellite tile — different
  sensor, resolution, geography, and season. This domain gap is real and
  **untested**; no claim is made about performance on maps-API imagery.
- 3 classes only (background / roof / vegetation). Water and roads are
  mapped to background. No driveway, pool, or lawn-vs-woodland distinction.
- Pricing is a deterministic area-based rate card (`src/pricing.py`). The
  comp-based component — adjusting by neighbour spend — is an explicit stub
  returning 1.0, because no transaction data exists to learn from.

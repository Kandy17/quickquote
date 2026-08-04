# Observed limitations

Rule: only write what you actually observed during this build. No
typical-sounding limitations copied from blog posts. Each entry needs the
evidence next to it (a number from metrics.json/training_log.csv, or a
specific sample image that shows it).

## Model / data limitations (fill during & after training)

<!-- Where to look:
  - training_log.csv: did train loss keep dropping while val loss flattened
    or rose? That's observed overfitting — note the epoch.
  - metrics.json: per-class IoU gap (e.g. roof much lower than vegetation)
    is an observed weakness on small/rare structures.
  - samples/*.png: look for missed small buildings, shadow confusion,
    ragged boundaries, false positives on driveways/roads. Only note what
    you can point to in a specific sample image.
-->

- (none recorded yet)

## What broke during the build

<!-- Crashes, wrong first attempts, fixes. This feeds checkpoint question 6. -->

- (none recorded yet)

## Known gaps by design (true today, no run needed)

- Trained on LandCover.ai orthophotos of Poland (25/50 cm rural/urban
  aerial imagery). Production QuickQuote would consume a maps API's
  satellite tiles — different sensor, resolution, geography, and look. This
  domain gap is expected and untested until we run on maps-API imagery.
- 3 classes only (background/roof/vegetation). No driveway, pool, or
  lawn-vs-woodland distinction yet.
- Pricing engine status: [confirm with Ani before writing anywhere].

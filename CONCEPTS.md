# Interview prep — the six checkpoint concepts

Read this, then explain each back in your own words (in chat, before Phase 2).
Items 5 and 6 can only be answered after your training run.

## 1. Why segmentation, not object detection or manual polygon drawing

Object detection outputs bounding boxes — axis-aligned rectangles. A roof is
rarely an axis-aligned rectangle, so a box overstates area (bad for quoting,
where price scales with area) and can't be rendered as a tight highlight on
the satellite view. Semantic segmentation classifies **every pixel**, so the
region boundary follows the actual roof/lawn outline: accurate area, precise
highlight. Manual polygon drawing works but doesn't scale and kills the UX —
the whole product moment is "your roof lights up automatically."

## 2. Why U-Net, and what skip connections do

U-Net is an encoder–decoder. The **encoder** repeatedly downsamples,
trading spatial resolution for semantic understanding ("this region is
building-ish"). The **decoder** upsamples back to full resolution to make a
per-pixel prediction. Problem: by the bottleneck, precise edge positions are
gone. **Skip connections** copy feature maps from each encoder stage directly
across to the matching decoder stage, so the decoder gets both the semantics
(from below) and the fine spatial detail (from the skip). That's why U-Net
predictions have crisp boundaries — exactly what polygon extraction needs.

## 3. Why a pretrained encoder, and what fine-tuning means here

The encoder is a ResNet34 whose weights were already trained on ImageNet
(1.2M+ labeled photos). Its early/mid layers learned generic visual features —
edges, textures, shapes — that transfer to aerial imagery. With only a few
thousand training tiles, learning those features from random weights would
overfit or underperform; starting from ImageNet weights means we only need to
*adapt* them. Fine-tuning here concretely: all layers stay trainable, but at
a small learning rate (1e-4), so the pretrained features shift gently toward
aerial imagery instead of being destroyed and relearned.

## 4. The loss, and why pixel accuracy would mislead

Loss = Dice loss + cross-entropy.

- Most pixels are background. A model that predicts "background everywhere"
  can score 90%+ **pixel accuracy** while finding zero roofs — accuracy is the
  wrong yardstick under class imbalance.
- **Dice loss** optimizes overlap between predicted and true regions per
  class (2·|A∩B| / (|A|+|B|)), so rare classes like roofs count fully.
- **Cross-entropy** supplies smooth, well-behaved per-pixel gradients that
  make optimization stable, especially early.
- Same logic for evaluation: we report **IoU** (intersection over union) and
  **Dice** per class, never pixel accuracy.

## 5. Your actual numbers — fill in after the run

From `outputs/metrics.json` only. Know: roof IoU, vegetation IoU, mean IoU,
and what they mean (IoU 0.5 = predicted and true regions overlap by half of
their union; 0.5–0.7 is a reasonable first pass, 0.8+ is strong). Expect
roof IoU below vegetation IoU — roofs are small, rare, and high-detail.

## 6. One real thing that didn't work — fill in after the run

Comes from your actual run: a crash, a first model that missed small
buildings, a loss that plateaued too early, a bad tile filter — whatever
actually happened, plus what changed. Recorded in LIMITATIONS.md as it
happens. Do not invent one.

## Bonus decisions worth being able to defend

- **Image-level split**: tiles from the same orthophoto share texture,
  season, lighting. Splitting at the tile level would leak near-duplicates
  into val/test and inflate metrics. We split the 33 source images 70/15/15
  first, then tile. Held-out evaluation is the only reason the metrics mean
  anything: numbers measured on data the model trained on measure
  memorization, not capability.
- **3-class scope**: LandCover.ai has water (and roads); QuickQuote v1 maps
  them to background. A working narrow model beats a broken wide one, and
  roof + vegetation covers the core quoting services.
- **Augmentation choice**: flips/90° rotations are safe because aerial
  imagery has no canonical orientation; the same transforms applied to a
  portrait photo dataset would be riskier.

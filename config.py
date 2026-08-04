"""Central config for the QuickQuote segmentation pipeline.

Every value here is a real design decision. If you change one, know why.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"                  # raw LandCover.ai download
RAW_DIR = DATA_DIR / "raw"
TILES_DIR = DATA_DIR / "tiles"            # prepared 256x256 tiles
OUTPUTS_DIR = ROOT / "outputs"            # logs, metrics, checkpoints, samples
CKPT_DIR = OUTPUTS_DIR / "checkpoints"
SAMPLES_DIR = OUTPUTS_DIR / "samples"

DATASET_URL = "https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip"

# --- Classes ---------------------------------------------------------------
# LandCover.ai labels: 0=background, 1=building, 2=woodland, 3=water (v1 also
# ships 4=road in some releases). We scope v1 of QuickQuote to 3 classes:
# a working narrow model beats a broken wide one.
# Remap: building -> roof(1), woodland -> vegetation(2), everything else -> 0.
CLASS_NAMES = ["background", "roof", "vegetation"]
NUM_CLASSES = 3

# --- Tiling ----------------------------------------------------------------
TILE_SIZE = 256
TILE_STRIDE = 256   # no overlap; keeps tiles independent

# Tile filtering: raw tiling produces tens of thousands of tiles, most of them
# pure background. Keep a tile if it has meaningful roof or vegetation
# content, plus a random slice of background-only tiles so the model still
# sees "nothing here" examples.
MIN_ROOF_FRAC = 0.002       # >=0.2% roof pixels
MIN_VEG_FRAC = 0.05         # >=5% vegetation pixels
BG_KEEP_PROB = 0.10         # keep 10% of tiles that pass neither threshold

# Caps keep a CPU run feasible; raise them on a GPU if you want.
MAX_TILES = {"train": 4000, "val": 800, "test": 800}

# --- Split -----------------------------------------------------------------
# Split at the SOURCE IMAGE level (70/15/15 over the 33 orthophotos), not the
# tile level. Neighboring tiles from one orthophoto are nearly identical in
# texture/season/lighting; a tile-level random split would leak that into
# val/test and inflate every metric. This is a spatial-leakage decision worth
# explaining in an interview.
SPLIT_FRACS = (0.70, 0.15, 0.15)
SEED = 42

# --- Model / training --------------------------------------------------------
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
LR = 1e-4
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 8     # stop when val loss hasn't improved for 8 epochs
PLATEAU_PATIENCE = 4        # halve LR after 4 epochs without improvement
BATCH_SIZE = {"cuda": 16, "mps": 8, "cpu": 4}
NUM_WORKERS = 2

# ImageNet normalization — must match the pretrained encoder's expectations.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Inference ---------------------------------------------------------------
MIN_CONTOUR_AREA_PX = 100   # drop speck polygons below this pixel area
POLY_EPSILON_FRAC = 0.01    # approxPolyDP epsilon as fraction of perimeter

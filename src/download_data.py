"""Download and unzip LandCover.ai v1 (~1.5 GB). Resumable.

Usage: python src/download_data.py
"""
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

CHUNK = 1 << 20  # 1 MiB


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"User-Agent": "quickquote-pipeline"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"Resuming at {existing / 1e6:.0f} MB")
    req = Request(url, headers=headers)
    with urlopen(req) as resp:
        if existing and resp.status == 200:
            # Server ignored Range; start over.
            existing = 0
        total = existing + int(resp.headers.get("Content-Length", 0))
        mode = "ab" if existing else "wb"
        done = existing
        with open(dest, mode) as f:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                print(f"\r{done / 1e6:.0f} / {total / 1e6:.0f} MB", end="")
    print("\nDownload complete.")


def main() -> None:
    zip_path = config.RAW_DIR / "landcover.ai.v1.zip"
    marker = config.RAW_DIR / "images"
    if marker.exists() and any(marker.iterdir()):
        print("Dataset already extracted, skipping.")
        return
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000_000:
        download(config.DATASET_URL, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(config.RAW_DIR)
    n_img = len(list((config.RAW_DIR / "images").glob("*.tif")))
    n_msk = len(list((config.RAW_DIR / "masks").glob("*.tif")))
    print(f"Extracted: {n_img} images, {n_msk} masks.")
    if n_img == 0:
        sys.exit("ERROR: no images found after extraction — inspect the zip.")


if __name__ == "__main__":
    main()

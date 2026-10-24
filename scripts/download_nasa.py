"""Download NASA PCoE Li-ion battery aging .mat files (B0005/B0006/B0007)."""

from __future__ import annotations

import argparse
import zipfile
from io import BytesIO
from pathlib import Path

import requests
from tqdm import tqdm

# Mirrors used by open-source battery SOH projects
NASA_ZIP_URLS = [
    "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip",
    "https://phm-datasets.s3.amazonaws.com/NASA/4.+Battery.zip",
]

BATTERY_IDS = ["B0005", "B0006", "B0007"]


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def extract_batteries_from_zip(zip_bytes: bytes, out_dir: Path) -> list[str]:
    extracted = []
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            stem = Path(name).stem.upper()
            if stem in BATTERY_IDS and name.lower().endswith(".mat"):
                out_path = out_dir / f"{stem}.mat"
                out_path.write_bytes(zf.read(name))
                extracted.append(stem)
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NASA battery .mat files")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/nasa"),
        help="Output directory for .mat files",
    )
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [b for b in BATTERY_IDS if not (out_dir / f"{b}.mat").exists()]
    if not missing:
        print(f"All batteries already present in {out_dir}")
        return

    print(f"Downloading NASA battery data to {out_dir} ...")
    last_err = None
    for url in NASA_ZIP_URLS:
        try:
            print(f"Trying {url}")
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()
            extracted = extract_batteries_from_zip(resp.content, out_dir)
            if extracted:
                print(f"Extracted: {sorted(set(extracted))}")
                missing = [b for b in BATTERY_IDS if not (out_dir / f"{b}.mat").exists()]
                if not missing:
                    print("Download complete.")
                    return
        except Exception as exc:
            last_err = exc
            print(f"  failed: {exc}")

    # Fallback: GitHub mirrors that host the .mat files
    direct_urls = {
        "B0005": [
            "https://raw.githubusercontent.com/vicky0718/Battery-SOH/main/B0005.mat",
            "https://github.com/yuva-py/Smart-Battery-Health-Tracker-RUL-Predictor/raw/main/B0005.mat",
        ],
        "B0006": [
            "https://raw.githubusercontent.com/vicky0718/Battery-SOH/main/B0006.mat",
        ],
        "B0007": [
            "https://raw.githubusercontent.com/vicky0718/Battery-SOH/main/B0007.mat",
            "https://github.com/yuva-py/Smart-Battery-Health-Tracker-RUL-Predictor/raw/main/B0007.mat",
        ],
    }
    for bid in missing:
        for url in direct_urls.get(bid, []):
            try:
                print(f"Direct download {bid} from {url}")
                download_file(url, out_dir / f"{bid}.mat")
                if (out_dir / f"{bid}.mat").stat().st_size > 1_000_000:
                    break
            except Exception as exc:
                last_err = exc
                print(f"  failed: {exc}")

    still_missing = [b for b in BATTERY_IDS if not (out_dir / f"{b}.mat").exists()]
    if still_missing:
        raise RuntimeError(
            f"Could not download {still_missing}. Last error: {last_err}\n"
            "Manual: place B0005.mat, B0006.mat, B0007.mat under data/nasa/"
        )
    print("Download complete.")


if __name__ == "__main__":
    main()

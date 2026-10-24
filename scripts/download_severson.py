"""Download Severson 2019 data.

Default source is Hugging Face via hf-mirror.com (IPv4), processed tar.gz.
Optional --source matr downloads the three official .mat batches from data.matr.io.

Citation: Severson et al., Nature Energy 4, 383-391 (2019), CC-BY-4.0.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen


def force_ipv4() -> None:
    """huggingface.co IPv6 is refused on this host; matr/hf-mirror work on IPv4."""
    real = socket.getaddrinfo

    def ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
        return real(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = ipv4_only


force_ipv4()

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data" / "severson"

MATR_FILES = [
    {
        "name": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
        "url": "https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download",
        "bytes": 3_025_320_241,
    },
    {
        "name": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
        "url": "https://data.matr.io/1/api/v1/file/5c86bf13fa2ede00015ddd82/download",
        "bytes": 2_007_331_155,
    },
    {
        "name": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
        "url": "https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download",
        "bytes": 3_236_690_412,
    },
]

HF_MIRROR_FILES = [
    {
        "name": "severson_processed_20260807.tar.gz",
        "url": "https://hf-mirror.com/datasets/fryabcfry/severson-2019-processed/resolve/main/severson_processed_20260807.tar.gz",
        "bytes": 3_587_570_430,
    },
]

UA = "tcn-attention-research/1.0 (academic; jjp@mails.guet.edu.cn)"
PRINT_LOCK = threading.Lock()


def log(msg: str) -> None:
    with PRINT_LOCK:
        print(msg, flush=True)


def fetch_range(url: str, start: int, end: int, dest: Path, chunk: int = 1 << 20, retries: int = 40) -> None:
    need = end - start + 1
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have > need:
            dest.unlink()
            have = 0
        if have == need:
            return
        abs_start = start + have
        req = Request(url, headers={"User-Agent": UA, "Range": f"bytes={abs_start}-{end}"})
        try:
            with urlopen(req, timeout=180) as resp:
                with open(dest, "ab") as f:
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
        except Exception as exc:
            last_err = exc
            wait = min(30, 2 * attempt)
            log(f"  retry {attempt}/{retries} {dest.name}: {exc}; sleep {wait}s")
            time.sleep(wait)
            continue
        got = dest.stat().st_size
        if got == need:
            return
        last_err = RuntimeError(f"{dest.name}: got {got}, expected {need}")
        wait = min(30, 2 * attempt)
        log(f"  retry {attempt}/{retries} {last_err}; sleep {wait}s")
        time.sleep(wait)
    raise last_err or RuntimeError(f"{dest.name}: range download failed")


def split_remaining(prefix: int, total: int, workers: int) -> list[tuple[int, int]]:
    lo = prefix
    span = total - lo
    if span <= 0:
        return []
    n = max(1, min(workers, span // (8 << 20) or 1))
    return [
        (lo + span * i // n, lo + span * (i + 1) // n - 1)
        for i in range(n)
    ]


def copy_bytes(src: Path, out, nbytes: int | None = None) -> None:
    remaining = nbytes
    with open(src, "rb") as inp:
        while remaining is None or remaining > 0:
            n = 8 << 20
            if remaining is not None:
                n = min(n, remaining)
            buf = inp.read(n)
            if not buf:
                break
            out.write(buf)
            if remaining is not None:
                remaining -= len(buf)
    if remaining:
        raise RuntimeError(f"{src.name}: short prefix, missing {remaining} bytes")


def assemble(
    dest: Path,
    prefix_path: Path | None,
    prefix_len: int,
    segments: list[Path],
    expected: int,
) -> None:
    tmp = dest.with_suffix(dest.suffix + ".joining")
    with open(tmp, "wb") as out:
        if prefix_path is not None and prefix_len:
            copy_bytes(prefix_path, out, prefix_len)
        for seg in segments:
            copy_bytes(seg, out)
    if tmp.stat().st_size != expected:
        raise RuntimeError(f"{dest.name}: joined {tmp.stat().st_size}, expected {expected}")
    tmp.replace(dest)
    if prefix_path is not None and prefix_path.exists():
        prefix_path.unlink()
    for seg in segments:
        if seg.exists():
            seg.unlink()


def download_one(item: dict, dest_dir: Path, workers: int) -> Path:
    dest = dest_dir / item["name"]
    expected = int(item["bytes"])
    if dest.exists() and dest.stat().st_size == expected:
        log(f"OK {dest.name}")
        return dest
    prefix_path = dest.with_suffix(dest.suffix + ".part")
    prefix = prefix_path.stat().st_size if prefix_path.exists() else 0
    if prefix > expected:
        prefix_path.unlink()
        prefix = 0
    ranges = split_remaining(prefix, expected, workers)
    log(f"GET {dest.name} prefix={prefix} ranges={len(ranges)} workers={workers}")
    if not ranges:
        assemble(dest, prefix_path, prefix, [], expected)
        log(f"DONE {dest.name}")
        return dest
    seg_paths = []
    jobs = []
    with ThreadPoolExecutor(max_workers=max(1, len(ranges))) as ex:
        for a, b in ranges:
            seg = dest_dir / f"{item['name']}.seg{a}-{b}"
            seg_paths.append(seg)
            jobs.append(ex.submit(fetch_range, item["url"], a, b, seg))
        for fut in as_completed(jobs):
            fut.result()
            done = prefix + sum(p.stat().st_size for p in seg_paths if p.exists())
            log(f"  {dest.name}: {done / 1e9:.2f} / {expected / 1e9:.2f} GB")
    assemble(dest, prefix_path if prefix else None, prefix, seg_paths, expected)
    log(f"DONE {dest.name}")
    return dest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--source", choices=("hf-mirror", "matr"), default="hf-mirror")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = HF_MIRROR_FILES if args.source == "hf-mirror" else MATR_FILES
    lock = args.out_dir / "download.lock"
    try:
        lock_fd = lock.open("x")
    except FileExistsError:
        raise SystemExit(f"Another download holds {lock}. Delete it only if no download is running.")
    lock_fd.write(str(os_getpid()))
    lock_fd.close()
    try:
        paths = []
        for item in files:
            paths.append(download_one(item, args.out_dir, args.workers))
        src = (
            "https://hf-mirror.com/datasets/fryabcfry/severson-2019-processed"
            if args.source == "hf-mirror"
            else "https://data.matr.io/1/projects/5c48dd2bc625d700019f3204"
        )
        manifest = {
            "source": src,
            "mirror": args.source,
            "citation": "Severson et al., Nature Energy 4, 383-391 (2019)",
            "license": "CC-BY-4.0",
            "files": [{"name": x.name, "bytes": x.stat().st_size} for x in paths],
            "note": "T_prior must come from D_train siblings, never median(T_all).",
        }
        man = args.out_dir / "download_manifest.json"
        man.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log(f"Wrote {man}")
    finally:
        if lock.exists():
            lock.unlink()


def os_getpid() -> int:
    return os.getpid()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise

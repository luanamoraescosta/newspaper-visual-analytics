#!/usr/bin/env python3
"""Run Tesseract OCR on the gallery crops and store the text in the dashboard data.

The crop files are named ``<detection_id>.webp`` inside ``dashboard_assets/crops``.
The extracted text is written per image to ``ocr_text_cache.json`` (id -> text)
and merged into every gallery item of ``dashboard_data.json`` as ``ocr_text``.

Run once offline::

    python ocr_crops.py

The cache makes re-runs cheap: images already processed are skipped. Results can
be merged without re-running the OCR with ``--merge-only``.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


ROOT = Path(__file__).resolve().parent
CROPS_DIR = ROOT / "dashboard_assets" / "crops"
DATA_PATH = ROOT / "dashboard_data.json"
CACHE_PATH = ROOT / "ocr_text_cache.json"

LANG = "por"
WORKERS = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Do not run OCR; only merge the existing cache into the data.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help="Number of parallel OCR workers.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N crops (0 = all). For testing.",
    )
    return parser.parse_args()


def ocr_image(path):
    """Run Tesseract on a single image and return the recognised text."""
    try:
        result = subprocess.run(
            [
                "tesseract",
                str(path),
                "stdout",
                "-l",
                LANG,
            ],
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def run_ocr(cache, limit):
    files = sorted(CROPS_DIR.glob("*.webp"))
    if limit and limit > 0:
        files = files[:limit]

    todo = [
        f
        for f in files
        if f.stem not in cache
    ]

    print(
        f"{len(files)} crops, {len(todo)} to OCR, "
        f"{len(files) - len(todo)} already cached."
    )

    if not todo:
        return

    with ProcessPoolExecutor(
        max_workers=args_workers
    ) as pool:
        for path, text in zip(
            todo,
            tqdm(
                pool.map(ocr_image, todo),
                total=len(todo),
                desc="OCR",
            ),
        ):
            cache[path.stem] = text
            # Persist progressively so a long run can be interrupted.
            if len(cache) % 100 == 0:
                CACHE_PATH.write_text(
                    json.dumps(
                        cache,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )


def merge_into_data(cache):
    if not DATA_PATH.exists():
        print("dashboard_data.json not found.", file=sys.stderr)
        return

    data = json.loads(
        DATA_PATH.read_text(encoding="utf-8")
    )

    gallery = data.get("gallery", [])
    with_ocr = 0
    for item in gallery:
        text = cache.get(
            str(item.get("id")),
            "",
        )
        if text:
            with_ocr += 1
        item["ocr_text"] = text

    DATA_PATH.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Merged ocr_text into {len(gallery)} gallery items "
        f"({with_ocr} with text)."
    )


args_workers = WORKERS


def main():
    global args_workers
    args = parse_args()
    args_workers = args.workers

    cache = {}
    if CACHE_PATH.exists():
        cache = json.loads(
            CACHE_PATH.read_text(encoding="utf-8")
        )

    if not args.merge_only:
        run_ocr(cache, args.limit)
        CACHE_PATH.write_text(
            json.dumps(
                cache,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    merge_into_data(cache)


if __name__ == "__main__":
    main()

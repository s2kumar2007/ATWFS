"""Build leftImg8bit/{train,val,test} + gtFine/{train,val,test} from data/splits/*.txt.

The IDD_RESIZED archive contains 512x512 images that were *squashed* from a 16:9
camera frame.  At inference (inference/live.py) the pipeline reads a native 16:9
frame and processes it at 288x512 without letterboxing (scale-and-stretch).  To
match that at training time the masks must also be built from the squashed source,
and the dataset class must stretch (not letterbox) 512x512 images to 288x512.

Label collapsing mirrors convert_idd.py:
  drivable (mask == drivable_value) -> level-3 id 0
  everything else                   -> level-3 id 3

Usage
-----
  python scripts/build_dataset_from_splits.py \\
      --src "C:/path/to/IDD_RESIZED" \\
      --dst data/idd_converted \\
      --drivable-value 1

The script writes data/splits/name_mapping.json with
  { "Image_N.png": "d<drive>/<stem>_leftImg8bit.png" }
for use by the class-balanced sampler.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert IDD_RESIZED archive to IDD training layout using scene-disjoint splits."
    )
    ap.add_argument(
        "--src",
        required=True,
        help="Root of IDD_RESIZED: must contain image_archive/ and mask_archive/",
    )
    ap.add_argument(
        "--dst",
        default="data/idd_converted",
        help="Destination root (default: data/idd_converted)",
    )
    ap.add_argument(
        "--splits-dir",
        default="data/splits",
        help="Directory containing train.txt / val.txt / test.txt (default: data/splits)",
    )
    ap.add_argument(
        "--drivable-value",
        type=int,
        default=1,
        help="Mask pixel value that represents drivable-fallback (default: 1)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing destination files.",
    )
    return ap.parse_args()


def read_split(splits_dir: Path, split: str) -> list[str]:
    """Return list of Image_N.png filenames for this split."""
    p = splits_dir / f"{split}.txt"
    if not p.exists():
        raise FileNotFoundError(
            f"Split file not found: {p}\n"
            f"Run scripts/build_dataset_from_splits.py AFTER generating data/splits/*.txt."
        )
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> None:
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        sys.exit("Pillow is required: pip install Pillow")

    a = parse_args()
    src = Path(a.src)
    dst = Path(a.dst)
    splits_dir = Path(a.splits_dir)

    img_arch = src / "image_archive"
    mask_arch = src / "mask_archive"
    if not img_arch.is_dir():
        sys.exit(f"ERROR: {img_arch} not found. Check --src.")
    if not mask_arch.is_dir():
        sys.exit(f"ERROR: {mask_arch} not found. Check --src.")

    splits = {s: read_split(splits_dir, s) for s in ("train", "val", "test")}
    total = sum(len(v) for v in splits.values())
    logger.info(
        "Splits: train=%d  val=%d  test=%d  total=%d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
        total,
    )

    # Build a mapping Image_N.png -> (split, drive_folder, stem)
    # We assign each Image_N to drive d<block_index:04d> where block_index
    # is the position of the image within its split (groups of 20 consecutive
    # images share a drive, matching the idd_converted convention).
    name_mapping: dict[str, str] = {}  # Image_N.png -> "d0000/000000_leftImg8bit.png"
    processed = 0

    for split, fnames in splits.items():
        for block_start in range(0, len(fnames), 20):
            drive_idx = block_start // 20
            drive = f"d{drive_idx:04d}"
            block = fnames[block_start : block_start + 20]
            for fname in block:
                img_id_str = fname.replace("Image_", "").replace(".png", "")
                try:
                    img_id = int(img_id_str)
                except ValueError:
                    logger.warning("Unexpected filename format: %s – skipping.", fname)
                    continue
                stem = f"{img_id:06d}"
                src_img = img_arch / fname
                src_mask = mask_arch / f"Mask_{img_id}.png"
                if not src_img.exists():
                    logger.warning("Missing image: %s – skipping.", src_img)
                    continue
                if not src_mask.exists():
                    logger.warning("Missing mask: %s – skipping.", src_mask)
                    continue

                dst_img_dir = dst / "leftImg8bit" / split / drive
                dst_gt_dir = dst / "gtFine" / split / drive
                dst_img_dir.mkdir(parents=True, exist_ok=True)
                dst_gt_dir.mkdir(parents=True, exist_ok=True)

                dst_img = dst_img_dir / f"{stem}_leftImg8bit.png"
                dst_mask_path = dst_gt_dir / f"{stem}_gtFine_labellevel3Ids.png"

                if not a.force and dst_img.exists() and dst_mask_path.exists():
                    pass  # already done
                else:
                    shutil.copy2(src_img, dst_img)
                    # Collapse mask
                    m = np.array(Image.open(src_mask))
                    out = np.where(m == a.drivable_value, 0, 3).astype(np.uint8)
                    Image.fromarray(out).save(dst_mask_path)

                name_mapping[fname] = f"{drive}/{stem}_leftImg8bit.png"
                processed += 1
        logger.info("  %s: %d files processed", split, len(fnames))

    # Save mapping for use by the class-balanced sampler
    mapping_path = splits_dir / "name_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(name_mapping, f, indent=2)
    logger.info("Saved name mapping to %s (%d entries)", mapping_path, len(name_mapping))
    logger.info("Done.  Processed %d/%d images.  Destination: %s", processed, total, dst)


if __name__ == "__main__":
    main()

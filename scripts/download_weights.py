"""Download pre-trained checkpoints (teacher_best.pt, student_best.pt, fusionnet.pt) from GitHub Releases.

Reads ``paths.weights_url`` from ``config.yaml`` (or ``--url`` CLI flag).
If no URL is configured and checkpoints are missing, prints:
  ``NEEDS USER DATA: Set paths.weights_url in config.yaml (or pass --url)``
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config, resolve_path
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_FILES = ["teacher_best.pt", "student_best.pt"]


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_weights(
    config_path: Optional[str] = None,
    base_url: Optional[str] = None,
    checkpoint_dir: Optional[str | Path] = None,
    files: Optional[List[str]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Download checkpoint files from ``base_url`` or ``cfg['paths']['weights_url']``."""
    cfg = load_config(config_path)
    url = (base_url if base_url is not None else cfg.get("paths", {}).get("weights_url", "")).strip()
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else resolve_path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target_files = files or DEFAULT_FILES

    existing = [f for f in target_files if (ckpt_dir / f).is_file() and (ckpt_dir / f).stat().st_size > 0]
    if len(existing) == len(target_files) and not force and not url:
        logger.info("All requested checkpoints already exist in %s: %s", ckpt_dir, existing)
        return {"status": "exists", "files": [str(ckpt_dir / f) for f in existing]}

    if not url or "TODO" in url.upper():
        print("NEEDS USER DATA: Set paths.weights_url in config.yaml (or pass --url) to download release weights.")
        return {"status": "needs_user_data", "files": [str(ckpt_dir / f) for f in existing]}

    downloaded: List[str] = []
    for fname in target_files:
        dest = ckpt_dir / fname
        if dest.exists() and dest.stat().st_size > 0 and not force:
            logger.info("Skipping %s (already exists)", dest)
            downloaded.append(str(dest))
            continue

        file_url = f"{url.rstrip('/')}/{fname}"
        logger.info("Downloading %s -> %s", file_url, dest)
        try:
            urllib.request.urlretrieve(file_url, str(dest))
            downloaded.append(str(dest))
        except Exception as exc:
            logger.error("Failed to download %s: %s", file_url, exc)
            print(f"NEEDS USER DATA: Valid reachable URL for {fname} (failed: {exc})")
            return {"status": "error", "error": str(exc), "files": downloaded}

    return {"status": "downloaded", "files": downloaded}


def main() -> None:
    ap = argparse.ArgumentParser(description="Download model checkpoints from GitHub Releases.")
    ap.add_argument("--config", default=None, help="Path to config.yaml")
    ap.add_argument("--url", default=None, help="Base URL for release assets")
    ap.add_argument("--out-dir", default=None, help="Target checkpoint directory")
    ap.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = ap.parse_args()

    setup_logging()
    download_weights(config_path=args.config, base_url=args.url, checkpoint_dir=args.out_dir, force=args.force)


if __name__ == "__main__":
    main()

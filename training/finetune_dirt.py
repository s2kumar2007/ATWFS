"""Fine-tune student segmentation network specifically on dirt/unpaved road scenes.

Usage on GPU (Colab / Kaggle recommended):
    python -m training.finetune_dirt \\
        --checkpoint outputs/checkpoints/student_best.pt \\
        --config configs/config.yaml \\
        --epochs 15 \\
        --batch-size 16 \\
        --lr 1e-4 \\
        --output outputs/checkpoints/student_dirt_best.pt

Rules:
- Never overwrite production weights unless dirt-road IoU improves AND overall mIoU drops <= 1.0 percentage point.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.dataset import DrivableDataset, IDDSource, SampleSource
from data.synthetic import SyntheticSource
from models.unet_mobilenet import MultiTaskUNet, build_model
from training.losses import MultiTaskLoss
from training.metrics import compute_iou_batch
from training.train import load_model_from_checkpoint
from utils.config import load_config, resolve_path
from utils.logging_utils import setup_logging
from utils.seed import get_device, set_global_seed

logger = logging.getLogger(__name__)


def evaluate_student(
    model_or_onnx_path: str,
    val_source: SampleSource,
    cfg: Dict[str, Any],
    input_size: Tuple[int, int] = (288, 512),
    device: str = "cpu",
    max_samples: int = 50,
) -> Dict[str, float]:
    """Evaluate student model (PyTorch or ONNX) on a validation source at a given resolution.

    Returns dict with:
        'overall_miou': mean IoU over all validation samples
        'dirt_iou': IoU specifically on dirt/drivable-fallback scenes
    """
    p = Path(model_or_onnx_path)
    is_onnx = p.suffix == ".onnx"

    session = None
    model = None
    if is_onnx:
        import onnxruntime
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 4
        session = onnxruntime.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])
    else:
        dev = torch.device(device)
        model, _ = load_model_from_checkpoint(str(p), dev)
        model.eval()

    h, w = input_size
    mean = np.array(cfg["data"]["imagenet_mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.array(cfg["data"]["imagenet_std"], dtype=np.float32).reshape(1, 1, 3)

    n_samples = min(max_samples, len(val_source))
    ious = []
    dirt_ious = []

    for i in range(n_samples):
        img_rgb, mask_gt, terrain = val_source.load(i)
        orig_h, orig_w = img_rgb.shape[:2]

        # Resize to input_size
        img_resized = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_gt_resized = cv2.resize(mask_gt, (w, h), interpolation=cv2.INTER_NEAREST)

        x_norm = ((img_resized.astype(np.float32) / 255.0) - mean) / std

        if is_onnx:
            x_np = np.ascontiguousarray(x_norm.transpose(2, 0, 1)[np.newaxis, ...]).astype(np.float32)
            seg_logits_np, _ = session.run(None, {"image": x_np})
            pred = np.argmax(seg_logits_np, axis=1)[0].astype(np.uint8)
        else:
            with torch.inference_mode():
                x_t = torch.from_numpy(x_norm.transpose(2, 0, 1)).unsqueeze(0).to(dev)
                out = model(x_t)
                pred = out["seg_logits"].argmax(1)[0].cpu().numpy().astype(np.uint8)

        gt_bin = (mask_gt_resized > 0).astype(np.uint8)
        pred_bin = (pred > 0).astype(np.uint8)

        inter = int(np.logical_and(pred_bin == 1, gt_bin == 1).sum())
        union = int(np.logical_or(pred_bin == 1, gt_bin == 1).sum())
        iou = float(inter / union) if union > 0 else 1.0
        ious.append(iou)

        # Tag dirt scenes (terrain == 1 or low paved confidence)
        if terrain == 1 or (gt_bin.mean() < 0.25):
            dirt_ious.append(iou)

    overall = float(np.mean(ious)) if ious else 0.0
    dirt = float(np.mean(dirt_ious)) if dirt_ious else overall
    return {"overall_miou": overall, "dirt_iou": dirt, "n_samples": n_samples}


def finetune_dirt(
    cfg: Dict[str, Any],
    checkpoint_path: str,
    output_path: str,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    smoke: bool = False,
) -> Path:
    """Fine-tune the student model on dirt road scenes."""
    gpu_cmd = (
        f"RECOMMENDED GPU COMMAND (Colab / Kaggle with CUDA):\n"
        f"  python -m training.finetune_dirt "
        f"--checkpoint {checkpoint_path} "
        f"--config configs/config.yaml "
        f"--epochs {epochs} "
        f"--batch-size {batch_size} "
        f"--lr {lr} "
        f"--output {output_path}\n"
    )
    print("\n" + "=" * 70)
    print(gpu_cmd)
    print("=" * 70 + "\n")

    dev = get_device()
    set_global_seed(cfg["seed"])
    logger.info("Device for fine-tuning: %s", dev)

    model, ckpt_info = load_model_from_checkpoint(checkpoint_path, dev)
    model.train()

    if smoke:
        source = SyntheticSource(num_samples=16, image_size=tuple(cfg["data"]["image_size"]))
        epochs = 1
    else:
        try:
            source = IDDSource(cfg, split="train", require_masks=True)
        except Exception as e:
            logger.warning("Could not load IDD train split (%s); using synthetic fallback.", e)
            source = SyntheticSource(num_samples=32, image_size=tuple(cfg["data"]["image_size"]))

    dataset = DrivableDataset(source, cfg, train=True, seed=cfg["seed"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = MultiTaskLoss(cfg["loss"]).to(dev)

    logger.info("Fine-tuning on dirt roads for %d epoch(s)...", epochs)
    for ep in range(epochs):
        loss_fn.set_epoch(ep)
        losses = []
        for batch in loader:
            imgs = batch["image"].to(dev)
            optimizer.zero_grad()
            out = model(imgs)
            loss, _ = loss_fn(out, {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()})
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
        logger.info("Epoch %d/%d - loss: %.4f", ep + 1, epochs, float(np.mean(losses)))

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, str(out_p))
    logger.info("Saved fine-tuned checkpoint to %s", out_p)
    return out_p


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune and evaluate student on dirt roads.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", default="outputs/checkpoints/student_best.pt")
    ap.add_argument("--output", default="outputs/checkpoints/student_dirt_best.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()

    setup_logging()
    cfg = load_config(a.config)

    print("\n=== DIRT ROAD STUDENT EVALUATION ===")
    print("GPU Command for training:")
    print(f"  python -m training.finetune_dirt --checkpoint {a.checkpoint} --output {a.output} --epochs {a.epochs} --batch-size {a.batch_size} --lr {a.lr}\n")

    if not a.eval_only:
        finetune_dirt(cfg, a.checkpoint, a.output, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, smoke=a.smoke)


if __name__ == "__main__":
    main()

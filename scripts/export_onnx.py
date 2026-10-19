"""Export MultiTaskUNet to ONNX for accelerated CPU/GPU inference.

Exports segmentation logits and evidential Dirichlet alpha parameters.
Usage:
    python scripts/export_onnx.py --checkpoint outputs/checkpoints/student_best.pt --output outputs/checkpoints/student_best.onnx
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train import load_model_from_checkpoint
from utils.config import load_config, resolve_path
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


class ONNXWrapper(torch.nn.Module):
    """Wrap MultiTaskUNet to output a clean tuple of (seg_logits, alpha)."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)
        return out["seg_logits"], out["alpha"]


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    input_size: Tuple[int, int] = (288, 512),
    opset: int = 14,
    int8: bool = False,
    data_src: Optional[str] = None,
) -> Path:
    device = torch.device("cpu")
    model, _ = load_model_from_checkpoint(checkpoint_path, device)
    model.eval()

    h, w = input_size
    dummy_input = torch.randn(1, 3, h, w, dtype=torch.float32)

    wrapper = ONNXWrapper(model)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting model to ONNX at %s (input size: %dx%d)...", out_file, h, w)
    try:
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(out_file),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["seg_logits", "alpha"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "seg_logits": {0: "batch_size"},
                "alpha": {0: "batch_size"},
            },
        )
        logger.info("Successfully exported ONNX model (%d bytes)", out_file.stat().st_size)

        if int8:
            try:
                import cv2
                import numpy as np
                from onnxruntime.quantization import (
                    CalibrationDataReader,
                    QuantFormat,
                    QuantType,
                    quantize_static,
                )

                calib_paths = []
                if data_src:
                    calib_paths = sorted(list((Path(data_src) / "image_archive").glob("*.png")))[:200]
                if not calib_paths:
                    calib_paths = sorted(list(Path("data/idd_converted/leftImg8bit/train").glob("*/*_leftImg8bit.png")))[:200]

                if calib_paths:
                    class IDDCalibReader(CalibrationDataReader):
                        def __init__(self, paths, insize=(288, 512)):
                            self.paths = paths
                            self.insize = insize
                            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                            self.idx = 0

                        def get_next(self):
                            if self.idx >= len(self.paths):
                                return None
                            p = self.paths[self.idx]
                            self.idx += 1
                            im = cv2.imread(str(p))
                            if im is None:
                                return self.get_next()
                            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                            im = cv2.resize(im, (self.insize[1], self.insize[0]))
                            norm = (im.astype(np.float32) / 255.0 - self.mean) / self.std
                            return {"image": norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)}

                        def rewind(self):
                            self.idx = 0

                    int8_path = out_file.with_stem(out_file.stem + "_int8")
                    logger.info("Applying static INT8 calibration on %d IDD images...", len(calib_paths))
                    reader = IDDCalibReader(calib_paths, input_size)
                    quantize_static(
                        model_input=str(out_file),
                        model_output=str(int8_path),
                        calibration_data_reader=reader,
                        quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8,
                    )
                    logger.info("Successfully exported static INT8 ONNX model (%d bytes)", int8_path.stat().st_size)
                    return int8_path
                else:
                    from onnxruntime.quantization import quantize_dynamic, QuantType

                    int8_path = out_file.with_stem(out_file.stem + "_int8")
                    logger.info("No calibration images found; applying dynamic INT8 quantization at %s...", int8_path)
                    quantize_dynamic(str(out_file), str(int8_path), weight_type=QuantType.QUInt8)
                    return int8_path
            except Exception as q_exc:
                logger.warning("INT8 quantization failed (%s); returning fp32 ONNX model.", q_exc)

        # OpenVINO export check
        export_openvino(str(out_file), str(out_file.parent))

        return out_file
    except Exception as exc:
        print(f"NEEDS USER DATA: pip install onnx onnxruntime (export failed: {exc})")
        logger.warning("ONNX export unavailable (%s); MultiTaskUNet remains available in PyTorch.", exc)
        return out_file


def export_openvino(onnx_model_path: str, output_dir: str) -> Optional[Path]:
    """Attempt export to OpenVINO IR format; skips with clear message if openvino is missing."""
    try:
        import openvino as ov

        logger.info("Converting ONNX to OpenVINO IR...")
        core = ov.Core()
        ov_model = core.read_model(onnx_model_path)
        ir_path = Path(output_dir) / f"{Path(onnx_model_path).stem}.xml"
        ov.save_model(ov_model, str(ir_path))
        logger.info("Successfully exported OpenVINO IR model at %s", ir_path)
        return ir_path
    except ImportError:
        msg = "OpenVINO not installed; skipping OpenVINO IR export and evaluation."
        print(f"[{msg}]")
        logger.info(msg)
        return None
    except Exception as exc:
        logger.warning("OpenVINO export failed (%s); continuing without OpenVINO.", exc)
        return None


def verify_onnx(
    checkpoint: str,
    onnx_path: str,
    data_src: Optional[str] = None,
    n_samples: int = 50,
    input_size: Tuple[int, int] = (288, 512),
) -> None:
    """Verify numerical equivalence between PyTorch and exported ONNX model."""
    import cv2
    import numpy as np
    import onnxruntime as ort

    device = torch.device("cpu")
    model, _ = load_model_from_checkpoint(checkpoint, device)
    model.eval()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    h_in, w_in = input_size
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    imgs = []
    if data_src and (Path(data_src) / "image_archive").is_dir():
        imgs = sorted(list((Path(data_src) / "image_archive").glob("*.png")))[:n_samples]
    if not imgs:
        imgs = sorted(list(Path("data/idd_converted/leftImg8bit/test").glob("*/*.png")))[:n_samples]
    if not imgs:
        imgs = sorted(list(Path("data/idd_converted/leftImg8bit/val").glob("*/*.png")))[:n_samples]

    max_diff_l = 0.0
    max_diff_a = 0.0
    mask_ious = []

    logger.info("Verifying PyTorch vs ONNX on %d images at %dx%d...", max(len(imgs), n_samples), h_in, w_in)
    with torch.no_grad():
        for i in range(max(len(imgs), n_samples)):
            if i < len(imgs):
                bgr = cv2.imread(str(imgs[i]))
            else:
                bgr = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            resized = cv2.resize(bgr, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            x = (rgb.astype(np.float32) / 255.0 - mean) / std
            chw = x.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

            py_out = model(torch.from_numpy(chw).to(device))
            py_l = py_out["seg_logits"].numpy()
            py_a = py_out["alpha"].numpy()
            py_mask = np.argmax(py_l, axis=1)[0]

            ox_l, ox_a = sess.run(None, {"image": chw})
            ox_mask = np.argmax(ox_l, axis=1)[0]

            max_diff_l = max(max_diff_l, float(np.max(np.abs(py_l - ox_l))))
            max_diff_a = max(max_diff_a, float(np.max(np.abs(py_a - ox_a))))

            inter = np.logical_and(py_mask == 1, ox_mask == 1).sum()
            union = np.logical_or(py_mask == 1, ox_mask == 1).sum()
            iou = 1.0 if union == 0 else float(inter / union)
            mask_ious.append(iou)

    mean_iou = float(np.mean(mask_ious))
    min_iou = float(np.min(mask_ious))
    logger.info(
        "Numerical Verification (%d images): max abs diff logits=%.4e, alpha=%.4e, mean Mask IoU=%.6f, min Mask IoU=%.6f",
        len(mask_ious), max_diff_l, max_diff_a, mean_iou, min_iou,
    )
    assert mean_iou >= 0.99, f"Mean Mask IoU {mean_iou:.6f} < 0.99"
    assert min_iou >= 0.95, f"Min Mask IoU {min_iou:.6f} < 0.95"
    logger.info("ONNX verification PASSED: mask IoU >= 0.99 threshold met.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export student model to ONNX and OpenVINO.")
    ap.add_argument("--checkpoint", default="outputs/checkpoints/student_best.pt")
    ap.add_argument("--output", default="outputs/checkpoints/student_best.onnx")
    ap.add_argument("--input-size", nargs=2, type=int, default=[288, 512], help="H W")
    ap.add_argument("--int8", action="store_true", help="Apply static INT8 calibration on 200 IDD train images")
    ap.add_argument("--data-src", default=None, help="Root of IDD_RESIZED for calibration/verification")
    ap.add_argument("--verify", action="store_true", help="Verify PyTorch vs ONNX on N images")
    ap.add_argument("--verify-n", type=int, default=50, help="Number of images for numerical verification")
    ap.add_argument("--openvino", action="store_true", help="Export OpenVINO IR if openvino installed")
    args = ap.parse_args()

    setup_logging()
    onnx_p = export_to_onnx(args.checkpoint, args.output, tuple(args.input_size), int8=args.int8, data_src=args.data_src)
    if args.verify:
        verify_onnx(args.checkpoint, str(onnx_p), data_src=args.data_src, n_samples=args.verify_n, input_size=tuple(args.input_size))
    if args.openvino:
        export_openvino(str(onnx_p), str(Path(args.output).parent))


if __name__ == "__main__":
    main()



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

                calib_paths = sorted(list(Path("c:/Users/K/Downloads/archive (2)/IDD_RESIZED/image_archive").glob("*.png")))[:200]
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Export student model to ONNX and OpenVINO.")
    ap.add_argument("--checkpoint", default="outputs/checkpoints/student_best.pt")
    ap.add_argument("--output", default="outputs/checkpoints/student_best.onnx")
    ap.add_argument("--input-size", nargs=2, type=int, default=[288, 512], help="H W")
    ap.add_argument("--int8", action="store_true", help="Apply static INT8 calibration on 200 IDD train images")
    ap.add_argument("--openvino", action="store_true", help="Export OpenVINO IR if openvino installed")
    args = ap.parse_args()

    setup_logging()
    onnx_p = export_to_onnx(args.checkpoint, args.output, tuple(args.input_size), int8=args.int8)
    if args.openvino:
        export_openvino(str(onnx_p), str(Path(args.output).parent))


if __name__ == "__main__":
    main()


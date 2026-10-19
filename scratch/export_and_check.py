import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import onnxruntime as ort

REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from utils.config import load_config
from training.train import load_model_from_checkpoint
from scripts.export_onnx import export_to_onnx

def letterbox_image(bgr: np.ndarray, target_size=(288, 512)):
    mh, mw = target_size
    fh, fw = bgr.shape[:2]
    scale = min(mw / fw, mh / fh)
    nw, nh = int(round(fw * scale)), int(round(fh * scale))
    small = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((mh, mw, 3), dtype=np.uint8)
    pad_top = (mh - nh) // 2
    pad_left = (mw - nw) // 2
    padded[pad_top:pad_top + nh, pad_left:pad_left + nw] = small
    return padded

def main():
    print("=== Step 5: EXPORT AND NUMERICAL CHECK ===")
    device = torch.device("cpu")
    student_ckpt = REPO_DIR / "outputs" / "checkpoints" / "student_best.pt"
    onnx_out = REPO_DIR / "outputs" / "checkpoints" / "student_v2.onnx"
    
    print(f"Exporting {student_ckpt} to {onnx_out} at 288x512...")
    export_to_onnx(str(student_ckpt), str(onnx_out), input_size=(288, 512), opset=14, int8=False)
    
    # Load PyTorch model
    py_model, _ = load_model_from_checkpoint(str(student_ckpt), device)
    py_model.eval()
    
    # Load ONNX session
    sess = ort.InferenceSession(str(onnx_out), providers=["CPUExecutionProvider"])
    
    # Load 50 images from test split
    test_split_file = REPO_DIR / "data" / "splits" / "test.txt"
    test_lines = [line.strip() for line in test_split_file.read_text().splitlines() if line.strip()][:50]
    img_dir = Path(r"C:\Users\K\Downloads\archive (2)\IDD_RESIZED\image_archive")
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    
    max_abs_diff_logits = 0.0
    max_abs_diff_alpha = 0.0
    mask_ious = []
    
    print(f"Evaluating {len(test_lines)} test images on PyTorch vs ONNX...")
    with torch.no_grad():
        for fname in test_lines:
            im_path = img_dir / fname
            bgr = cv2.imread(str(im_path))
            if bgr is None:
                continue
            padded = letterbox_image(bgr, (288, 512))
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            norm = (rgb.astype(np.float32) / 255.0 - mean) / std
            chw = norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
            
            # PyTorch forward
            t_in = torch.from_numpy(chw).to(device)
            py_out = py_model(t_in)
            py_logits = py_out["seg_logits"].numpy()
            py_alpha = py_out["alpha"].numpy()
            py_mask = np.argmax(py_logits, axis=1)[0]
            
            # ONNX forward
            onnx_logits, onnx_alpha = sess.run(None, {"image": chw})
            onnx_mask = np.argmax(onnx_logits, axis=1)[0]
            
            # Differences
            diff_l = float(np.max(np.abs(py_logits - onnx_logits)))
            diff_a = float(np.max(np.abs(py_alpha - onnx_alpha)))
            max_abs_diff_logits = max(max_abs_diff_logits, diff_l)
            max_abs_diff_alpha = max(max_abs_diff_alpha, diff_a)
            
            # Mask IoU
            inter = np.logical_and(py_mask == 1, onnx_mask == 1).sum()
            union = np.logical_or(py_mask == 1, onnx_mask == 1).sum()
            if union == 0:
                iou = 1.0 if np.array_equal(py_mask, onnx_mask) else 0.0
            else:
                iou = float(inter / union)
            mask_ious.append(iou)
            
    mean_mask_iou = float(np.mean(mask_ious))
    min_mask_iou = float(np.min(mask_ious))
    
    print("\n--- Numerical Verification Results (50 images) ---")
    print(f"Max abs diff (seg_logits): {max_abs_diff_logits:.6e}")
    print(f"Max abs diff (alpha):      {max_abs_diff_alpha:.6e}")
    print(f"Mean Mask IoU:             {mean_mask_iou:.6f}")
    print(f"Min Mask IoU:              {min_mask_iou:.6f}")
    print(f"Pass criteria (>= 0.99):   {mean_mask_iou >= 0.99 and min_mask_iou >= 0.99}")
    
    return {
        "max_abs_diff_logits": max_abs_diff_logits,
        "max_abs_diff_alpha": max_abs_diff_alpha,
        "mean_mask_iou": mean_mask_iou,
        "min_mask_iou": min_mask_iou,
        "passed": bool(mean_mask_iou >= 0.99 and min_mask_iou >= 0.99)
    }

if __name__ == "__main__":
    main()

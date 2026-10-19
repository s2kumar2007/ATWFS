import sys
from pathlib import Path
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import onnxruntime as ort

REPO_DIR = Path(r"c:\Users\K\OneDrive\Documents\idd_drivable\idd_drivable")
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from utils.config import load_config
from training.train import load_model_from_checkpoint
from models.unet_mobilenet import evidential_probs, evidential_uncertainty
from training.metrics import ECEAccumulator
from evaluation.evaluate import official_confusion, official_ious

def letterbox_pair(img_rgb: np.ndarray, mask: np.ndarray, target_size=(288, 512), ignore_label=255):
    mh, mw = target_size
    fh, fw = img_rgb.shape[:2]
    scale = min(mw / fw, mh / fh)
    nw, nh = int(round(fw * scale)), int(round(fh * scale))
    small_img = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    small_mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    if (nw, nh) == (mw, mh):
        return small_img, small_mask
    padded_img = np.zeros((mh, mw, 3), dtype=np.uint8)
    padded_mask = np.full((mh, mw), ignore_label, dtype=np.uint8)
    pad_top = (mh - nh) // 2
    pad_left = (mw - nw) // 2
    padded_img[pad_top:pad_top + nh, pad_left:pad_left + nw] = small_img
    padded_mask[pad_top:pad_top + nh, pad_left:pad_left + nw] = small_mask
    return padded_img, padded_mask

class TestSplitDataset(Dataset):
    def __init__(self, file_list, img_dir, mask_dir, input_size=(288, 512)):
        self.files = file_list
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.input_size = input_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        ip = self.img_dir / fname
        mp = self.mask_dir / fname.replace("Image_", "Mask_")
        
        bgr = cv2.imread(str(ip))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raw_mask = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if raw_mask.ndim == 3:
            raw_mask = raw_mask[:, :, 0]
            
        pad_rgb, pad_mask = letterbox_pair(rgb, raw_mask, self.input_size, ignore_label=255)
        chw = (pad_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0 - self.mean) / self.std
        
        return {
            "image": torch.from_numpy(chw).float(),
            "mask": torch.from_numpy(pad_mask).long(),
            "fname": fname
        }

def evaluate_pytorch(model, dataloader, ece_bins=15):
    device = torch.device("cpu")
    model.eval()
    mat = np.zeros((3, 3), np.int64)
    dirt_mat = np.zeros((3, 3), np.int64)
    ece_acc = ECEAccumulator(ece_bins)
    uncertainties = []
    dirt_samples = 0
    total_samples = 0

    with torch.no_grad():
        for b in dataloader:
            imgs = b["image"].to(device)
            masks = b["mask"].numpy()
            out = model(imgs)
            
            preds = out["seg_logits"].argmax(1).cpu().numpy()
            alphas = out["alpha"].cpu().float()
            probs = evidential_probs(alphas)
            uncs = evidential_uncertainty(alphas).mean(dim=(1, 2)).numpy()
            uncertainties.extend(uncs.tolist())

            for i in range(len(imgs)):
                pad_mask = masks[i]
                pred = preds[i]
                prob = probs[i:i+1]
                
                valid = pad_mask != 255
                gt_target = np.where(valid, pad_mask, 255)
                mat = official_confusion(gt_target, pred, 2, mat)
                
                gt_t = torch.from_numpy(np.where(valid, pad_mask, 0)).long().unsqueeze(0)
                ece_acc.update(prob, gt_t)
                
                drivable_frac = float((pad_mask[valid] == 1).mean()) if valid.any() else 0.0
                if 0.05 < drivable_frac < 0.35:
                    dirt_mat = official_confusion(gt_target, pred, 2, dirt_mat)
                    dirt_samples += 1
                total_samples += 1

    ious = official_ious(mat)
    dirt_ious = official_ious(dirt_mat)
    return {
        "mIoU": float(np.nanmean(ious)),
        "iou_nondrivable": float(ious[0]),
        "iou_drivable": float(ious[1]),
        "dirt_drivable_iou": float(dirt_ious[1]),
        "ECE": float(ece_acc.compute()),
        "mean_uncertainty": float(np.mean(uncertainties)),
        "dirt_samples": dirt_samples,
        "total_samples": total_samples
    }

def evaluate_onnx(sess, dataloader, ece_bins=15):
    mat = np.zeros((3, 3), np.int64)
    dirt_mat = np.zeros((3, 3), np.int64)
    ece_acc = ECEAccumulator(ece_bins)
    uncertainties = []
    dirt_samples = 0
    total_samples = 0

    for b in dataloader:
        imgs = b["image"].numpy()
        masks = b["mask"].numpy()
        
        seg_logits_np, alpha_np = sess.run(None, {"image": imgs})
        preds = np.argmax(seg_logits_np, axis=1)
        alphas = torch.from_numpy(alpha_np).float()
        probs = evidential_probs(alphas)
        uncs = evidential_uncertainty(alphas).mean(dim=(1, 2)).numpy()
        uncertainties.extend(uncs.tolist())

        for i in range(len(imgs)):
            pad_mask = masks[i]
            pred = preds[i]
            prob = probs[i:i+1]
            
            valid = pad_mask != 255
            gt_target = np.where(valid, pad_mask, 255)
            mat = official_confusion(gt_target, pred, 2, mat)
            
            gt_t = torch.from_numpy(np.where(valid, pad_mask, 0)).long().unsqueeze(0)
            ece_acc.update(prob, gt_t)
            
            drivable_frac = float((pad_mask[valid] == 1).mean()) if valid.any() else 0.0
            if 0.05 < drivable_frac < 0.35:
                dirt_mat = official_confusion(gt_target, pred, 2, dirt_mat)
                dirt_samples += 1
            total_samples += 1

    ious = official_ious(mat)
    dirt_ious = official_ious(dirt_mat)
    return {
        "mIoU": float(np.nanmean(ious)),
        "iou_nondrivable": float(ious[0]),
        "iou_drivable": float(ious[1]),
        "dirt_drivable_iou": float(dirt_ious[1]),
        "ECE": float(ece_acc.compute()),
        "mean_uncertainty": float(np.mean(uncertainties)),
        "dirt_samples": dirt_samples,
        "total_samples": total_samples
    }

def main():
    print("=== Step 6: EVALUATION ON HELD-OUT TEST SPLIT (288x512) ===")
    test_split_path = REPO_DIR / "data" / "splits" / "test.txt"
    test_files = [line.strip() for line in test_split_path.read_text().splitlines() if line.strip()]
    print(f"Total test split images: {len(test_files)}")
    
    img_dir = Path(r"C:\Users\K\Downloads\archive (2)\IDD_RESIZED\image_archive")
    mask_dir = Path(r"C:\Users\K\Downloads\archive (2)\IDD_RESIZED\mask_archive")
    
    # We evaluate a representative 250-sample subset of test split for speed on CPU (or all)
    eval_files = test_files[:250]
    print(f"Evaluating {len(eval_files)} images across test scenes at 288x512...")
    
    dataset = TestSplitDataset(eval_files, img_dir, mask_dir, input_size=(288, 512))
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # 1. Teacher
    teacher_ckpt = REPO_DIR / "outputs" / "checkpoints" / "teacher_best.pt"
    print(f"\n[1/3] Evaluating Teacher: {teacher_ckpt.name}...")
    teacher_model, _ = load_model_from_checkpoint(str(teacher_ckpt), torch.device("cpu"))
    teacher_metrics = evaluate_pytorch(teacher_model, loader)
    print("Teacher:", teacher_metrics)
    
    # 2. Student (PyTorch)
    student_ckpt = REPO_DIR / "outputs" / "checkpoints" / "student_best.pt"
    print(f"\n[2/3] Evaluating Student (PyTorch): {student_ckpt.name}...")
    student_model, _ = load_model_from_checkpoint(str(student_ckpt), torch.device("cpu"))
    student_metrics = evaluate_pytorch(student_model, loader)
    print("Student (PyTorch):", student_metrics)
    
    # 3. ONNX Student
    onnx_path = REPO_DIR / "outputs" / "checkpoints" / "student_v2.onnx"
    print(f"\n[3/3] Evaluating ONNX Student: {onnx_path.name}...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    onnx_sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    onnx_metrics = evaluate_onnx(onnx_sess, loader)
    print("Student (ONNX):", onnx_metrics)
    
    final_report = {
        "split": "held-out-test",
        "resolution": "288x512",
        "evaluated_samples": len(eval_files),
        "total_test_samples": len(test_files),
        "teacher": teacher_metrics,
        "student": student_metrics,
        "student_onnx": onnx_metrics
    }
    
    out_dir = REPO_DIR / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "final_eval.json"
    with open(report_file, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nSuccessfully saved final evaluation to {report_file}")

if __name__ == "__main__":
    main()

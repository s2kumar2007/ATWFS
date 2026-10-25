import argparse
import numpy as np
import torch
import cv2

from data.dataset import build_dataloader, build_dataset
from models.unet_mobilenet import build_model, evidential_uncertainty
from training.train import load_model_from_checkpoint
from utils.config import load_config, resolve_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--video", required=True)
    a = ap.parse_args()
    
    cfg = load_config(a.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model_from_checkpoint(resolve_path(cfg["inference"]["checkpoint"]), device)
    model.eval()
    
    ds = build_dataset(cfg, "val")
    dl = build_dataloader(ds, cfg, False, batch_size=4)
    
    print("Evaluating IDD val...")
    idd_unc = []
    with torch.no_grad(), torch.inference_mode():
        for b in dl:
            alpha = model(b["image"].to(device))["alpha"]
            unc = evidential_uncertainty(alpha).mean().item()
            idd_unc.append(unc)
            if len(idd_unc) > 20: break
    
    print(f"IDD Val Mean Uncertainty: {np.mean(idd_unc):.4f}")
    
    print("Evaluating non-IDD video...")
    cap = cv2.VideoCapture(a.video)
    vid_unc = []
    mh, mw = cfg["inference"]["input_size"]
    mean = np.asarray(cfg["data"]["imagenet_mean"], np.float32)
    std = np.asarray(cfg["data"]["imagenet_std"], np.float32)
    
    with torch.no_grad(), torch.inference_mode():
        for i in range(100):
            ret, frame = cap.read()
            if not ret: break
            if i % 10 != 0: continue
            
            # Letterbox
            fh, fw = frame.shape[:2]
            scale = min(mw / fw, mh / fh)
            nw, nh = int(fw * scale), int(fh * scale)
            small = cv2.resize(frame, (nw, nh))
            padded = np.zeros((mh, mw, 3), dtype=np.uint8)
            padded[(mh - nh)//2:(mh - nh)//2 + nh, (mw - nw)//2:(mw - nw)//2 + nw] = small
            
            x = (cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - mean) / std
            x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
            alpha = model(x)["alpha"]
            unc = evidential_uncertainty(alpha).mean().item()
            vid_unc.append(unc)
            
    print(f"Video Mean Uncertainty: {np.mean(vid_unc):.4f}")

if __name__ == "__main__":
    main()

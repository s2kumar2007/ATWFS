import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
from fvcore.nn import FlopCountAnalysis

from inference.detector import build_detector
from inference.live import LivePipeline
from models.unet_mobilenet import build_model
from training.train import load_model_from_checkpoint
from utils.config import load_config, resolve_path


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["seg_logits"], out["alpha"], out["terrain_logits"]


class MemoryMonitor:
    def __init__(self):
        self.keep_measuring = True
        self.peak_rss = 0
        self.thread = threading.Thread(target=self.measure_memory)

    def measure_memory(self):
        process = psutil.Process(os.getpid())
        while self.keep_measuring:
            try:
                rss = process.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except psutil.NoSuchProcess:
                pass
            time.sleep(0.01)

    def start(self):
        self.keep_measuring = True
        self.peak_rss = psutil.Process(os.getpid()).memory_info().rss
        self.thread.start()

    def stop(self):
        self.keep_measuring = False
        self.thread.join()
        return self.peak_rss / (1024 * 1024)


def get_hardware_info():
    import platform
    cpu_name = platform.processor()
    try:
        # For Windows
        out = subprocess.check_output(["wmic", "cpu", "get", "name"]).decode().strip().split('\n')
        if len(out) > 1:
            cpu_name = out[1].strip()
    except Exception:
        pass
    
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    torch_threads = torch.get_num_threads()
    
    return {
        "cpu_name": cpu_name,
        "cores": f"{physical_cores} physical / {logical_cores} logical",
        "ram": f"{ram_gb:.1f} GB",
        "torch_version": torch.__version__,
        "torch_threads": torch_threads,
        "device": "cpu"
    }


def worker_main(args):
    # Set to CPU
    device = torch.device("cpu")
    torch.set_num_threads(torch.get_num_threads()) # ensure it is using max threads for cpu

    cfg = load_config()
    input_size = cfg.get("inference", {}).get("input_size", cfg.get("data", {}).get("image_size", [288, 512]))
    h, w = input_size
    
    metrics = {"input_size": f"{h}x{w}"}
    
    # 1. Checkpoint size
    ckpt_path = args.checkpoint
    if os.path.exists(ckpt_path):
        metrics["ckpt_size_MB"] = os.path.getsize(ckpt_path) / (1024 * 1024)
    else:
        metrics["ckpt_size_MB"] = "not measured"
        
    try:
        if os.path.exists(ckpt_path):
            model, _ = load_model_from_checkpoint(ckpt_path, device)
        else:
            model = build_model({**cfg["model"], "pretrained": False}).to(device)
        model.eval()
        
        # Params
        params_M = sum(p.numel() for p in model.parameters()) / 1e6
        metrics["params_M"] = params_M
        
        # FLOPs
        dummy_input = torch.randn(1, 3, h, w).to(device)
        wrapper = ModelWrapper(model)
        flops = FlopCountAnalysis(wrapper, dummy_input)
        gmacs = flops.total() / 1e9
        metrics["GMACs"] = gmacs
        metrics["GFLOPs"] = gmacs * 2
        
        # Inference time
        monitor = MemoryMonitor()
        monitor.start()
        
        with torch.inference_mode():
            for _ in range(10):
                model(dummy_input)
                
            times = []
            for _ in range(100):
                t0 = time.time()
                model(dummy_input)
                times.append((time.time() - t0) * 1000)
                
        metrics["peak_mem_model_MB"] = monitor.stop()
        metrics["inference_ms_mean"] = np.mean(times)
        metrics["inference_ms_median"] = np.median(times)
        metrics["inference_ms_p95"] = np.percentile(times, 95)
        metrics["fps"] = 1000.0 / metrics["inference_ms_mean"]

    except Exception as e:
        print(f"Error in model metrics: {e}")
        metrics["params_M"] = "not measured"
        metrics["GMACs"] = "not measured"
        metrics["GFLOPs"] = "not measured"
        metrics["peak_mem_model_MB"] = "not measured"
        metrics["inference_ms_mean"] = "not measured"
        metrics["inference_ms_median"] = "not measured"
        metrics["inference_ms_p95"] = "not measured"
        metrics["fps"] = "not measured"

    # End-to-end
    try:
        monitor = MemoryMonitor()
        
        cap = cv2.VideoCapture(str(resolve_path(args.video)))
        frames = []
        for _ in range(55):
            ret, frame = cap.read()
            if not ret: break
            if frame.shape[1] > 1280:
                scale = 1280 / frame.shape[1]
                frame = cv2.resize(frame, (1280, int(frame.shape[0] * scale)))
            frames.append(frame)
        cap.release()
        
        if len(frames) > 0:
            detector = build_detector(cfg["inference"]["detector"], cfg["paths"]["output_dir"])
            pipeline = LivePipeline(cfg, model, device, detector)
            
            monitor.start()
            
            # Warm up
            for f in frames[:5]:
                pipeline.process_frame(f)
                
            e2e_times = []
            for f in frames[5:55]:
                t0 = time.time()
                pipeline.process_frame(f)
                e2e_times.append((time.time() - t0) * 1000)
                
            metrics["peak_mem_e2e_MB"] = monitor.stop()
            metrics["e2e_ms_mean"] = np.mean(e2e_times)
            metrics["e2e_ms_median"] = np.median(e2e_times)
            metrics["e2e_ms_p95"] = np.percentile(e2e_times, 95)
            metrics["e2e_fps"] = 1000.0 / metrics["e2e_ms_mean"]
            
        else:
            raise ValueError("No frames read from video")
            
    except Exception as e:
        print(f"Error in e2e metrics: {e}")
        metrics["peak_mem_e2e_MB"] = "not measured"
        metrics["e2e_ms_mean"] = "not measured"
        metrics["e2e_ms_median"] = "not measured"
        metrics["e2e_ms_p95"] = "not measured"
        metrics["e2e_fps"] = "not measured"
        
    with open(args.output_json, "w") as f:
        json.dump(metrics, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="whatsapp_result.mp4")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    
    if args.worker:
        worker_main(args)
        return
        
    hardware = get_hardware_info()
    
    results = {}
    for name, ckpt in [("Teacher", "outputs/checkpoints/teacher_best.pt"), ("Student", "outputs/checkpoints/student_best.pt")]:
        tmp_json = f"outputs/reports/tmp_{name}.json"
        os.makedirs("outputs/reports", exist_ok=True)
        print(f"Measuring {name}...")
        
        cmd = [sys.executable, "-m", "evaluation.slide_metrics", "--worker", "--video", args.video, "--checkpoint", ckpt, "--output-json", tmp_json]
        subprocess.run(cmd, check=True)
        
        with open(tmp_json, "r") as f:
            res = json.load(f)
        results[name] = res
        os.remove(tmp_json)

    # Compile CSV
    csv_path = "outputs/reports/slide_metrics.csv"
    json_path = "outputs/reports/slide_metrics.json"
    
    keys = ["input_size", "params_M", "GMACs", "GFLOPs", "inference_ms_mean", "inference_ms_p95", "fps", "e2e_ms_mean", "e2e_ms_p95", "e2e_fps", "peak_mem_model_MB", "peak_mem_e2e_MB", "ckpt_size_MB"]
    
    with open(csv_path, "w") as f:
        f.write("Model," + ",".join(keys) + "\n")
        for name, res in results.items():
            f.write(name + "," + ",".join(str(res.get(k, "not measured")) for k in keys) + "\n")
            
    with open(json_path, "w") as f:
        json.dump({"hardware": hardware, "results": results}, f, indent=4)
        
    # Print Table
    print("\n" + "="*80)
    print(f"Input size: {results['Teacher']['input_size']}")
    print(f"Hardware: {hardware['cpu_name']}, {hardware['cores']} cores, {hardware['ram']} RAM, Torch {hardware['torch_version']} ({hardware['torch_threads']} threads), CPU")
    print("="*80)
    
    headers = ["Metric", "Teacher", "Student"]
    rows = [
        ["Inference time (ms/frame)", 
         f"{results['Teacher']['inference_ms_mean']:.1f} (p95: {results['Teacher']['inference_ms_p95']:.1f})" if isinstance(results['Teacher']['inference_ms_mean'], float) else "not measured",
         f"{results['Student']['inference_ms_mean']:.1f} (p95: {results['Student']['inference_ms_p95']:.1f})" if isinstance(results['Student']['inference_ms_mean'], float) else "not measured"],
        ["FLOPs (GFLOPs)", 
         f"{results['Teacher'].get('GFLOPs', 'not measured'):.2f}" if isinstance(results['Teacher'].get('GFLOPs'), float) else "not measured",
         f"{results['Student'].get('GFLOPs', 'not measured'):.2f}" if isinstance(results['Student'].get('GFLOPs'), float) else "not measured"],
        ["Parameters (M)", 
         f"{results['Teacher'].get('params_M', 'not measured'):.2f}" if isinstance(results['Teacher'].get('params_M'), float) else "not measured",
         f"{results['Student'].get('params_M', 'not measured'):.2f}" if isinstance(results['Student'].get('params_M'), float) else "not measured"],
        ["Throughput (FPS)", 
         f"{results['Teacher']['fps']:.1f}" if isinstance(results['Teacher']['fps'], float) else "not measured",
         f"{results['Student']['fps']:.1f}" if isinstance(results['Student']['fps'], float) else "not measured"],
        ["End-to-end latency (ms)", 
         f"{results['Teacher']['e2e_ms_mean']:.1f} (p95: {results['Teacher']['e2e_ms_p95']:.1f})" if isinstance(results['Teacher']['e2e_ms_mean'], float) else "not measured",
         f"{results['Student']['e2e_ms_mean']:.1f} (p95: {results['Student']['e2e_ms_p95']:.1f})" if isinstance(results['Student']['e2e_ms_mean'], float) else "not measured"],
        ["Peak memory (MB)", 
         f"{results['Teacher']['peak_mem_model_MB']:.1f} (model) / {results['Teacher']['peak_mem_e2e_MB']:.1f} (e2e)" if isinstance(results['Teacher']['peak_mem_model_MB'], float) else "not measured",
         f"{results['Student']['peak_mem_model_MB']:.1f} (model) / {results['Student']['peak_mem_e2e_MB']:.1f} (e2e)" if isinstance(results['Student']['peak_mem_model_MB'], float) else "not measured"],
    ]
    
    # Print formatted table
    col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *rows)]
    fmt = " | ".join("{{:<{}}}".format(w) for w in col_widths)
    print(fmt.format(*headers))
    print("-" * (sum(col_widths) + 3 * len(col_widths) - 1))
    for row in rows:
        print(fmt.format(*row))
    print("="*80)
    

if __name__ == "__main__":
    main()

import sys, time, numpy as np, torch, cv2, psutil
from torch.utils.flop_counter import FlopCounterMode
from training.train import load_model_from_checkpoint

ckpt, S = sys.argv[1], 256
m, _ = load_model_from_checkpoint(ckpt, torch.device("cpu"))
m.eval()
x = torch.randn(1, 3, S, S)
mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

with torch.no_grad():
    with FlopCounterMode(display=False) as fc:
        m(x)
    gflops = fc.get_total_flops() / 1e9
    for _ in range(10):
        m(x)
    t = []
    for _ in range(100):
        a = time.perf_counter(); m(x); t.append((time.perf_counter() - a) * 1000)
    frame = (np.random.rand(1080, 1920, 3) * 255).astype(np.uint8)
    e = []
    for _ in range(50):
        a = time.perf_counter()
        r = (cv2.resize(frame, (S, S)).astype(np.float32) / 255.0 - mean) / std
        o = m(torch.from_numpy(r.transpose(2, 0, 1)).float()[None])["seg_logits"].argmax(1)[0].numpy().astype(np.uint8)
        cv2.resize(o, (1920, 1080), interpolation=cv2.INTER_NEAREST)
        e.append((time.perf_counter() - a) * 1000)

mi = psutil.Process().memory_info()
peak = getattr(mi, "peak_wset", mi.rss) / 1e6
print(f"GFLOPs {gflops:.2f} | model ms {np.median(t):.1f} | end-to-end ms {np.median(e):.1f} | peak mem MB {peak:.0f}")
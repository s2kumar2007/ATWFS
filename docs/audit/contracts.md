# Phase 2 contract review

Inference now follows data.letterbox for both frame geometries. Regression tests capture the actual ONNX input and compare it to DrivableDataset for 16:9, square and other aspect ratios, in both modes. Random model fallback removed; explicit corrupt/missing ONNX errors fail closed. CPU profile explicitly chooses CPU. Configuration rejects invalid image sizes, std, fusion thresholds, queues and decision limits. Camera profiles and detector artifacts are rooted under the run directory. Negative/NaN sequence FPS uses 25 FPS, explicitly a fallback timebase rather than a measured source FPS.

NOT RUN: student_v2 ONNX vs student_v2_best.pt agreement: matching PyTorch weights missing. Existing student_best.pt is not relabelled as v2.

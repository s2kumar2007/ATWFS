# Phase 2 contract review

Inference now follows data.letterbox for both frame geometries. Regression tests capture the actual ONNX input and compare it to DrivableDataset for 16:9, square and other aspect ratios, in both modes. Random model fallback removed; explicit corrupt/missing ONNX errors fail closed. CPU profile explicitly chooses CPU. Configuration rejects invalid image sizes, std, fusion thresholds, queues and decision limits. Camera profiles and detector artifacts are rooted under the run directory. Negative/NaN sequence FPS uses 25 FPS, explicitly a fallback timebase rather than a measured source FPS.

NOT RUN: student_v2 ONNX vs student_v2_best.pt agreement: matching PyTorch weights missing. Existing student_best.pt is not relabelled as v2.

## Available pair and model I/O

The older available student_best.pt / student_best.onnx pair was compared on the first sorted held-out test images. This is not a substitute for the missing v2 PyTorch pair.

| Measurement | Value | Evidence |
|---|---:|---|
| Images compared | 50 | <!-- metric: outputs/reports/audit_available_pair_agreement.json#/images = 50 --> |
| Mean mask IoU | 1.000000 | <!-- metric: outputs/reports/audit_available_pair_agreement.json#/mean_mask_iou = 1.000000 --> |
| Minimum mask IoU | 1.000000 | <!-- metric: outputs/reports/audit_available_pair_agreement.json#/minimum_mask_iou = 1.000000 --> |

ONNX input is image, float32 NCHW, with fixed spatial input dimensions defined by the model. Outputs are seg_logits and alpha, float32. Symbolic output shape labels in the supplied export are not reliable fixed spatial dimensions; the runtime tensors were used for the checks. Full metadata and numerical differences are in outputs/audit/evaluation/model_io.json and available_pair_agreement.json.

NOT RUN: exhaustive dynamic configuration-key usage coverage across every optional training/export path. The added validator covers required inference boundary contracts and the parser test checks documented single-line module-command flags; it is not a full clean-clone execution of every training example.

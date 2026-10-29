# Held-out evaluation and integrity

The full converted held-out test split was evaluated with the existing student_v2.onnx, stretch preprocessing and no weight changes:

```powershell
python -m evaluation.test_split_eval --onnx outputs/checkpoints/student_v2.onnx --output outputs/reports/audit_full_test.json --batch-size 4 --num-workers 0
```

| Measurement | Value | Evidence |
|---|---:|---|
| Images | 1052 | <!-- metric: outputs/reports/audit_full_test.json#/total_test_images = 1052 --> |
| mIoU | 0.932687 | <!-- metric: outputs/reports/audit_full_test.json#/results/student_v2_onnx/mIoU = 0.932687 --> |
| Drivable IoU | 0.902134 | <!-- metric: outputs/reports/audit_full_test.json#/results/student_v2_onnx/iou_drivable = 0.902134 --> |
| Non-drivable IoU | 0.963239 | <!-- metric: outputs/reports/audit_full_test.json#/results/student_v2_onnx/iou_nondrivable = 0.963239 --> |
| ECE | 0.002104 | <!-- metric: outputs/reports/audit_full_test.json#/results/student_v2_onnx/ECE = 0.002104 --> |
| Mean uncertainty | 0.053549 | <!-- metric: outputs/reports/audit_full_test.json#/results/student_v2_onnx/mean_uncertainty = 0.053549 --> |

These are real model predictions against the locally available converted labels. The dataset conversion's provenance and IDD usage terms remain limitations; these figures do not establish an official challenge-server result. The evaluator's total_samples field counts pixels, not images; the image count is total_test_images. No metric formula or threshold was changed.

Split-disjointness tests check frame and scene manifests. An AST review of training/train.py and training/distill.py finds only literal train and val dataset calls. Optional fine-tuning routines have separate inputs; these were not run. File-based split disjointness alone does not prove byte-level deduplication or correct source annotations.

Current document metrics are checked using `python scripts/check_metric_claims.py`. Evidence comments map displayed numeric measurements to JSON pointers. The checker also rejects unannotated numerical performance table rows. It does not semantically classify every arbitrary number in prose or code examples; architecture sizes, configuration limits, identifiers and targets are specifications rather than measured results.

NOT RUN: v2 PyTorch/ONNX agreement because matching student_v2_best.pt is missing. Download helper returned needs_user_data because paths.weights_url is empty. An available older student_best pair is checked separately, without calling it v2.

Taxonomy, Grad-CAM, ECE hand-computation and synthetic augmentation resilience are exercised in the smoke suite. NOT RUN: BDD100K real zero-shot evaluation, new real compression benchmark and full robustness study on labelled corruptions. Historical quality_guard.json is a failed observation, not a new audit measurement; its threshold is retained and its CLI now returns nonzero on failure. The historical golden masks lack sufficient provenance to certify unchanged quality for the present video/model pair.

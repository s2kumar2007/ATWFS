# Platform audit

Commands were run on Windows with Python from the installed host environment. Input paths containing spaces were exercised by the real video runs. No personal input path is stored in source or documentation.

| Check | Status | Evidence / limitation |
|---|---|---|
| Windows paths and input with spaces | PASS | Task A/B commands in video_report.md, with ATWFS_TEST_VIDEO |
| PNG sequences and grayscale | PASS pending final CLI rerun | ImageSequenceCapture and test_grayscale_png_sequence_is_bgr avoid native video-codec instability |
| Non-widescreen / large frames | PASS | preprocessing parameterization and failure-injection harness |
| mp4v writer and H.264 re-encode | PASS | Task A/B files decoded by independent verifier; optional imageio-ffmpeg executable |
| avc1 codec availability / fallback | NOT RUN | No explicit avc1 deployment validation; _open_writer falls back to MJPG AVI |
| Windows DataLoader workers | PASS | config.yaml data.num_workers=0; evaluation uses --num-workers 0 |
| ONNX CPU provider | PASS | Real student_v2 ONNX inference and held-out test evaluation |
| ONNX CUDA to CPU fallback | NOT RUN | GPU/provider configuration unavailable on this host |
| Linux execution | NOT RUN | No Linux host attached |
| macOS execution | NOT RUN | No macOS host attached |
| Long Windows paths | NOT RUN | Inputs with spaces tested; extended path limits not comprehensively exercised |
| Shell script endings | Reviewed | .sh and .ps1 checked in phase 7; native scripts not executed if they train models |
| Headless display | NOT RUN | Display requires GUI OpenCV; requirements.txt uses headless OpenCV |

Fresh-environment compatibility is BLOCKED by shell package-index access. Installed host versions differ from requirements; see build_health.md. No cross-platform pass is inferred from Windows-only results.

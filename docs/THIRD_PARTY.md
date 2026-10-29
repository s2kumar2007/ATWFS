# Third-party components

Inventory for the audited source and locally installed packages; this is attribution and a review checklist, not a legal conclusion. Retain the exact upstream notices with distributed binaries. Transitive dependencies and actual wheel/build notices also need review before distribution.

| Component | Use | Licence / terms to review | Primary source |
|---|---|---|---|
| PyTorch | Model execution and training code | BSD-3-Clause; bundled libraries have additional notices | https://github.com/pytorch/pytorch/blob/main/LICENSE |
| torchvision / MobileNetV3 | Encoder and optional pretrained initialization | BSD code; pretrained weights and ImageNet data terms require separate review | https://github.com/pytorch/vision/blob/main/LICENSE |
| Ultralytics YOLOv8n / YOLOv8s, BYTETracker integration | Obstacle detection, reference detection, tracking | AGPL-3.0; upstream offers other licensing options; do not infer project-wide legal compliance from this inventory | https://www.ultralytics.com/license |
| COCO-pretrained YOLO weights | Downloaded object-detector parameters | Review upstream model licence and COCO annotation/image terms; image licences vary | https://cocodataset.org/#termsofuse |
| IDD / converted IDD archive | Segmentation data and split manifests | IDD dataset access/redistribution/usage terms MUST be checked with the provider; conversion does not remove those terms | https://insaan.iiit.ac.in/datasets/ |
| BDD100K | Optional external evaluation | Dataset terms must be checked before use or redistribution; not bundled | https://bdd-data.berkeley.edu/ |
| OpenCV | Video, image decoding, drawing, geometry | Apache-2.0 code; video codec/build notices vary | https://github.com/opencv/opencv/blob/4.x/LICENSE |
| NumPy / SciPy | Arrays and image/loss calculations | BSD-family and bundled component notices; inspect installed distributions | https://numpy.org/doc/stable/license.html ; https://github.com/scipy/scipy/blob/main/LICENSE.txt |
| Pillow | Image support | MIT-CMU | https://github.com/python-pillow/Pillow/blob/main/LICENSE |
| PyYAML | Configuration | MIT | https://github.com/yaml/pyyaml/blob/main/LICENSE |
| Matplotlib | Reports/plots | Matplotlib/PSF-based licence and bundled notices | https://matplotlib.org/stable/project/license.html |
| ONNX / ONNX Runtime | Model format and execution | Apache-2.0 / MIT, plus runtime dependencies | https://github.com/onnx/onnx/blob/main/LICENSE ; https://github.com/microsoft/onnxruntime/blob/main/LICENSE |
| pyserial | Optional serial transport | BSD-3-Clause | https://github.com/pyserial/pyserial/blob/master/LICENSE.txt |
| imageio-ffmpeg / FFmpeg | Optional output re-encoding | Python wrapper BSD-2-Clause; FFmpeg binary licence depends on build/configuration | https://github.com/imageio/imageio-ffmpeg/blob/main/LICENSE ; https://ffmpeg.org/legal.html |
| lap | Tracker assignment | BSD-2-Clause in installed metadata | https://github.com/gatagat/lap |
| pytest / pyflakes | Tests and lint | MIT | https://github.com/pytest-dev/pytest/blob/main/LICENSE ; https://github.com/PyCQA/pyflakes/blob/main/LICENSE |
| Arduino core / Servo | Firmware runtime and servo output | Review selected board core and Servo library notices; not vendored here | https://github.com/arduino/ArduinoCore-avr ; https://github.com/arduino-libraries/Servo |

The software licence does not establish original authorship of the vehicle or compliance with competition rules. The test input video is user-supplied and is not redistributed. Model weights, dataset images and generated videos are excluded from Git.

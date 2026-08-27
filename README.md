# Emotians

Real-time facial-expression recognition from a laptop webcam. The project uses
OpenCV for capture and visualization, and trains its neural-network components
from scratch. It does **not** load FER, DeepFace, Haar cascades, or any other
pre-trained model.

Start with the project blueprint in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
For an implementation-focused Stage 1 report in Spanish, see
[docs/AVANCE_1.md](docs/AVANCE_1.md).
The Stage 2 dataset and from-scratch classifier guide is in
[docs/AVANCE_2.md](docs/AVANCE_2.md).
The Stage 3 detector and live-inference guide is in
[docs/AVANCE_3.md](docs/AVANCE_3.md).

## Stage 1: camera preview

Create a Python 3.10+ virtual environment, then install and run the console
preview from the repository root:

```text
python -m pip install -e ".[dev]"
python -m emotians.app --config configs/preview.yaml --mode preview
```

The yellow rectangle is a positioning guide only. This first advance has no
face detector and no expression model. Press `q` or `Esc` to close it.

Run the non-camera tests with `python -m pytest`.

## Collect expression photos

To capture your own labeled expression crops with the webcam, open the capture
mode and press `c` whenever you want to save the current guided crop:

```text
python -m emotians.app --mode collect-expressions --camera 0 --subject-id subject_01 --capture-label happy
```

Saved images go to:

```text
data/raw/expressions/<subject-id>/<capture-label>/
```

Repeat the command changing `--capture-label` for `sad`, `angry`, and
`surprised`. After collecting at least three subjects, regenerate the expression
split manifest with `--mode prepare-data`.

## Stage 3: pretrained detector and live demo

The live path uses the pretrained YuNet face detector and the expression model:

```text
camera frame -> YuNet -> face crop -> EmotionCNN -> smoothing -> overlay
```

The YuNet ONNX model is stored at:

```text
artifacts/models/face_detection_yunet_2023mar.onnx
```

Run the live demo with only the expression checkpoint from this project:

```text
python -m emotians.app --config configs/preview.yaml --mode live-demo --face-model artifacts/models/face_detection_yunet_2023mar.onnx --expression-checkpoint artifacts/runs/stage_2_balanced/best_expression_cnn.pt
```

FER2013 and the expression-capture dataset are used only to train the
expression classifier. The old detector training commands remain available for
historical experiments but are not used by `live-demo`.

## Web application

Run the browser application locally:

```text
python -m pip install -e .
emotians-web
```

Open `http://127.0.0.1:8000`. The browser requests camera permission and sends
periodic JPEG frames to the Python API. On Render, use the included
`render.yaml`; its HTTPS URL is required by browsers for camera access.

The web app uses the balanced checkpoint in
`artifacts/runs/stage_2_balanced/best_expression_cnn.pt`. On the test split,
`surprised` recall improved from 80.26% to 86.52%; overall accuracy changed
from 79.67% to 78.30%.

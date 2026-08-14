# EMOTIANS architecture and delivery stages

## Scope and source-of-truth decisions

The supplied brief requires a laptop-webcam demo that draws a face rectangle and
shows the dominant expression in real time. Its listed classes are:

`happy`, `sad`, `angry`, `surprised`.

The brief mentions FER/DeepFace as possible expression components. They are not
part of this design: the project requirement takes priority, so every learned
model is trained from project data from random initialization. This also means a
pre-trained OpenCV Haar cascade is out of scope.

OpenCV is used for camera I/O, image transforms, drawing, and the console demo.
PyTorch is used only to define and train our own neural networks.

## Target runtime flow

```text
Laptop webcam (OpenCV VideoCapture)
        |
        v
Frame validation / resize / color conversion
        |
        +--> Face detector trained from scratch ----> face bounding box
        |                                               |
        v                                               v
Central-ROI fallback (Stage 1 only)              crop + align + normalize
                                                        |
                                                        v
                                           expression CNN trained from scratch
                                                        |
                                                        v
                                         temporal smoothing + confidence gate
                                                        |
                                                        v
                         OpenCV overlay: rectangle, label, confidence, FPS
```

The central-ROI fallback is deliberately temporary. It makes the early console
demo usable without smuggling in a pre-trained detector; Stage 3 replaces it
with our trained face detector.

## Repository layout

```text
src/emotians/
  app/             # console entry point and webcam loop
  config/          # typed settings and class mapping
  data/            # import, validation, splitting, augmentation
  vision/          # OpenCV transforms, ROI selection, rendering
  models/          # from-scratch detector and expression CNN definitions
  training/        # train/evaluate/checkpoint loops
  inference/       # model loading, prediction, smoothing
  utils/           # logging, reproducibility, paths
tests/             # unit and smoke tests
configs/           # YAML experiment settings
data/
  raw/             # ignored source images/videos
  processed/       # ignored derived datasets
artifacts/
  checkpoints/     # ignored trained weights
  runs/            # ignored metrics and samples
docs/
```

## Four development stages

### Stage 1 — Foundation and architecture

**Goal:** create a reproducible console application shell and prove that the
webcam/overlay pipeline works without any learned or downloaded model.

Deliverables:

- Python environment and pinned dependency lockfile.
- `src/emotians` package, configuration schema, structured logging, and CLI.
- OpenCV webcam loop with clean shutdown, FPS, central ROI guidance, and a
  placeholder label such as `collecting data`.
- Dataset contract: image format, label names, consent policy, and
  train/validation/test subject-disjoint split manifest.
- Smoke tests for configuration, preprocessing shapes, and camera-free overlay
  rendering.

Exit criteria: `python -m emotians.app --camera 0 --mode preview` opens a
window, draws only a guidance rectangle, and exits with `q` or `Esc`.

**Current implementation:** the Stage 1 preview shell is in `src/emotians`.
It deliberately has no inference path: the yellow rectangle is only a capture
guide. Run it after installing dependencies with:

```text
python -m pip install -e .
python -m emotians.app --config configs/preview.yaml --mode preview
```

### Stage 2 — Data pipeline and expression classifier

**Goal:** train a four-class expression classifier from random initialization.

Deliverables:

- Data importer, validator, deduplication report, subject-disjoint splits, and
  label-balanced sampling.
- OpenCV preprocessing: BGR-to-grayscale or RGB selection, crop, resize to
  `48x48`, pixel normalization, and training-only augmentation.
- A compact custom CNN: convolution/block stacks, global average pooling,
  dropout, and a four-logit output layer. No external weights are loaded.
- Training, validation, checkpoint, and evaluation commands; confusion matrix
  and per-class precision/recall/F1.

Exit criteria: a saved checkpoint can reproduce its validation metrics and the
test report includes all four classes. A result is not accepted merely because
overall accuracy is high while one expression is poorly represented.

### Stage 3 — Face detector and real-time integration

**Goal:** replace the temporary ROI with a face-localization model trained from
scratch, then run complete inference from webcam to overlay.

Deliverables:

- Face-box annotations and detector dataset pipeline. The selected approach
  should be a small custom single-class detector, with anchors/target encoding
  documented in its config.
- Detector training/evaluation using IoU, precision, recall, and false-positive
  inspection; no Haar/DNN cascade or imported detector weights.
- Inference orchestration: detector -> crop -> expression classifier ->
  confidence threshold -> exponential moving-average label smoothing.
- OpenCV renderer with a red/green face rectangle, expression label,
  confidence, and measured FPS.

Exit criteria: the console demo finds a visible face, overlays one of the four
labels, and handles missed detections safely (no stale crop or crash).

### Stage 4 — Validation, packaging, and demonstration

**Goal:** make the prototype dependable and easy to demonstrate.

Deliverables:

- End-to-end tests using recorded consented clips, including no-face, multiple
  face, low-light, and abrupt-expression cases.
- Error review by class and lighting; document known limits and do not infer
  emotional state beyond the four visible-expression categories.
- Console commands for preview, training, evaluation, and live demo; setup and
  troubleshooting guide.
- Reproducible configuration, checkpoint metadata, demo recording steps, and
  a final acceptance checklist.

Exit criteria: a new machine can follow the README, run the live demo from a
known checkpoint, and see a face rectangle plus dominant expression label.

## Interfaces to stabilize in Stage 1

```python
# models/contracts.py
class FaceDetector(Protocol):
    def detect(self, bgr_frame: np.ndarray) -> list[Detection]: ...

class ExpressionClassifier(Protocol):
    def predict(self, face_bgr: np.ndarray) -> Prediction: ...

# inference/pipeline.py
def process_frame(frame_bgr: np.ndarray) -> FrameResult: ...
```

`Detection` holds `xyxy` coordinates and a confidence score. `Prediction`
holds one of the four fixed labels, a confidence score, and the complete
probability vector. Keeping these contracts independent lets the Stage 1
central-ROI adapter be swapped for the Stage 3 detector without changing the
webcam loop.

## Data and ethics guardrails

- Capture and retain facial data only with informed consent.
- Keep participant identities separate from labels and use subject-disjoint
  splits to avoid identity leakage.
- Store raw images and trained weights outside version control.
- Describe output as *visible facial-expression classification*, not a reliable
  determination of someone’s internal emotion.

# Emotians

Real-time facial-expression recognition from a laptop webcam. The project uses
OpenCV for capture and visualization, and trains its neural-network components
from scratch. It does **not** load FER, DeepFace, Haar cascades, or any other
pre-trained model.

Start with the project blueprint in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
For an implementation-focused Stage 1 report in Spanish, see
[docs/AVANCE_1.md](docs/AVANCE_1.md).

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

"""
YOLO training script for beam structure detection.
Customize dataset paths and parameters before running.

Usage:
    python yolo/train.py
"""

from ultralytics import YOLO


def train():
    # Start from a pretrained YOLO model
    model = YOLO("yolo11n.pt")  # or yolo11s.pt / yolo11m.pt for better accuracy

    results = model.train(
        data="/d/beam_analysis/yolo/dataset/data.yaml",  # dataset config (see below)
        epochs=100,
        imgsz=640,
        batch=16,
        name="beam_detector",
        patience=20,   # early stopping
        device=0,      # GPU; use "cpu" if no CUDA
        workers=4,
        lr0=0.001,
        cos_lr=True,
        augment=True,
        val=True,
        save=True,
        save_period=10,
    )

    # Export to best.pt is auto-saved; also export to ONNX
    model.export(format="onnx")
    print("Training complete. Best model saved in runs/detect/beam_detector/weights/")


if __name__ == "__main__":
    train()

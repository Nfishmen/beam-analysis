"""
One-click: Generate synthetic training data + train YOLO model.

Usage:
    python train_all.py              # full: generate + train
    python train_all.py --skip-gen   # train only (data already generated)
    python train_all.py --gen-only   # generate data only
"""

import sys, os, argparse
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))


def step_generate():
    print("\n" + "=" * 55)
    print("  STEP 1: Generate synthetic training data")
    print("=" * 55)
    from generate_training_data import generate, TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL, N_TRAIN, N_VAL
    generate(N_TRAIN, TRAIN_IMG, TRAIN_LBL, "train_")
    generate(N_VAL, VAL_IMG, VAL_LBL, "val_")
    t = len(list(TRAIN_IMG.glob("*.jpg"))) if TRAIN_IMG.exists() else 0
    v = len(list(VAL_IMG.glob("*.jpg"))) if VAL_IMG.exists() else 0
    print(f"  Total images generated: {t + v}")


def step_train():
    print("\n" + "=" * 55)
    print("  STEP 2: Train YOLO model")
    print("=" * 55)

    data_yaml = BASE / "yolo" / "dataset" / "data.yaml"
    if not data_yaml.exists():
        print("  ERROR: data.yaml not found. Run with --gen-only first?")
        sys.exit(1)

    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=16,
        name="beam_detector",
        patience=20,
        device=0,
        workers=4,
        lr0=0.001,
        cos_lr=True,
        augment=True,
        val=True,
        save=True,
        save_period=10,
    )

    model.export(format="onnx")
    print("\n" + "=" * 55)
    print("  Training complete!")
    print("  Model: runs/detect/beam_detector/weights/best.pt")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--gen-only", action="store_true")
    args = parser.parse_args()

    if args.gen_only:
        step_generate()
    elif args.skip_gen:
        step_train()
    else:
        step_generate()
        step_train()

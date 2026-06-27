"""
YOLO beam structure detector.
Loads a trained YOLO model and runs inference on images.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class BeamDetector:
    """YOLO-based detector for beam structures, supports, and loads."""

    CLASS_NAMES = {
        0: "beam",
        1: "fixed_support",
        2: "pinned_support",
        3: "roller_support",
        4: "point_load",
        5: "distributed_load",
    }

    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

    def detect(self, image: np.ndarray | str) -> list[dict]:
        """
        Run detection on an image.

        Args:
            image: image path or numpy array (BGR)

        Returns:
            List of detection dicts with keys:
                class_name, class_id, confidence, bbox (x1,y1,x2,y2), center (cx,cy)
        """
        results = self.model(image, conf=self.conf, iou=self.iou, verbose=False)
        detections = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf_val = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = xyxy.tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            detections.append({
                "class_name": self.CLASS_NAMES.get(cls_id, f"unknown_{cls_id}"),
                "class_id": cls_id,
                "confidence": round(conf_val, 4),
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
            })

        return detections

    def predict_and_visualize(self, image_path: str, save: bool = True) -> np.ndarray:
        """Run YOLO predict and return the annotated image."""
        results = self.model(image_path, conf=self.conf, iou=self.iou)
        annotated = results[0].plot()
        if save:
            out_path = Path(image_path).stem + "_detected.jpg"
            cv2.imwrite(out_path, annotated)
        return annotated

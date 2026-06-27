"""
Convert YOLO detections into a BeamModel for structural analysis.
"""

import numpy as np
from src.mechanics import BeamModel, BeamType, CrossSection


class BeamGeometryParser:
    """
    Parses raw YOLO detection results into a structured BeamModel.
    Uses pixel coordinates; a scale_factor converts px → meters.
    """

    def __init__(self, scale_factor: float = 0.001, image_width: int = 640, image_height: int = 480):
        """
        Args:
            scale_factor: meters per pixel (user must calibrate)
            image_width, image_height: image dimensions in pixels
        """
        self.scale = scale_factor
        self.img_w = image_width
        self.img_h = image_height

    def parse(self, detections: list[dict], cross_section: dict | None = None) -> BeamModel:
        """
        Convert a list of detection dicts into a BeamModel.

        Detections expected classes:
            beam, fixed_support, pinned_support, roller_support,
            point_load, distributed_load
        """
        beams = [d for d in detections if d["class_name"] == "beam"]
        supports = [d for d in detections if "support" in d["class_name"]]
        point_loads = [d for d in detections if d["class_name"] == "point_load"]
        dist_loads = [d for d in detections if d["class_name"] == "distributed_load"]

        if not beams:
            raise ValueError("No beam detected in the image")

        # use the largest beam detection
        beam = max(beams, key=lambda d: self._bbox_area(d["bbox"]))
        beam_left, beam_right = self._get_beam_span(beam)

        length = (beam_right - beam_left) * self.scale

        parsed_supports = self._parse_supports(supports, beam_left, beam_right)
        parsed_points = self._parse_point_loads(point_loads, beam_left, beam_right)
        parsed_dist = self._parse_distributed_loads(dist_loads, beam_left, beam_right)

        return BeamModel(
            length=length,
            beam_type=BeamType.SIMPLY_SUPPORTED,  # auto-classified later
            supports=parsed_supports,
            point_loads=parsed_points,
            distributed_loads=parsed_dist,
            cross_section=cross_section or {
                "type": CrossSection.RECTANGLE,
                "width": 0.1,
                "height": 0.2,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bbox_area(self, bbox) -> float:
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def _get_beam_span(self, beam: dict) -> tuple[float, float]:
        """Return (left_x, right_x) of the beam in pixels."""
        x1, _, x2, _ = beam["bbox"]
        return x1, x2

    def _position_along_beam(self, cx: float, beam_left: float, beam_right: float) -> float:
        """Project a detection center onto the beam axis, clamped to [0, length_m]."""
        t = (cx - beam_left) / (beam_right - beam_left)
        t = max(0.0, min(1.0, t))
        return t * (beam_right - beam_left) * self.scale

    def _parse_supports(self, supports: list[dict], beam_left: float, beam_right: float) -> list[dict]:
        result = []
        for s in supports:
            cx, _ = s["center"]
            result.append({
                "type": s["class_name"].replace("_support", ""),  # "fixed", "pinned", "roller"
                "position": self._position_along_beam(cx, beam_left, beam_right),
            })
        return result

    def _parse_point_loads(self, loads: list[dict], beam_left: float, beam_right: float) -> list[dict]:
        result = []
        for pl in loads:
            cx, cy = pl["center"]
            result.append({
                "position": self._position_along_beam(cx, beam_left, beam_right),
                "value": -10000.0,  # placeholder, user can override
            })
        return result

    def _parse_distributed_loads(self, loads: list[dict], beam_left: float, beam_right: float) -> list[dict]:
        result = []
        for dl in loads:
            x1, _, x2, _ = dl["bbox"]
            result.append({
                "start": self._position_along_beam(x1, beam_left, beam_right),
                "end": self._position_along_beam(x2, beam_left, beam_right),
                "value": -5000.0,  # placeholder N/m
            })
        return result

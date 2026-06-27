"""
Full analysis pipeline: image → YOLO detect → geometry parse → mechanics → visualize.

Usage:
    from pipeline import BeamAnalysisPipeline

    pipe = BeamAnalysisPipeline()
    result = pipe.run("path/to/image.jpg", output_dir="static/results")
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.mechanics import BeamModel, BeamAnalyzer, BeamType, CrossSection, BeamResults
from src.visualize import plot_analysis_report, plot_stress_distribution, plot_with_detection
from src.geometry import BeamGeometryParser


# ---------------------------------------------------------------------------
# Demo cases — used when no YOLO model is available
# ---------------------------------------------------------------------------

def _demo_simply_supported():
    """Simply-supported beam, point load at mid-span."""
    return BeamModel(
        length=4.0,
        beam_type=BeamType.SIMPLY_SUPPORTED,
        supports=[
            {"type": "pinned", "position": 0.0},
            {"type": "roller", "position": 4.0},
        ],
        point_loads=[{"position": 2.0, "value": -20000}],
        distributed_loads=[],
        cross_section={"type": CrossSection.RECTANGLE, "width": 0.15, "height": 0.3},
        material={"E": 200e9, "nu": 0.3, "yield_stress": 250e6},
    )


def _demo_cantilever():
    """Cantilever beam with uniform distributed load."""
    return BeamModel(
        length=2.5,
        beam_type=BeamType.CANTILEVER,
        supports=[{"type": "fixed", "position": 0.0}],
        point_loads=[],
        distributed_loads=[{"start": 0.0, "end": 2.5, "value": -8000}],
        cross_section={
            "type": CrossSection.I_BEAM,
            "flange_width": 0.12, "flange_thickness": 0.012,
            "web_thickness": 0.008, "height": 0.25,
        },
        material={"E": 200e9, "nu": 0.3, "yield_stress": 250e6},
    )


DEMO_MODELS = [_demo_simply_supported(), _demo_cantilever()]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class BeamAnalysisPipeline:
    """
    Full pipeline: image → detect → parse → analyze → visualize.

    Tries to load a YOLO model; falls back to demo mode when no model is found.
    """

    def __init__(self, model_path: str | None = None):
        """
        Args:
            model_path: path to trained YOLO .pt file.
                        Auto-searches for 'best.pt' if not provided.
        """
        self.detector = None
        self.has_yolo = False
        self._init_detector(model_path)

    def _init_detector(self, model_path: str | None):
        """Try to load YOLO; remain in demo mode on failure."""
        _root = Path(__file__).parent  # D:\beam_analysis — always correct
        if model_path is None:
            # search common locations (use absolute paths)
            candidates = [
                _root / "best.pt",
                _root / "yolo" / "best.pt",
                _root / "runs" / "detect" / "beam_detector" / "weights" / "best.pt",
                _root / "runs" / "detect" / "beam_detector-2" / "weights" / "best.pt",
            ]
            for c in candidates:
                if c.exists():
                    model_path = str(c)
                    break

        if model_path and os.path.exists(model_path):
            try:
                from src.detector import BeamDetector
                self.detector = BeamDetector(model_path)
                self.has_yolo = True
                print(f"[pipeline] YOLO model loaded: {model_path}")
            except Exception as e:
                print(f"[pipeline] Failed to load YOLO model: {e}")
                print("[pipeline] Falling back to demo mode.")
        else:
            print("[pipeline] No YOLO model found. Running in demo mode.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, image_path: str, output_dir: str = "static/results") -> dict:
        """
        Run the full analysis pipeline.

        Args:
            image_path: path to uploaded beam diagram image
            output_dir: directory for generated chart PNGs

        Returns:
            dict with keys:
                success, mode, beam_model, results, charts, detections
        """
        os.makedirs(output_dir, exist_ok=True)
        stem = Path(image_path).stem

        # --- 1. Detection ---
        detections = []
        beam_model = None
        mode = "demo"

        if self.has_yolo:
            try:
                detections = self.detector.detect(image_path)
            except Exception as e:
                print(f"[pipeline] YOLO detection failed: {e}, falling back to demo")
                detections = []
            if detections:
                parser = BeamGeometryParser()
                try:
                    beam_model = parser.parse(detections)
                    mode = "yolo"
                except ValueError as e:
                    print(f"[pipeline] YOLO parse failed: {e}, using demo")

        # fall back to demo
        if beam_model is None:
            # pick demo model based on a hash of the filename (for variety)
            demo_idx = hash(stem) % len(DEMO_MODELS)
            beam_model = DEMO_MODELS[demo_idx]

        # --- 2. Mechanics analysis ---
        analyzer = BeamAnalyzer()
        results = analyzer.analyze(beam_model)

        # --- 3. Visualization ---
        report_path = os.path.join(output_dir, f"{stem}_report.png")
        stress_path = os.path.join(output_dir, f"{stem}_stress.png")

        plot_analysis_report(results, save_path=report_path)
        plot_stress_distribution(results, save_path=stress_path)

        charts = {
            "report": "/" + report_path.replace("\\", "/"),
            "stress": "/" + stress_path.replace("\\", "/"),
        }

        # detection overlay (only when YOLO is active)
        if self.has_yolo and detections:
            overlay_path = os.path.join(output_dir, f"{stem}_detected.jpg")
            plot_with_detection(
                __import__("cv2").imread(image_path), detections, save_path=overlay_path
            )
            charts["detection_overlay"] = "/" + overlay_path.replace("\\", "/")

        # --- 4. Build response ---
        return {
            "success": True,
            "mode": mode,
            "beam_model": self._serialize_beam(beam_model),
            "results": self._serialize_results(results),
            "charts": charts,
            "detections": detections,
        }

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_beam(beam: BeamModel) -> dict:
        def serialize_cs(cs):
            d = dict(cs)
            d["type"] = d["type"].value if isinstance(d["type"], CrossSection) else str(d["type"])
            return d

        return {
            "length": beam.length,
            "beam_type": beam.beam_type.value,
            "supports": beam.supports,
            "point_loads": beam.point_loads,
            "distributed_loads": beam.distributed_loads,
            "cross_section": serialize_cs(beam.cross_section),
            "material": beam.material,
        }

    @staticmethod
    def _serialize_results(r: BeamResults) -> dict:
        # convert reaction keys from tuples to strings
        reactions_str = {}
        for k, v in r.reactions.items():
            key_str = f"{k[0]:.2f}m_{k[1]}"
            reactions_str[key_str] = v

        return {
            "length": r.beam.length,
            "beam_type": r.beam.beam_type.value,
            "reactions": reactions_str,
            "max_shear": round(r.max_shear, 1),
            "max_moment": round(r.max_moment, 1),
            "max_deflection": round(r.max_deflection, 6),
            "max_stress": round(r.max_stress, 1),
            # keep a few sample points for optional front-end charts
            "positions_sample": r.positions[::50].tolist(),
            "shear_sample": r.shear[::50].tolist(),
            "moment_sample": r.moment[::50].tolist(),
            "deflection_sample": (r.deflection * 1000)[::50].tolist(),  # mm
        }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pipe = BeamAnalysisPipeline()
    # test with a non-existent image → falls back to demo
    result = pipe.run("test.jpg", output_dir="static/results")
    print(f"Mode: {result['mode']}")
    print(f"Beam type: {result['results']['beam_type']}")
    print(f"Max moment: {result['results']['max_moment']:.1f} N·m")
    print(f"Charts: {result['charts']}")
    print("Pipeline test OK ✓")

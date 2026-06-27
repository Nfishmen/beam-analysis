"""
Main entry point — demo pipeline with a manually constructed beam model.
Replace with YOLO detection pipeline once your model is trained.
"""

import sys
sys.path.insert(0, "/d/beam_analysis")

import numpy as np
from src.mechanics import BeamModel, BeamAnalyzer, BeamType, CrossSection
from src.visualize import plot_analysis_report, plot_stress_distribution


def demo_simply_supported():
    """Simply-supported beam with a point load at mid-span."""
    beam = BeamModel(
        length=4.0,
        beam_type=BeamType.SIMPLY_SUPPORTED,
        supports=[
            {"type": "pinned", "position": 0.0},
            {"type": "roller", "position": 4.0},
        ],
        point_loads=[{"position": 2.0, "value": -20000}],  # 20 kN downward
        distributed_loads=[],
        cross_section={"type": CrossSection.RECTANGLE, "width": 0.15, "height": 0.3},
        material={"E": 200e9, "nu": 0.3, "yield_stress": 250e6},
    )

    analyzer = BeamAnalyzer()
    results = analyzer.analyze(beam)

    print(f"=== {beam.beam_type.value} Beam Analysis ===")
    print(f"Length: {beam.length} m")
    print(f"Reactions: {results.reactions}")
    print(f"Max shear:     {results.max_shear:.1f} N")
    print(f"Max moment:    {results.max_moment:.1f} N·m")
    print(f"Max deflection: {results.max_deflection*1e3:.3f} mm")
    print(f"Max stress:     {results.max_stress/1e6:.1f} MPa")

    # Generate report figures
    report_path = plot_analysis_report(results, save_path="beam_report.png")
    print(f"Report saved: {report_path}")

    stress_path = plot_stress_distribution(results, save_path="stress_distribution.png")
    print(f"Stress distribution saved: {stress_path}")


def demo_cantilever():
    """Cantilever beam with uniform distributed load."""
    beam = BeamModel(
        length=2.5,
        beam_type=BeamType.CANTILEVER,
        supports=[{"type": "fixed", "position": 0.0}],
        point_loads=[],
        distributed_loads=[{"start": 0.0, "end": 2.5, "value": -8000}],  # 8 kN/m
        cross_section={"type": CrossSection.I_BEAM, "flange_width": 0.12, "flange_thickness": 0.012,
                       "web_thickness": 0.008, "height": 0.25},
    )

    analyzer = BeamAnalyzer()
    results = analyzer.analyze(beam)

    print(f"\n=== {beam.beam_type.value} Beam Analysis ===")
    print(f"Length: {beam.length} m")
    print(f"Reactions: {results.reactions}")
    print(f"Max shear:     {results.max_shear:.1f} N")
    print(f"Max moment:    {results.max_moment:.1f} N·m")
    print(f"Max deflection: {results.max_deflection*1e3:.3f} mm")
    print(f"Max stress:     {results.max_stress/1e6:.1f} MPa")

    report_path = plot_analysis_report(results, save_path="cantilever_report.png")
    print(f"Report saved: {report_path}")

    stress_path = plot_stress_distribution(results, save_path="cantilever_stress.png")
    print(f"Stress distribution saved: {stress_path}")


if __name__ == "__main__":
    demo_simply_supported()
    demo_cantilever()

"""
Visualization tools for beam analysis results.
Produces SFD, BMD, deflection curves, and detection overlays.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.mechanics import BeamResults, BeamModel, CrossSection


# Chinese-capable font fallback
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_analysis_report(results: BeamResults, save_path: str = "beam_report.png") -> str:
    """
    Generate a 3×2 report figure:
      Row 1: SFD | BMD | Deflection
      Row 2: Slope | Bending Stress | Cross-section
    """
    fig = plt.figure(figsize=(18, 10), layout="constrained")
    x = results.positions
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # --- Shear Force Diagram ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(x, results.shear, 0, alpha=0.3, color="steelblue")
    ax1.plot(x, results.shear, color="steelblue", linewidth=1.5)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_title("SFD — Shear Force", fontweight="bold")
    ax1.set_xlabel("Position (m)")
    ax1.set_ylabel("Shear (N)")
    ax1.grid(True, alpha=0.3)

    # --- Bending Moment Diagram ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(x, results.moment, 0, alpha=0.3, color="darkorange")
    ax2.plot(x, results.moment, color="darkorange", linewidth=1.5)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_title("BMD — Bending Moment", fontweight="bold")
    ax2.set_xlabel("Position (m)")
    ax2.set_ylabel("Moment (N·m)")
    ax2.grid(True, alpha=0.3)

    # --- Deflection ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.fill_between(x, results.deflection * 1000, 0, alpha=0.3, color="seagreen")
    ax3.plot(x, results.deflection * 1000, color="seagreen", linewidth=1.5)
    ax3.axhline(y=0, color="black", linewidth=0.5)
    ax3.set_title("Deflection Curve", fontweight="bold")
    ax3.set_xlabel("Position (m)")
    ax3.set_ylabel("Deflection (mm)")
    ax3.grid(True, alpha=0.3)

    # --- Slope ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(x, results.slope * 1000, color="mediumpurple", linewidth=1.5)
    ax4.axhline(y=0, color="black", linewidth=0.5)
    ax4.set_title("Slope (θ)", fontweight="bold")
    ax4.set_xlabel("Position (m)")
    ax4.set_ylabel("Slope (mrad)")
    ax4.grid(True, alpha=0.3)

    # --- Bending Stress ---
    ax5 = fig.add_subplot(gs[1, 1])
    stress_mpa = results.bending_stress / 1e6
    ax5.fill_between(x, stress_mpa, 0, alpha=0.3, color="firebrick")
    ax5.plot(x, stress_mpa, color="firebrick", linewidth=1.5)
    ax5.axhline(y=0, color="black", linewidth=0.5)
    ax5.set_title("Max Bending Stress", fontweight="bold")
    ax5.set_xlabel("Position (m)")
    ax5.set_ylabel("Stress (MPa)")
    ax5.grid(True, alpha=0.3)

    # --- Cross-section sketch ---
    ax6 = fig.add_subplot(gs[1, 2])
    _draw_cross_section(ax6, results.beam.cross_section)
    ax6.set_title("Cross Section", fontweight="bold")
    ax6.set_aspect("equal")

    # summary subtitle
    summary = (
        f"L={results.beam.length:.2f}m | "
        f"V_max={results.max_shear:.1f}N | "
        f"M_max={results.max_moment:.1f}N·m | "
        f"v_max={results.max_deflection*1e3:.2f}mm | "
        f"σ_max={results.max_stress/1e6:.1f}MPa"
    )
    fig.suptitle(summary, y=0.99, fontsize=10, color="gray")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_with_detection(image: np.ndarray, detections: list[dict], save_path: str = "detection_overlay.jpg") -> str:
    """Draw detection boxes on the original image."""
    import cv2

    colors = {
        "beam": (0, 255, 0),
        "fixed_support": (255, 0, 0),
        "pinned_support": (0, 0, 255),
        "roller_support": (255, 191, 0),
        "point_load": (255, 0, 255),
        "distributed_load": (0, 255, 255),
    }

    img = image.copy()
    for d in detections:
        color = colors.get(d["class_name"], (128, 128, 128))
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f'{d["class_name"]} {d["confidence"]:.2f}'
        cv2.putText(img, label, (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(save_path, img)
    return save_path


def plot_stress_distribution(results: BeamResults, save_path: str = "stress_distribution.png") -> str:
    """
    Plot bending + shear stress distribution across the cross-section
    at the location of maximum moment.
    """
    beam = results.beam
    cs = beam.cross_section
    h = cs.get("height", 0.2)
    b = cs.get("width", 0.1)

    y = np.linspace(-h / 2, h / 2, 100)

    # Bending stress: σ = M·y / I  (linear)
    I = _moment_of_inertia_viz(cs)
    idx_max = np.argmax(np.abs(results.moment))
    M_max = results.moment[idx_max]
    sigma_bend = M_max * y / I

    # Shear stress (approximate parabolic for rectangle): τ = VQ/(Ib)
    # Q = b*(h/2 - y)*(h/4 + y/2) for rectangle → τ = 3V/(2A) * (1 - (2y/h)²)
    idx_shear = np.argmax(np.abs(results.shear))
    V_max_val = results.shear[idx_shear]
    A = b * h
    tau = 1.5 * V_max_val / A * (1 - (2 * y / h) ** 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    ax1.plot(sigma_bend / 1e6, y * 1000, color="firebrick", linewidth=2)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.axvline(x=0, color="black", linewidth=0.5)
    ax1.set_xlabel("Bending Stress (MPa)")
    ax1.set_ylabel("Height (mm)")
    ax1.set_title("Bending Stress Distribution", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(tau / 1e6, y * 1000, color="royalblue", linewidth=2)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.axvline(x=0, color="black", linewidth=0.5)
    ax2.set_xlabel("Shear Stress (MPa)")
    ax2.set_ylabel("Height (mm)")
    ax2.set_title("Shear Stress Distribution", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Stress at Critical Section  (M={M_max:.1f} N·m, V={V_max_val:.1f} N)")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _draw_cross_section(ax, cs: dict):
    t = cs["type"]
    if t == CrossSection.RECTANGLE:
        w, h = cs["width"], cs["height"]
        rect = plt.Rectangle((-w / 2, -h / 2), w, h, facecolor="lightgray",
                              edgecolor="black", linewidth=2)
        ax.add_patch(rect)
        ax.set_xlim(-w, w)
        ax.set_ylim(-h, h)
        ax.text(0, 0, f"{w*1000:.0f}×{h*1000:.0f} mm", ha="center", va="center", fontsize=9)
    elif t == CrossSection.CIRCLE:
        d = cs.get("diameter", 0.1)
        circle = plt.Circle((0, 0), d / 2, facecolor="lightgray", edgecolor="black", linewidth=2)
        ax.add_patch(circle)
        ax.set_xlim(-d, d)
        ax.set_ylim(-d, d)
        ax.text(0, 0, f"D={d*1000:.0f} mm", ha="center", va="center", fontsize=9)
    elif t == CrossSection.I_BEAM:
        bf, tf = cs.get("flange_width", 0.1), cs.get("flange_thickness", 0.01)
        tw, h = cs.get("web_thickness", 0.006), cs.get("height", 0.2)
        # top flange
        ax.add_patch(plt.Rectangle((-bf / 2, h / 2 - tf), bf, tf, facecolor="lightgray", edgecolor="black"))
        # bottom flange
        ax.add_patch(plt.Rectangle((-bf / 2, -h / 2), bf, tf, facecolor="lightgray", edgecolor="black"))
        # web
        ax.add_patch(plt.Rectangle((-tw / 2, -h / 2 + tf), tw, h - 2 * tf, facecolor="lightgray", edgecolor="black"))
        ax.set_xlim(-bf / 2 * 1.5, bf / 2 * 1.5)
        ax.set_ylim(-h / 2 * 1.3, h / 2 * 1.3)
    else:
        ax.text(0, 0, "N/A", ha="center", va="center")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    ax.set_xticks([])
    ax.set_yticks([])


def _moment_of_inertia_viz(cs: dict) -> float:
    from src.mechanics import BeamAnalyzer
    return BeamAnalyzer._moment_of_inertia(cs)

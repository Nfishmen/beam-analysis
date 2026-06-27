"""
Generate 2 clear beam structure diagrams for testing, saved to Desktop.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop")
# Use DejaVu Sans which handles basic symbols well
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def draw_support_pinned(ax, x, y):
    tri = mpatches.Polygon(
        [(x - 0.15, y), (x + 0.15, y), (x, y + 0.2)],
        closed=True, facecolor="white", edgecolor="black", linewidth=2)
    ax.add_patch(tri)
    for dx in [-0.25, -0.08, 0.08, 0.25]:
        ax.plot([x + dx, x + dx + 0.06], [y - 0.08, y - 0.15], color="black", linewidth=1.5)


def draw_support_roller(ax, x, y):
    tri = mpatches.Polygon(
        [(x - 0.15, y), (x + 0.15, y), (x, y + 0.2)],
        closed=True, facecolor="white", edgecolor="black", linewidth=2)
    ax.add_patch(tri)
    circle1 = plt.Circle((x - 0.12, y - 0.15), 0.07, facecolor="white", edgecolor="black", linewidth=1.5)
    circle2 = plt.Circle((x + 0.12, y - 0.15), 0.07, facecolor="white", edgecolor="black", linewidth=1.5)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.plot([x - 0.35, x + 0.35], [y - 0.28, y - 0.28], color="black", linewidth=1.5)


def draw_support_fixed(ax, x, y):
    rect = mpatches.FancyBboxPatch(
        (x - 0.25, y - 0.35), 0.5, 0.55,
        boxstyle="round,pad=0.02", facecolor="#555555", edgecolor="black", linewidth=2)
    ax.add_patch(rect)
    for dy in np.linspace(-0.25, 0.15, 12):
        ax.plot([x - 0.2, x + 0.2], [y + dy, y + dy], color="white", linewidth=0.5)


def draw_arrow(ax, x, y, dx, dy, color="red", lw=2.5, label=""):
    ax.arrow(x, y, dx, dy, head_width=0.12, head_length=0.15,
             fc=color, ec=color, linewidth=lw, length_includes_head=True, zorder=5)
    if label:
        ax.text(x + dx / 2 + 0.08, y + dy / 2 + 0.1, label, fontsize=11, color=color, fontweight="bold")


def draw_udl_arrows(ax, x_start, x_end, y_top, n=7, color="red"):
    for i in range(n):
        xi = x_start + (i + 0.5) * (x_end - x_start) / n
        ax.arrow(xi, y_top, 0, -0.4, head_width=0.06, head_length=0.12,
                 fc=color, ec=color, linewidth=1.8, length_includes_head=True, zorder=5)
    ax.text((x_start + x_end) / 2, y_top - 0.65, "w = 8 kN/m",
            fontsize=11, ha="center", color=color, fontweight="bold")


# ===================================================================
# Figure 1: Simply-supported beam with point load at midspan
# ===================================================================
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.set_xlim(-0.5, 8.5)
ax1.set_ylim(-0.8, 2.2)
ax1.set_aspect("equal")
ax1.axis("off")

beam_y = 0.5
ax1.plot([0.5, 7.5], [beam_y, beam_y], color="#2c3e50", linewidth=6, solid_capstyle="round")

# Dimension
ax1.plot([0.5, 7.5], [-0.15, -0.15], color="black", linewidth=1)
ax1.plot([0.5, 0.5], [-0.25, -0.05], color="black", linewidth=1)
ax1.plot([7.5, 7.5], [-0.25, -0.05], color="black", linewidth=1)
ax1.annotate("", xy=(7.5, -0.15), xytext=(0.5, -0.15),
              arrowprops=dict(arrowstyle="<->", color="black", lw=1))
ax1.text(4.0, -0.35, "L = 4 m", fontsize=12, ha="center", fontweight="bold")
ax1.plot([4.0, 4.0], [beam_y, beam_y + 0.8], color="gray", linewidth=1, linestyle="--")
ax1.text(4.0, -0.55, "L/2 = 2 m", fontsize=10, ha="center", color="gray")

# Point load
draw_arrow(ax1, 4.0, beam_y + 1.0, 0, -0.55, color="#e74c3c", label="P = 20 kN")

# Supports
draw_support_pinned(ax1, 0.5, beam_y)
draw_support_roller(ax1, 7.5, beam_y)

# Reaction labels
ax1.text(0.5, beam_y - 0.72, "Ra = 10 kN", fontsize=11, ha="center", color="#2980b9", fontweight="bold")
ax1.text(7.5, beam_y - 0.72, "Rb = 10 kN", fontsize=11, ha="center", color="#2980b9", fontweight="bold")

ax1.set_title("Simply-Supported Beam with Point Load at Midspan\n[simply supported / point load]",
              fontsize=14, fontweight="bold", pad=15)

path1 = os.path.join(OUT_DIR, "beam_simply_supported.png")
fig1.savefig(path1, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig1)
print(f"Saved: {path1}")


# ===================================================================
# Figure 2: Cantilever beam with uniform distributed load
# ===================================================================
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.set_xlim(-0.8, 6.2)
ax2.set_ylim(-0.8, 2.2)
ax2.set_aspect("equal")
ax2.axis("off")

beam_y2 = 0.5
ax2.plot([0.34, 5.5], [beam_y2, beam_y2], color="#2c3e50", linewidth=6, solid_capstyle="round")

# Dimension
ax2.plot([0.34, 5.5], [-0.15, -0.15], color="black", linewidth=1)
ax2.plot([0.34, 0.34], [-0.25, -0.05], color="black", linewidth=1)
ax2.plot([5.5, 5.5], [-0.25, -0.05], color="black", linewidth=1)
ax2.annotate("", xy=(5.5, -0.15), xytext=(0.34, -0.15),
              arrowprops=dict(arrowstyle="<->", color="black", lw=1))
ax2.text(2.92, -0.35, "L = 2.5 m", fontsize=12, ha="center", fontweight="bold")

# UDL
draw_udl_arrows(ax2, 0.34, 5.5, beam_y2 + 0.45, n=10, color="#e74c3c")

# Fixed support
draw_support_fixed(ax2, 0.34, beam_y2)

# Reaction
ax2.text(0.34, beam_y2 - 0.65, "R = 20 kN\nM = 25 kN*m",
         fontsize=11, ha="center", color="#2980b9", fontweight="bold")

ax2.text(5.5, beam_y2 + 0.85, "Free End",
         fontsize=10, ha="center", color="gray")

ax2.set_title("Cantilever Beam with Uniform Distributed Load\n[cantilever / UDL]",
              fontsize=14, fontweight="bold", pad=15)

path2 = os.path.join(OUT_DIR, "beam_cantilever_udl.png")
fig2.savefig(path2, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"Saved: {path2}")

print("\nDone! 2 beam diagrams saved to Desktop.")

"""
Auto-generate synthetic beam structure diagrams with YOLO-format labels.

Output per image:
    yolo/dataset/images/{train,val}/*.jpg   — diagram image
    yolo/dataset/labels/{train,val}/*.txt   — one YOLO bbox per line

Usage:  python generate_training_data.py

The generator programmatically draws beam diagrams with random:
  - beam type (simply-supported, cantilever, fixed-pinned, fixed-fixed, overhanging)
  - support types / positions
  - point loads & distributed loads
  - visual styles (line width, colour, background grid, noise)
Every drawn element records its bounding box → automatically converted to YOLO format.
"""

import os, random, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DS         = BASE_DIR / "yolo" / "dataset"
TRAIN_IMG  = DS / "images" / "train"
TRAIN_LBL  = DS / "labels" / "train"
VAL_IMG    = DS / "images" / "val"
VAL_LBL    = DS / "labels" / "val"

N_TRAIN    = 400
N_VAL      = 100
DPI        = 100

# Class IDs  (must match src/detector.py CLASS_NAMES)
CLS_BEAM             = 0
CLS_FIXED_SUPPORT    = 1
CLS_PINNED_SUPPORT   = 2
CLS_ROLLER_SUPPORT   = 3
CLS_POINT_LOAD       = 4
CLS_DISTRIBUTED_LOAD = 5

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ── helpers ────────────────────────────────────────────────────────────────

def _yolo_line(fig, ax, cls_id, x_ctr, y_ctr, w, h):
    """
    Convert a data-coordinate bbox to a YOLO-format label string.

    Uses matplotlib's transform pipeline so the mapping is exact even
    when axis limits include margins.

    Parameters
    ----------
    fig, ax : Figure / Axes
    cls_id  : int   class label
    x_ctr, y_ctr, w, h : float   centre & size in **data** coordinates
    """
    pw, ph = fig.canvas.get_width_height()               # pixels
    x1d, y1d = ax.transData.transform((x_ctr - w / 2, y_ctr - h / 2))
    x2d, y2d = ax.transData.transform((x_ctr + w / 2, y_ctr + h / 2))

    cx = ((x1d + x2d) / 2) / pw
    cy = 1.0 - ((y1d + y2d) / 2) / ph                     # flip y
    bw = abs(x2d - x1d) / pw
    bh = abs(y2d - y1d) / ph

    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.0005, min(1.0, bw))
    bh = max(0.0005, min(1.0, bh))
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


# ── drawing functions ──────────────────────────────────────────────────────

def _draw_beam(ax, x0, x1, y, color, lw):
    ax.plot([x0, x1], [y, y], color=color, linewidth=lw, solid_capstyle="butt")
    # bbox: horizontal extent = (x1-x0), vertical ≈ 3× line-thickness in data units
    h_data = (lw / 72.0) * 3       # pt → inches → fraction (rough)
    return ("beam", x0, x1, y - h_data / 2, y + h_data / 2)


def _draw_fixed(ax, x, y, color):
    w, h = random.uniform(0.04, 0.07), random.uniform(0.12, 0.20)
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.003", facecolor="#555555",
        edgecolor=color, linewidth=1.5, zorder=10)
    ax.add_patch(rect)
    for dy in np.linspace(y - h / 2 + 0.01, y + h / 2 - 0.01, 8):
        ax.plot([x - w / 2 + 0.004, x + w / 2 - 0.004], [dy, dy],
                color="white", linewidth=0.7)
    return ("fixed_support", x - w / 2, x + w / 2, y - h / 2, y + h / 2)


def _draw_pinned(ax, x, y, color):
    w, h = random.uniform(0.05, 0.09), random.uniform(0.10, 0.17)
    bot, top = y - h / 2, y + h / 2
    tri = mpatches.Polygon(
        [(x - w / 2, bot), (x + w / 2, bot), (x, top)],
        closed=True, facecolor="white", edgecolor=color, linewidth=2, zorder=10)
    ax.add_patch(tri)
    for dx in [-w * 1.2, -w * 0.25, w * 0.25, w * 1.2]:
        ax.plot([x + dx, x + dx + w * 0.2], [bot - 0.015, bot - 0.04],
                color=color, linewidth=1.2)
    return ("pinned_support", x - w * 0.7, x + w * 0.7, bot - 0.04, top)


def _draw_roller(ax, x, y, color):
    w, h = random.uniform(0.05, 0.09), random.uniform(0.10, 0.17)
    bot, top = y - h / 2, y + h / 2
    tri = mpatches.Polygon(
        [(x - w / 2, bot), (x + w / 2, bot), (x, top)],
        closed=True, facecolor="white", edgecolor=color, linewidth=2, zorder=10)
    ax.add_patch(tri)
    for cx_c in [x - w * 0.55, x + w * 0.55]:
        c = plt.Circle((cx_c, bot - 0.05), w * 0.18,
                       facecolor="white", edgecolor=color, linewidth=1.2, zorder=10)
        ax.add_patch(c)
    ax.plot([x - w * 1.0, x + w * 1.0], [bot - 0.11, bot - 0.11],
            color=color, linewidth=1.5)
    return ("roller_support", x - w * 0.7, x + w * 0.7, bot - 0.12, top)


def _draw_point_load(ax, x, y_beam, color):
    arr_len = random.uniform(0.25, 0.45)
    y_top = y_beam + arr_len
    ax.arrow(x, y_top, 0, -arr_len + 0.02,
             head_width=0.025, head_length=0.06,
             fc="#c0392b", ec="#c0392b", linewidth=random.randint(1, 3),
             length_includes_head=True, zorder=8)
    if random.random() < 0.6:
        val = round(random.uniform(5, 50), 1)
        ax.text(x + 0.02, y_top + 0.02, f"P={val}kN", fontsize=9,
                color="#c0392b", fontweight="bold", ha="center")
    return ("point_load", x - 0.05, x + 0.05, y_beam, y_top + 0.04)


def _draw_dist_load(ax, x_start, x_end, y_beam, color):
    n = random.randint(5, 10)
    arr_len = random.uniform(0.18, 0.32)
    y_top = y_beam + arr_len
    for i in range(n):
        xi = x_start + (i + 0.5) * (x_end - x_start) / n
        ax.arrow(xi, y_top, 0, -arr_len + 0.02,
                 head_width=0.015, head_length=0.05,
                 fc="#c0392b", ec="#c0392b",
                 linewidth=random.randint(1, 3), length_includes_head=True, zorder=8)
    ax.plot([x_start, x_end], [y_top, y_top], color="#c0392b", linewidth=2, zorder=7)
    for xm in [x_start, x_end]:
        ax.plot([xm, xm], [y_beam, y_top], color="#c0392b", linewidth=1, linestyle=":", zorder=7)
    if random.random() < 0.6:
        val = round(random.uniform(2, 30), 1)
        ax.text((x_start + x_end) / 2, y_top + 0.03, f"w={val}kN/m",
                fontsize=8, color="#c0392b", fontweight="bold", ha="center")
    return ("distributed_load", x_start, x_end, y_beam, y_top + 0.04)


# ── diagram generator ──────────────────────────────────────────────────────

BEAM_TYPES = ["simply_supported", "cantilever", "fixed_pinned", "fixed_fixed", "overhanging"]
BEAM_COLORS = ["#2c3e50", "#1a1a2e", "#0f0f23", "#16213e", "#0d1b2a"]


def _generate_one(seed: int) -> tuple[np.ndarray, list[str]]:
    random.seed(seed)
    np.random.seed(seed)

    # figure size  (full-bleed axes for clean coordinate mapping)
    w_px = random.randint(560, 960)
    h_px = int(w_px * random.uniform(0.42, 0.65))
    fig, ax = plt.subplots(figsize=(w_px / DPI, h_px / DPI), dpi=DPI)
    ax.set_position([0, 0, 1, 1])          # axes fill the whole figure
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")

    color = random.choice(BEAM_COLORS)
    lw    = random.randint(4, 9)
    records = []   # (label_name, x_min, x_max, y_min, y_max)  in data coords

    # ── beam ──
    margin  = random.uniform(0.08, 0.18)
    x0, x1  = margin, 1.0 - margin
    y_beam  = random.uniform(0.50, 0.65)
    records.append(_draw_beam(ax, x0, x1, y_beam, color, lw))

    # ── supports ──
    beam_type = random.choice(BEAM_TYPES)
    if beam_type == "simply_supported":
        sup_positions = [x0, x1]
        sup_types     = ["pinned", "roller"]
    elif beam_type == "cantilever":
        sup_positions = [x0]
        sup_types     = ["fixed"]
    elif beam_type == "fixed_pinned":
        sup_positions = [x0, x1]
        sup_types     = ["fixed", "pinned"]
    elif beam_type == "fixed_fixed":
        sup_positions = [x0, x1]
        sup_types     = ["fixed", "fixed"]
    else:   # overhanging
        p_mid = random.uniform(x0 + 0.15, x1 - 0.15)
        sup_positions = [x0, p_mid, x1]
        sup_types     = random.choice([
            ["pinned", "roller", "roller"],
            ["pinned", "pinned", "roller"],
            ["fixed", "roller", "roller"],
        ])

    for sx, st in zip(sup_positions, sup_types):
        sy = y_beam - random.uniform(0.02, 0.05)
        if st == "fixed":
            records.append(_draw_fixed(ax, sx, sy, color))
        elif st == "pinned":
            records.append(_draw_pinned(ax, sx, sy + 0.04, color))
        else:
            records.append(_draw_roller(ax, sx, sy + 0.04, color))

    # ── loads ──
    used = []   # (x_lo, x_hi) to avoid overlap

    def _overlap(a, b):
        for ua, ub in used:
            if not (b < ua or a > ub):
                return True
        return False

    n_points = random.randint(0, 3)
    for _ in range(n_points):
        px = random.uniform(x0 + 0.05, x1 - 0.05)
        if _overlap(px - 0.06, px + 0.06):
            continue
        records.append(_draw_point_load(ax, px, y_beam, color))
        used.append((px - 0.06, px + 0.06))

    n_dist = random.randint(0, 2)
    for _ in range(n_dist):
        max_span = (x1 - x0) * 0.85
        if max_span < 0.12:
            continue
        dw = random.uniform(0.12, max_span)
        ds = random.uniform(x0, x1 - dw)
        de = ds + dw
        if _overlap(ds, de):
            continue
        records.append(_draw_dist_load(ax, ds, de, y_beam, color))
        used.append((ds, de))

    # ── optional noise / decorations ──
    if random.random() < 0.3:
        for gy in np.linspace(0.05, 0.95, random.randint(6, 14)):
            ax.axhline(y=gy, color="#e0e0e0", linewidth=0.3, alpha=0.5)
        for gx in np.linspace(0.05, 0.95, random.randint(6, 14)):
            ax.axvline(x=gx, color="#e0e0e0", linewidth=0.3, alpha=0.5)
    if random.random() < 0.2:
        dy = y_beam - random.uniform(0.15, 0.30)
        ax.plot([x0, x1], [dy, dy], color="gray", linewidth=0.6)
        ax.plot([x0, x0], [dy - 0.01, dy + 0.01], color="gray", linewidth=0.6)
        ax.plot([x1, x1], [dy - 0.01, dy + 0.01], color="gray", linewidth=0.6)

    # ── render & convert records → YOLO labels ──
    fig.canvas.draw()
    labels = []
    cls_map = {
        "beam": CLS_BEAM,
        "fixed_support": CLS_FIXED_SUPPORT,
        "pinned_support": CLS_PINNED_SUPPORT,
        "roller_support": CLS_ROLLER_SUPPORT,
        "point_load": CLS_POINT_LOAD,
        "distributed_load": CLS_DISTRIBUTED_LOAD,
    }
    for name, rx0, rx1, ry0, ry1 in records:
        cid = cls_map[name]
        cx_data = (rx0 + rx1) / 2
        cy_data = (ry0 + ry1) / 2
        w_data  = rx1 - rx0
        h_data  = ry1 - ry0
        labels.append(_yolo_line(fig, ax, cid, cx_data, cy_data, w_data, h_data))

    # image array
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    if random.random() < 0.1:
        n_std = random.uniform(2, 7)
        noise = np.random.normal(0, n_std, img.shape).astype(np.uint8)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    plt.close(fig)
    return img[..., :3], labels


# ── batch generator ─────────────────────────────────────────────────────────

def generate(n: int, img_dir: Path, lbl_dir: Path, prefix: str):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    for i in range(n):
        try:
            img, labels = _generate_one(hash(f"{prefix}{i}") % 2**31)
        except Exception as e:
            print(f"  [!] skip {i}: {e}")
            continue
        name = f"{prefix}{i:04d}"
        plt.imsave(img_dir / f"{name}.jpg", img)
        (lbl_dir / f"{name}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        if (i + 1) % 400 == 0:
            print(f"  ... {i + 1}/{n}")
    actual = len(list(img_dir.glob("*.jpg")))
    print(f"  done: {actual} images → {img_dir}")


# ── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Synthetic Beam Diagram Generator")
    print(f"  Target: {N_TRAIN} train + {N_VAL} val")
    print("=" * 55)
    print("\n[1/2] Training set ...")
    generate(N_TRAIN, TRAIN_IMG, TRAIN_LBL, "train_")
    print("\n[2/2] Validation set ...")
    generate(N_VAL, VAL_IMG, VAL_LBL, "val_")

    t = len(list(TRAIN_IMG.glob("*.jpg"))) if TRAIN_IMG.exists() else 0
    v = len(list(VAL_IMG.glob("*.jpg"))) if VAL_IMG.exists() else 0
    print(f"\n{'=' * 55}")
    print(f"  Total: {t + v}  (train={t}  val={v})")
    print(f"  Classes: beam, fixed_support, pinned_support,")
    print(f"           roller_support, point_load, distributed_load")
    print(f"  Ready for: python yolo/train.py")
    print(f"{'=' * 55}")

"""
Figure 1 for the ICBINB paper.

Two panels, stacked for single-column layout:
  (a) system block diagram — electrodes → envelope → classifier → stability filter → actuator
  (b) forearm electrode-placement diagram, four sites labelled with muscle names

Anonymised by construction — no photos, no lab identifiers.

Output:
  analysis/plots/figures/fig1_system_and_placement.png
  analysis/plots/figures/fig1_system_and_placement.pdf
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Ellipse
import numpy as np

OUT_DIR = PROJECT_ROOT / "analysis" / "plots" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── shared style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
})

BOX_FACE = "#f6f8fa"
BOX_EDGE = "#2c3e50"
ARROW_COLOR = "#2c3e50"
FLEXOR_COLOR = "#c0392b"   # deep red — flexors
EXTENSOR_COLOR = "#2980b9"  # deep blue — extensors
SKIN = "#f4e4d4"
SKIN_EDGE = "#8e8577"

fig, (ax_a, ax_b) = plt.subplots(
    2, 1,
    figsize=(8.4, 5.4),
    gridspec_kw={"height_ratios": [0.55, 1.0], "hspace": 0.02},
)

# ═══════════════════════════════════════════════════════════════
# Panel (a) — system block diagram
# ═══════════════════════════════════════════════════════════════
ax = ax_a
ax.set_xlim(0, 12)
ax.set_ylim(0, 3)
ax.set_aspect("equal")
ax.axis("off")
ax.text(-0.2, 2.85, "(a)", fontsize=11, fontweight="bold", ha="left", va="top")

# Five boxes in a row
stages = [
    ("Electrodes",   "4× MyoWare 2.0\nforearm sites",     "~30 ms",  "#e8f4f8"),
    ("Envelope",     "Teensy 4.0\npeak-to-peak\n@ 20 Hz",  "50 ms",   "#e8f4f8"),
    ("Classifier",   "HGB\n(370 features)",                "~17 ms",  "#fff4e6"),
    ("Stability",    "2-tier\nsmoothing +\nhysteresis",    "100 ms",  "#fff4e6"),
    ("Actuator",     "Servo motor\ntendon-driven\nexoskeleton", "~76 ms", "#f0e8f4"),
]

n = len(stages)
box_w, box_h = 1.70, 1.35
gap = (12 - n * box_w) / (n + 1)
y_center = 1.5

for i, (title, detail, latency, face) in enumerate(stages):
    x = gap + i * (box_w + gap)
    box = FancyBboxPatch(
        (x, y_center - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1, edgecolor=BOX_EDGE, facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + box_w / 2, y_center + 0.42, title,
            ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(x + box_w / 2, y_center - 0.05, detail,
            ha="center", va="center", fontsize=7.5, linespacing=1.2)
    ax.text(x + box_w / 2, y_center - box_h / 2 - 0.22, latency,
            ha="center", va="center", fontsize=7.5, style="italic",
            color="#555555")

    if i < n - 1:
        x_next = gap + (i + 1) * (box_w + gap)
        arrow = FancyArrowPatch(
            (x + box_w + 0.02, y_center),
            (x_next - 0.02, y_center),
            arrowstyle="-|>", mutation_scale=12,
            linewidth=1.2, color=ARROW_COLOR,
        )
        ax.add_patch(arrow)

# Total latency callout
ax.text(6, 0.35, "end-to-end ≈ 275 ms (component-sum)",
        ha="center", va="center", fontsize=8, style="italic",
        color="#555555")

# Feedback loop label above
ax.text(6, 2.85, "Real-time closed-loop pipeline",
        ha="center", va="top", fontsize=8.5, color="#555555")

# ═══════════════════════════════════════════════════════════════
# Panel (b) — forearm electrode placement (anatomical, from GrabMyo)
# ═══════════════════════════════════════════════════════════════
ax = ax_b
ax.axis("off")

import matplotlib.image as mpimg

# Load the anatomical diagram from GrabMyo (Pradhan et al., PhysioNet).
# Panel (b) is annotated on top of it to show our specific 4-channel subselection.
GRABMYO_IMG = PROJECT_ROOT / "grabmyo" / "electrode_placement.png"
img = mpimg.imread(GRABMYO_IMG)
img_h, img_w = img.shape[:2]
ax.imshow(img, aspect="equal")

# Panel label
ax.text(20, 40, "(b)", fontsize=11, fontweight="bold", ha="left", va="top",
        color="#111111")

# Set an extra-wide viewport so we can put annotation text to the right of the image
ax.set_xlim(0, img_w * 1.75)
ax.set_ylim(img_h, 0)   # image coordinates: y goes downward

# Right-side annotation box — describes the 4 channels our paper uses
box_x = img_w + 30
ax.text(box_x, 45, "Our 4 channels", ha="left", va="top", fontsize=10,
        fontweight="bold", color="#111111")
ax.text(box_x, 85, "GrabMyo canonical F1, F5, F10, F14", ha="left", va="top",
        fontsize=8, style="italic", color="#555555")

# The 4 channels described, colour-coded flexor/extensor.
# Each row: coloured dot, then a two-line label (bold tag on top, italic muscle below).
ROW_HEIGHT = 78          # vertical spacing between rows (in image pixels)
FIRST_ROW_Y = 145        # y of the first row's dot centre (leaves room for two-line title)

def annot_row(idx, tag, muscle, full, colour):
    y = FIRST_ROW_Y + idx * ROW_HEIGHT
    # Coloured dot
    ax.text(box_x + 8, y, "●", ha="left", va="center", fontsize=16, color=colour)
    # Bold tag ABOVE the dot's midline (so it doesn't overlap the dot)
    ax.text(box_x + 70, y - 12, tag, ha="left", va="center", fontsize=9,
            fontweight="bold", color=colour)
    # Italic muscle name BELOW the dot's midline
    ax.text(box_x + 70, y + 14, f"{muscle} — {full}", ha="left", va="center",
            fontsize=7.5, style="italic", color="#222222")

annot_row(0, "Ring 1 · flexor",   "FCR", "flexor carpi radialis",           FLEXOR_COLOR)
annot_row(1, "Ring 1 · extensor", "ECR", "extensor carpi radialis",         EXTENSOR_COLOR)
annot_row(2, "Ring 2 · flexor",   "FDS", "flexor digitorum superficialis",  FLEXOR_COLOR)
annot_row(3, "Ring 2 · extensor", "EDC", "extensor digitorum communis",     EXTENSOR_COLOR)

# Interleaving note
notes_y = FIRST_ROW_Y + 4 * ROW_HEIGHT + 30
ax.text(box_x, notes_y,
        "One flexor + one extensor per ring across the two proximal rings\n"
        "of the 2 × 8 forearm setup, giving the interleaved\n"
        "[flexor, extensor, flexor, extensor] geometry the deployed\n"
        "MyoWare rig mirrors on Teensy pins A0, A1, A2, A4.",
        ha="left", va="top", fontsize=7.8, color="#222222", linespacing=1.5)

# Attribution
ax.text(box_x, notes_y + 130,
        "Anatomical diagram adapted from\n"
        "Pradhan et al., GrabMyo dataset (PhysioNet).",
        ha="left", va="top", fontsize=7, style="italic", color="#666666",
        linespacing=1.4)

# Final layout
plt.tight_layout()
out_png = OUT_DIR / "fig1_system_and_placement.png"
out_pdf = OUT_DIR / "fig1_system_and_placement.pdf"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"Wrote {out_png}")
print(f"Wrote {out_pdf}")

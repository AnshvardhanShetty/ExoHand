"""
Fig 1 — System diagram for the paper.

Three-tier block diagram showing the end-to-end ExoHand control loop:
  (i) Acquisition — 4 forearm EMG sensors → MyoWare envelope → Teensy 50 ms P-P window
 (ii) Inference  — 370 engineered features → HGB jointly trained on
                   GrabMyo (1.14 M windows, weight ×1) + per-session cal
                   (432 windows, weight ×100) → 3-class probabilities
(iii) Actuation  — six-layer post-processing → servo command → tendon-driven exo

Annotated with the 50 ms decision cycle and the ~275 ms end-to-end latency at L4.

This is a structural draft; final polish (typography, icons) is easier in a
vector tool. The matplotlib version keeps the figure script-reproducible.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from analysis.plots.style import apply_style, PALETTE, save_pair

OUT = PROJECT_ROOT / "analysis" / "plots" / "figures" / "system_diagram"


# ── Visual constants ────────────────────────────────────────────────────────

BOX_STYLE = dict(boxstyle="round,pad=0.012,rounding_size=0.012",
                 linewidth=1.0)
TIER_COLORS = {
    "acquisition": "#e8f1fb",   # pale blue
    "acquisition_edge": "#2b6cb0",
    "inference":   "#fff5e6",   # pale orange
    "inference_edge": "#c2691a",
    "actuation":   "#e8f5ea",   # pale green
    "actuation_edge": "#2c7a3a",
    "training":    "#f3eaf7",   # pale purple — data sources for HGB
    "training_edge": "#6a3899",
}
TIER_LABEL_FS = 9
BOX_LABEL_FS = 8.5
SUB_LABEL_FS = 7.5
ARROW_KW = dict(arrowstyle="->", linewidth=1.2, color="#444444",
                mutation_scale=12)


def add_box(ax, x, y, w, h, label, color, edge, sublabel=None,
            label_fs=BOX_LABEL_FS):
    """Rounded box with a title and optional sub-label."""
    patch = FancyBboxPatch((x, y), w, h, **BOX_STYLE,
                           facecolor=color, edgecolor=edge)
    ax.add_patch(patch)
    if sublabel:
        ax.text(x + w / 2, y + h * 0.62, label,
                ha="center", va="center", fontsize=label_fs, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.28, sublabel,
                ha="center", va="center", fontsize=SUB_LABEL_FS,
                color="#555555")
    else:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=label_fs, fontweight="bold")


def arrow(ax, x1, y1, x2, y2, label=None, label_pos=0.5, label_offset=(0, 0.012),
          linestyle="-", color="#444444"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="->", linewidth=1.2, color=color,
                        mutation_scale=12, linestyle=linestyle)
    ax.add_patch(a)
    if label is not None:
        lx = x1 + (x2 - x1) * label_pos + label_offset[0]
        ly = y1 + (y2 - y1) * label_pos + label_offset[1]
        ax.text(lx, ly, label, ha="center", va="bottom", fontsize=SUB_LABEL_FS,
                color="#666666", style="italic")


def tier_label(ax, x, y, text, color):
    """Left-margin tier label."""
    ax.text(x, y, text, ha="left", va="center", fontsize=TIER_LABEL_FS,
            fontweight="bold", color=color)


def main():
    apply_style()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    # Tier vertical positions
    Y_ACQ = 0.78
    Y_INF = 0.42
    Y_ACT = 0.13
    BOX_H = 0.12
    # Left margin reserved for tier labels
    X_LEFT = 0.13

    # ── Tier 1: Acquisition ────────────────────────────────────────────────
    tier_label(ax, 0.005, Y_ACQ + BOX_H / 2, "Acquisition",
               TIER_COLORS["acquisition_edge"])
    add_box(ax, X_LEFT, Y_ACQ, 0.18, BOX_H,
            "4 × MyoWare 2.0",
            TIER_COLORS["acquisition"], TIER_COLORS["acquisition_edge"],
            sublabel="forearm flexor + extensor")
    add_box(ax, 0.345, Y_ACQ, 0.18, BOX_H,
            "Teensy 4.0",
            TIER_COLORS["acquisition"], TIER_COLORS["acquisition_edge"],
            sublabel="50 ms P-P window")
    add_box(ax, 0.555, Y_ACQ, 0.19, BOX_H,
            "20 Hz envelope",
            TIER_COLORS["acquisition"], TIER_COLORS["acquisition_edge"],
            sublabel="4-channel P-P stream")
    add_box(ax, 0.775, Y_ACQ, 0.205, BOX_H,
            "200 ms window",
            TIER_COLORS["acquisition"], TIER_COLORS["acquisition_edge"],
            sublabel="50 ms stride · 75 % overlap")

    arrow(ax, X_LEFT + 0.18, Y_ACQ + BOX_H / 2, 0.345, Y_ACQ + BOX_H / 2)
    arrow(ax, 0.525, Y_ACQ + BOX_H / 2, 0.555, Y_ACQ + BOX_H / 2)
    arrow(ax, 0.745, Y_ACQ + BOX_H / 2, 0.775, Y_ACQ + BOX_H / 2)

    # ── Tier 2: Inference ──────────────────────────────────────────────────
    tier_label(ax, 0.005, Y_INF + BOX_H / 2, "Inference",
               TIER_COLORS["inference_edge"])

    # Two data-source boxes feeding HGB (placed ABOVE the HGB block)
    train_y = Y_INF + 0.17
    train_h = 0.075
    add_box(ax, 0.36, train_y, 0.155, train_h,
            "GrabMyo",
            TIER_COLORS["training"], TIER_COLORS["training_edge"],
            sublabel="1.14 M windows · weight ×1",
            label_fs=SUB_LABEL_FS)
    add_box(ax, 0.535, train_y, 0.155, train_h,
            "Per-session cal",
            TIER_COLORS["training"], TIER_COLORS["training_edge"],
            sublabel="≈432 windows · weight ×100",
            label_fs=SUB_LABEL_FS)

    add_box(ax, X_LEFT, Y_INF, 0.20, BOX_H,
            "Feature pipeline",
            TIER_COLORS["inference"], TIER_COLORS["inference_edge"],
            sublabel="370 engineered features")

    add_box(ax, 0.36, Y_INF, 0.33, BOX_H,
            "Histogram GBT",
            TIER_COLORS["inference"], TIER_COLORS["inference_edge"],
            sublabel="jointly trained · balanced class weights")

    # Arrows into HGB from training sources
    arrow(ax, 0.4375, train_y, 0.4375, Y_INF + BOX_H,
          color=TIER_COLORS["training_edge"])
    arrow(ax, 0.6125, train_y, 0.6125, Y_INF + BOX_H,
          color=TIER_COLORS["training_edge"])

    # Probabilities output
    add_box(ax, 0.74, Y_INF, 0.22, BOX_H,
            "P(rest, close, open)",
            TIER_COLORS["inference"], TIER_COLORS["inference_edge"],
            sublabel="3-class probabilities")

    # Tier 2 horizontal arrows
    arrow(ax, X_LEFT + 0.20, Y_INF + BOX_H / 2, 0.36, Y_INF + BOX_H / 2)
    arrow(ax, 0.69, Y_INF + BOX_H / 2, 0.74, Y_INF + BOX_H / 2)

    # ── Tier 3: Actuation ──────────────────────────────────────────────────
    tier_label(ax, 0.005, Y_ACT + BOX_H / 2, "Actuation",
               TIER_COLORS["actuation_edge"])
    # Wider post-processing box; sublabel fits inside
    add_box(ax, X_LEFT, Y_ACT, 0.50, BOX_H,
            "Six-layer post-processing",
            TIER_COLORS["actuation"], TIER_COLORS["actuation_edge"],
            sublabel="EMA · argmax · stability filter (N=3) · cooldown · hysteresis · floor")
    add_box(ax, 0.66, Y_ACT, 0.14, BOX_H,
            "Servo command",
            TIER_COLORS["actuation"], TIER_COLORS["actuation_edge"],
            sublabel="A### · 115 200 baud")
    add_box(ax, 0.82, Y_ACT, 0.16, BOX_H,
            "Exoskeleton",
            TIER_COLORS["actuation"], TIER_COLORS["actuation_edge"],
            sublabel="tendon-driven · £180 BOM")

    arrow(ax, X_LEFT + 0.50, Y_ACT + BOX_H / 2, 0.66, Y_ACT + BOX_H / 2)
    arrow(ax, 0.80, Y_ACT + BOX_H / 2, 0.82, Y_ACT + BOX_H / 2)

    # Cross-tier flow is implied by vertical layout — no dashed cross arrows.

    # ── Timing annotation strip (bottom) ───────────────────────────────────
    timing_y = 0.025
    ax.add_patch(Rectangle((X_LEFT - 0.02, timing_y - 0.015), 0.97 - X_LEFT, 0.045,
                           facecolor="#f8f8f8", edgecolor="#cccccc",
                           linewidth=0.6))
    ax.text(X_LEFT, timing_y + 0.015,
            "Decision cycle  50 ms",
            fontsize=SUB_LABEL_FS, fontweight="bold", va="center")
    ax.text(X_LEFT, timing_y - 0.001,
            "(Teensy P-P sampling window — fixed hardware budget)",
            fontsize=SUB_LABEL_FS - 0.5, color="#555555", style="italic", va="center")
    ax.text(0.58, timing_y + 0.015,
            "Intent-to-motor latency  ~275 ms (L4 profile)",
            fontsize=SUB_LABEL_FS, fontweight="bold", va="center")
    ax.text(0.58, timing_y - 0.001,
            "(envelope + sample + classify + stability wait + serial + servo slew)",
            fontsize=SUB_LABEL_FS - 0.5, color="#555555", style="italic", va="center")

    plt.tight_layout(pad=0.4)
    save_pair(fig, OUT)
    print(f"Wrote {OUT}.{{pdf,png}}")


if __name__ == "__main__":
    main()

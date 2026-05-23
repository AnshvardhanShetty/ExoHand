"""Shared plot style and color palette for paper figures.

Goals: consistent typography across all figures, print-readable at small size,
no decorative junk. Every figure imports from here.
"""

import matplotlib
import matplotlib.pyplot as plt

# ── Style ─────────────────────────────────────────────────────────────

def apply_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.titlepad": 8,
        "axes.labelsize": 9,
        "axes.labelpad": 4,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


# ── Palette ───────────────────────────────────────────────────────────

PALETTE = {
    "zero_shot": "#bdc3c7",     # neutral grey
    "calibrated": "#2980b9",    # primary blue
    "healthy_arm": "#3498db",   # lighter blue
    "impaired_arm": "#c0392b",  # red
    "rest": "#7f8c8d",
    "close": "#2980b9",
    "open": "#e67e22",
    "delta": "#27ae60",
    "ci_band": "#5dade2",
    "muted_text": "#34495e",
}


def axis_clean(ax, ylabel=None, xlabel=None):
    if ylabel is not None: ax.set_ylabel(ylabel)
    if xlabel is not None: ax.set_xlabel(xlabel)
    ax.tick_params(direction="out", which="both")
    return ax


def save_pair(fig, out_base):
    """Save both PDF (for paper) and PNG (for quick view) — same path stem."""
    fig.savefig(f"{out_base}.pdf")
    fig.savefig(f"{out_base}.png", dpi=180)
    return out_base

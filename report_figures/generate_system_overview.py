"""
Generate a professional system overview diagram for EMG-Controlled Exoskeleton System.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(14, 11))
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.set_aspect('equal')
ax.axis('off')

# Colors
colors = {
    'emg': '#1B4F72',        # Dark blue
    'signal': '#2E4053',      # Dark gray-blue
    'feature': '#2E4053',     # Dark gray-blue
    'classify': '#1E8449',    # Green
    'micro': '#1B4F72',       # Dark blue
    'actuation': '#D35400',   # Orange
    'exo': '#5DADE2',         # Light blue
    'number_bg': {
        1: '#1B4F72',
        2: '#2E4053',
        3: '#2E4053',
        4: '#1E8449',
        5: '#1B4F72',
        6: '#D35400',
        7: '#5DADE2'
    }
}

# Box style
box_style = dict(boxstyle="round,pad=0.02,rounding_size=0.02",
                 facecolor='white', edgecolor='#E5E8E8', linewidth=1.5)

def draw_component_box(ax, x, y, width, height, number, title, items, color):
    """Draw a component box with number badge, title, and bullet items."""
    # Main box
    box = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.02,rounding_size=0.15",
                          facecolor='white', edgecolor='#D5D8DC', linewidth=1.5,
                          transform=ax.transData, zorder=2)
    ax.add_patch(box)

    # Number circle
    circle = Circle((x + 0.35, y + height - 0.35), 0.25,
                    facecolor=color, edgecolor='white', linewidth=2, zorder=3)
    ax.add_patch(circle)
    ax.text(x + 0.35, y + height - 0.35, str(number),
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='white', zorder=4)

    # Title bar
    title_bar = FancyBboxPatch((x + 0.6, y + height - 0.6), width - 0.8, 0.5,
                                boxstyle="round,pad=0.01,rounding_size=0.1",
                                facecolor=color, edgecolor='none', zorder=3)
    ax.add_patch(title_bar)
    ax.text(x + 0.6 + (width - 0.8)/2, y + height - 0.35, title,
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='white', zorder=4)

    # Bullet items
    for i, item in enumerate(items):
        bullet_y = y + height - 1.1 - i * 0.4
        ax.plot(x + 0.35, bullet_y, 'o', markersize=5, color=color, zorder=3)
        ax.text(x + 0.55, bullet_y, item, ha='left', va='center',
                fontsize=9, color='#2C3E50', zorder=3)

# Define component positions (arranged in a cleaner flow)
# Top row: EMG Acquisition -> Signal Processing
# Middle-right: Feature Extraction -> Classification
# Bottom: Microcontroller -> Actuation -> Exoskeleton

components = {
    'emg': {
        'pos': (5.5, 7.5), 'size': (3, 2),
        'num': 1, 'title': 'EMG ACQUISITION', 'color': colors['emg'],
        'items': ['3-Channel Surface EMG', 'Forearm Placement', '1000 Hz Sampling']
    },
    'signal': {
        'pos': (10, 7.5), 'size': (3, 2),
        'num': 2, 'title': 'SIGNAL PROCESSING', 'color': colors['signal'],
        'items': ['20-450 Hz Bandpass', 'Full-Wave Rectification', 'Envelope Extraction']
    },
    'feature': {
        'pos': (10, 4.5), 'size': (3, 2),
        'num': 3, 'title': 'FEATURE EXTRACTION', 'color': colors['feature'],
        'items': ['RMS & MAV', 'Waveform Length', 'Zero Crossings']
    },
    'classify': {
        'pos': (7, 1.5), 'size': (3, 2),
        'num': 4, 'title': 'CLASSIFICATION', 'color': colors['classify'],
        'items': ['Gradient Boosting', 'Real-Time Inference', '95.9% Accuracy']
    },
    'micro': {
        'pos': (3.5, 1.5), 'size': (3, 2),
        'num': 5, 'title': 'MICROCONTROLLER', 'color': colors['micro'],
        'items': ['ESP32 MCU', 'WiFi/BLE Connectivity', 'PWM Generation']
    },
    'actuation': {
        'pos': (0.5, 4.5), 'size': (3, 2),
        'num': 6, 'title': 'ACTUATION', 'color': colors['actuation'],
        'items': ['5x Servo Motors', 'Tendon-Driven Design', 'Position Control']
    },
    'exo': {
        'pos': (0.5, 7.5), 'size': (3, 2),
        'num': 7, 'title': 'EXOSKELETON', 'color': colors['exo'],
        'items': ['3D Printed Frame', 'Modular Fingers', 'Sensory Feedback']
    }
}

# Draw all components
for name, comp in components.items():
    draw_component_box(ax, comp['pos'][0], comp['pos'][1],
                       comp['size'][0], comp['size'][1],
                       comp['num'], comp['title'], comp['items'], comp['color'])

# Arrow color
arrow_color = '#2C3E50'

# Draw clean arrows connecting components in sequence (clockwise flow)
# Arrow settings
arrow_props = dict(arrowstyle='->', color=arrow_color, lw=2,
                   shrinkA=0, shrinkB=3)

# 1 -> 2: EMG Acquisition -> Signal Processing (horizontal right)
ax.annotate('', xy=(10, 8.5), xytext=(8.5, 8.5), arrowprops=arrow_props)

# 2 -> 3: Signal Processing -> Feature Extraction (vertical down)
ax.annotate('', xy=(11.5, 6.5), xytext=(11.5, 7.5), arrowprops=arrow_props)

# 3 -> 4: Feature Extraction -> Classification (down then left)
ax.annotate('', xy=(10, 3.5), xytext=(11.5, 4.5),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2,
                           connectionstyle='angle,angleA=0,angleB=90'))

# 4 -> 5: Classification -> Microcontroller (horizontal left)
ax.annotate('', xy=(6.5, 2.5), xytext=(7, 2.5), arrowprops=arrow_props)

# 5 -> 6: Microcontroller -> Actuation (left then up)
ax.annotate('', xy=(2, 4.5), xytext=(3.5, 3.5),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2,
                           connectionstyle='angle,angleA=0,angleB=-90'))

# 6 -> 7: Actuation -> Exoskeleton (vertical up)
ax.annotate('', xy=(2, 7.5), xytext=(2, 6.5), arrowprops=arrow_props)

# 7 -> 1: Exoskeleton -> EMG (Feedback loop - dashed curved, above boxes)
ax.annotate('', xy=(5.5, 9.6), xytext=(3.5, 9.6),
            arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.5,
                           connectionstyle='arc3,rad=-0.1', linestyle='--'))
ax.text(4.5, 9.85, 'Sensory Feedback', ha='center', va='bottom',
        fontsize=8, fontstyle='italic', color='#7F8C8D')

# Central "CLOSED LOOP" circle
center_circle = Circle((6.5, 5.5), 1.0, facecolor='#F8F9F9',
                        edgecolor='#BDC3C7', linewidth=2, zorder=1)
ax.add_patch(center_circle)
ax.text(6.5, 5.7, 'CLOSED', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#2C3E50')
ax.text(6.5, 5.3, 'LOOP', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#2C3E50')

# Title
ax.text(7, 10.5, 'EMG-CONTROLLED EXOSKELETON SYSTEM',
        ha='center', va='center', fontsize=16, fontweight='bold', color='#1B4F72')

# Bottom stats bar
stats_y = 0.3
stats = [
    ('95.9%', 'ACCURACY'),
    ('98.8%', 'BINARY ACC.'),
    ('<50ms', 'LATENCY'),
    ('3', 'CHANNELS')
]

bar_width = 13
bar_x = 0.5
stat_width = bar_width / len(stats)

# Stats background
stats_bg = FancyBboxPatch((bar_x, stats_y - 0.1), bar_width, 0.9,
                           boxstyle="round,pad=0.01,rounding_size=0.05",
                           facecolor='#F8F9F9', edgecolor='#E5E8E8',
                           linewidth=1, zorder=1)
ax.add_patch(stats_bg)

for i, (value, label) in enumerate(stats):
    x = bar_x + (i + 0.5) * stat_width
    ax.text(x, stats_y + 0.5, value, ha='center', va='center',
            fontsize=14, fontweight='bold', color='#2C3E50')
    ax.text(x, stats_y + 0.15, label, ha='center', va='center',
            fontsize=8, color='#7F8C8D')
    # Vertical separator (except for last)
    if i < len(stats) - 1:
        ax.plot([bar_x + (i + 1) * stat_width, bar_x + (i + 1) * stat_width],
                [stats_y, stats_y + 0.8], color='#E5E8E8', linewidth=1, zorder=2)

plt.tight_layout()
plt.savefig('/Users/anshshetty/Library/Mobile Documents/com~apple~CloudDocs/ExoHand/report_figures/system_overview.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('/Users/anshshetty/Library/Mobile Documents/com~apple~CloudDocs/ExoHand/report_figures/system_overview.pdf',
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("System overview diagram saved!")

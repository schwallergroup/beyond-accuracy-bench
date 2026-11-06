# Unified styling for paper plots

"""
This module defines a unified styling configuration for plots across the project
to ensure consistency in publications.

- Linestyles distinguish the modeling method (e.g., NLL Agent, Raw Agent, Random).
- Markers distinguish the evaluation metric (e.g., Accuracy, Entropy).
"""

# 1. Linestyles represent the Method/Mode
#    - Agent (NLL mode): Solid line
#    - Agent (Raw data mode): Dashed line
#    - Random: Dotted line
METHOD_LINESTYLES = {
    'Agent (NLL mode)': '-',
    'Agent (Raw mode)': '--',
    'BO': '-.',
    'BO baseline': '-.',
    'Maximin LHS': '-.',
    'Maximin LHS baseline': '-.',
    'Random': ':',
    'Random baseline': ':',
}

# Convert string-based linestyles to matplotlib's dash tuple format.
# This is crucial for the `dashes` argument in seaborn and prevents TypeErrors.
# The tuples represent (on_length, off_length).
METHOD_DASHES = {
    key: (None, None) if style == '-' else    # Solid
         (6, 2) if style == '--' else         # Dashed
         (5, 1.5, 1.5, 1.5) if style == '-.' else  # Dash-dot
         (1, 1.65) if style == ':' else       # Dotted (approximates seaborn's 'dotted')
         style                                # Fallback
    for key, style in METHOD_LINESTYLES.items()
}


# 2. Markers represent the Metric
#    - Accuracy from NLL: Circle
#    - Top-1 Accuracy from Candidates: Triangle Up
#    - Top-5 Accuracy from Candidates: Triangle Down
#    - Entropy: Diamond
METRIC_MARKERS = {
    'Accuracy from NLL': 'o',
    'Top-1 Accuracy from Candidates': '^',
    'Top-5 Accuracy from Candidates': 'v',
    'Entropy': 'D',
    'Occupancy Entropy': 's',
    'Boundary Mass': 'X',
}


# 3. Unified paper palette
PAPER_COLORS = [
    '#8B0AA5',  # deep purple
    '#B93389',  # magenta
    '#DB5B68',  # pink-red
    '#F48849',  # orange
    '#FEBC2B',  # yellow
]


def apply_paper_rcparams():
    """
    Apply the paper plot style shared across plotting scripts.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.0,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def set_uniform_yticks(ax, ylim, step=0.2):
    """
    Set a consistent y-tick interval for a plot panel.
    """
    import numpy as np

    if ylim is None:
        return
    ax.set_ylim(*ylim)
    start, end = ylim
    ax.set_yticks(np.arange(start, end + 1e-9, step))

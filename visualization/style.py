"""Lancet Digital Health 出版级图形样式"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Lancet 调色板
LANCET_COLORS = {
    "red": "#E64B35",
    "blue": "#4DBBD5",
    "green": "#00A087",
    "navy": "#3C5488",
    "purple": "#8491B4",
    "salmon": "#F39B7F",
    "brown": "#B09C85",
    "teal": "#91D1C2",
    "yellow": "#E2D200",
    "black": "#333333",
}


def setup_lancet_style():
    """设置 Lancet 级图形全局参数"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def get_figure_size(style: str = "single") -> tuple:
    """Lancet 栏宽 (mm → inches)"""
    if style == "single":
        return (180 / 25.4, 120 / 25.4)  # ~7.1 x 4.7 inches
    elif style == "double":
        return (360 / 25.4, 200 / 25.4)  # ~14.2 x 7.9 inches
    elif style == "half":
        return (90 / 25.4, 90 / 25.4)  # ~3.5 x 3.5 inches
    return (180 / 25.4, 120 / 25.4)

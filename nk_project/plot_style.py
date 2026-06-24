from __future__ import annotations

import matplotlib.pyplot as plt


TITLE_SIZE = 16
SUPTITLE_SIZE = 20
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_FONT_SIZE = 12
LEGEND_TITLE_SIZE = 13
SMALL_TICK_LABEL_SIZE = 10


def set_presentation_style() -> None:
    """Set readable defaults for slide-facing figures."""
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": "bold",
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.labelweight": "bold",
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "legend.title_fontsize": LEGEND_TITLE_SIZE,
            "figure.titlesize": SUPTITLE_SIZE,
            "figure.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_legend(legend, *, fontsize: int = LEGEND_FONT_SIZE, title_fontsize: int = LEGEND_TITLE_SIZE) -> None:
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontsize(fontsize)
        text.set_fontweight("bold")
    title = legend.get_title()
    if title is not None:
        title.set_fontsize(title_fontsize)
        title.set_fontweight("bold")


def style_all_legends(fig, *, fontsize: int = LEGEND_FONT_SIZE, title_fontsize: int = LEGEND_TITLE_SIZE) -> None:
    for legend in fig.legends:
        style_legend(legend, fontsize=fontsize, title_fontsize=title_fontsize)
    for ax in fig.axes:
        style_legend(ax.get_legend(), fontsize=fontsize, title_fontsize=title_fontsize)


def style_axis(
    ax,
    *,
    title_size: int = TITLE_SIZE,
    label_size: int = AXIS_LABEL_SIZE,
    tick_size: int = TICK_LABEL_SIZE,
    title_weight: str = "bold",
    label_weight: str = "bold",
) -> None:
    ax.title.set_fontsize(title_size)
    ax.title.set_fontweight(title_weight)
    ax.xaxis.label.set_fontsize(label_size)
    ax.xaxis.label.set_fontweight(label_weight)
    ax.yaxis.label.set_fontsize(label_size)
    ax.yaxis.label.set_fontweight(label_weight)
    ax.tick_params(axis="both", labelsize=tick_size)


def style_figure(
    fig,
    *,
    title_size: int = TITLE_SIZE,
    label_size: int = AXIS_LABEL_SIZE,
    tick_size: int = TICK_LABEL_SIZE,
    legend_size: int = LEGEND_FONT_SIZE,
) -> None:
    for ax in fig.axes:
        style_axis(ax, title_size=title_size, label_size=label_size, tick_size=tick_size)
    style_all_legends(fig, fontsize=legend_size)

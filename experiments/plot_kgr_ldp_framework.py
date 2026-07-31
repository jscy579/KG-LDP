"""Generate the mechanism-independent KGR-LDP framework figure.

The diagram is conceptual and contains no experimental values.  It separates
the offline, public graph-to-partition configuration from local record
processing and server-side population estimation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "formal_v3"
STEM = OUT_DIR / "figure1_kgr_ldp_framework"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)


COLORS = {
    "ink": "#243238",
    "muted": "#5B676C",
    "public_edge": "#A75B13",
    "public_fill": "#FFF5E8",
    "public_box": "#FFFDFC",
    "local_edge": "#0D5A63",
    "local_fill": "#F3FAFA",
    "record": "#F1F3F4",
    "partition": "#E8F3F1",
    "aggregate": "#E7F0F8",
    "representation": "#EAE4F3",
    "privacy": "#F7E7E5",
    "server": "#EDF2F0",
    "white": "#FFFFFF",
}


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 1.0,
    radius: float = 0.014,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linewidth: float = 1.25,
    mutation_scale: float = 9,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            color=color,
            linewidth=linewidth,
            mutation_scale=mutation_scale,
            connectionstyle=connectionstyle,
            shrinkA=0,
            shrinkB=0,
        )
    )


def add_centered_text(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 7,
    color: str | None = None,
    weight: str = "normal",
    linespacing: float = 1.15,
) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=color or COLORS["ink"],
        fontweight=weight,
        linespacing=linespacing,
    )


def build_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Offline public configuration layer.
    rounded_box(
        ax,
        0.055,
        0.705,
        0.89,
        0.235,
        facecolor=COLORS["public_fill"],
        edgecolor=COLORS["public_edge"],
        linewidth=1.15,
        radius=0.018,
    )
    ax.text(
        0.075,
        0.915,
        "Offline public configuration",
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=COLORS["public_edge"],
    )

    top_boxes = [
        (0.085, 0.760, 0.215, 0.105, "Public knowledge graph", "typed concepts and relations\nschema information only"),
        (
            0.385,
            0.760,
            0.230,
            0.105,
            "Graph-to-partition rule",
            r"$\phi_{\mathcal{C}}:\mathcal{A}\rightarrow\mathcal{S}_{\mathcal{C}}$"
            "\nattribute-to-anchor assignment",
        ),
        (
            0.700,
            0.760,
            0.215,
            0.105,
            "Frozen semantic partition",
            r"$\mathcal{P}_{\mathcal{C}}=\{A_1,\ldots,A_{m_{\mathcal{C}}}\}$"
            "\n"
            r"$m_{\mathcal{C}}=|\operatorname{im}\phi_{\mathcal{C}}|<d$",
        ),
    ]
    for x, y, w, h, title, subtitle in top_boxes:
        rounded_box(
            ax,
            x,
            y,
            w,
            h,
            facecolor=COLORS["public_box"],
            edgecolor=COLORS["public_edge"],
            linewidth=0.9,
            radius=0.012,
        )
        add_centered_text(ax, x + w / 2, y + 0.069, title, size=7.1, weight="bold")
        add_centered_text(ax, x + w / 2, y + 0.032, subtitle, size=6.3, color=COLORS["muted"])

    arrow(ax, (0.305, 0.812), (0.378, 0.812), color=COLORS["public_edge"])
    arrow(ax, (0.620, 0.812), (0.693, 0.812), color=COLORS["public_edge"])
    ax.text(
        0.500,
        0.724,
        "No protected patient value is used to construct the partition",
        ha="center",
        va="center",
        fontsize=6.5,
        color=COLORS["public_edge"],
        fontweight="bold",
    )

    # Local-to-server pipeline.
    rounded_box(
        ax,
        0.020,
        0.075,
        0.960,
        0.560,
        facecolor=COLORS["white"],
        edgecolor=COLORS["local_edge"],
        linewidth=1.15,
        radius=0.018,
    )
    ax.axvline(
        0.855,
        ymin=0.095,
        ymax=0.620,
        color=COLORS["muted"],
        linewidth=0.9,
        linestyle=(0, (4, 3)),
    )
    ax.text(
        0.045,
        0.603,
        "Local device",
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=COLORS["local_edge"],
    )
    ax.text(
        0.875,
        0.603,
        "Server",
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=COLORS["local_edge"],
    )

    pipeline = [
        (
            0.035,
            0.270,
            0.130,
            0.185,
            COLORS["record"],
            "High-dimensional\nattribute pattern",
            r"$x_i\in\mathcal{X}^d$",
        ),
        (
            0.195,
            0.270,
            0.150,
            0.185,
            COLORS["partition"],
            "Semantic attribute\npartition",
            r"$x_i|_{A_1},\ldots,x_i|_{A_{m_{\mathcal{C}}}}$",
        ),
        (
            0.375,
            0.270,
            0.150,
            0.185,
            COLORS["aggregate"],
            "Within-group\nstate aggregation",
            r"$\psi:\;0/1/2$",
        ),
        (
            0.555,
            0.270,
            0.135,
            0.185,
            COLORS["representation"],
            "Low-dimensional\nrelease representation",
            r"$z_i\in\{0,1,2\}^{m_{\mathcal{C}}}$",
        ),
        (
            0.720,
            0.270,
            0.120,
            0.185,
            COLORS["privacy"],
            "Replaceable\nuser-level LDP",
            r"$Y_i\sim Q_{\varepsilon}(\cdot|z_i)$",
        ),
        (
            0.875,
            0.270,
            0.090,
            0.185,
            COLORS["server"],
            "Population\ndistribution\nestimation",
            r"$\widehat{\Theta}$",
        ),
    ]

    for x, y, w, h, face, title, symbol in pipeline:
        rounded_box(
            ax,
            x,
            y,
            w,
            h,
            facecolor=face,
            edgecolor=COLORS["local_edge"],
            linewidth=0.9,
            radius=0.012,
        )
        add_centered_text(ax, x + w / 2, y + 0.121, title, size=6.8, weight="bold")
        add_centered_text(ax, x + w / 2, y + 0.050, symbol, size=7.0, color=COLORS["muted"])

    arrow(ax, (0.167, 0.363), (0.187, 0.363), color=COLORS["local_edge"])
    arrow(ax, (0.347, 0.363), (0.367, 0.363), color=COLORS["local_edge"])
    arrow(ax, (0.527, 0.363), (0.547, 0.363), color=COLORS["local_edge"])
    arrow(ax, (0.692, 0.363), (0.712, 0.363), color=COLORS["local_edge"])
    arrow(ax, (0.842, 0.363), (0.867, 0.363), color=COLORS["local_edge"])

    # Link the frozen public partition to its local application.
    arrow(
        ax,
        (0.807, 0.752),
        (0.270, 0.465),
        color=COLORS["public_edge"],
        linewidth=1.0,
        mutation_scale=8,
        connectionstyle="arc3,rad=0.06",
    )
    ax.text(
        0.560,
        0.560,
        r"frozen $\mathcal{P}_{\mathcal{C}}$",
        ha="center",
        va="center",
        fontsize=6.4,
        color=COLORS["public_edge"],
        fontweight="bold",
    )

    ax.text(
        0.500,
        0.105,
        r"$d$ attributes"
        r"$\;\rightarrow\;$"
        r"$m_{\mathcal{C}}<d$ semantic release dimensions"
        r"$\;\rightarrow\;$"
        r"$\varepsilon$-LDP reports"
        r"$\;\rightarrow\;$"
        "population distributions",
        ha="center",
        va="center",
        fontsize=6.8,
        color=COLORS["ink"],
        fontweight="bold",
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

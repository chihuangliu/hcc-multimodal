"""Venn diagram helpers for feature-set overlap analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib_venn import venn2, venn3


def draw_venn3(
    ax: Axes,
    sets: list[set],
    title: str,
    fold_labels: list[str] | None = None,
) -> None:
    """Draw a 3-set Venn on *ax* and print intersection summary to stdout.

    Parameters
    ----------
    ax:
        Matplotlib axes to draw on.
    sets:
        Three sets to compare, in order [set1, set2, set3].
    title:
        Axes title string.
    fold_labels:
        Override the default ``"Fold i (n=…)"`` labels.  Must have length 3
        when provided.
    """
    sizes = [len(s) for s in sets]
    if fold_labels is None:
        fold_labels = [f"Fold {i + 1}\n(n={sizes[i]})" for i in range(3)]
    if sum(sizes) == 0:
        ax.text(0.5, 0.5, "no features selected", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        return
    venn3(sets, set_labels=tuple(fold_labels), ax=ax)
    ax.set_title(title, fontsize=10)
    f12 = sets[0] & sets[1] - sets[2]
    f13 = sets[0] & sets[2] - sets[1]
    f23 = sets[1] & sets[2] - sets[0]
    f123 = sets[0] & sets[1] & sets[2]
    print(f"  {title}")
    print(f"    F1∩F2∩F3 ({len(f123)}): {sorted(f123) or '-'}")
    print(f"    F1∩F2 only ({len(f12)}): {sorted(f12) or '-'}")
    print(f"    F1∩F3 only ({len(f13)}): {sorted(f13) or '-'}")
    print(f"    F2∩F3 only ({len(f23)}): {sorted(f23) or '-'}")


def draw_venn2(
    ax: Axes,
    set_a: set,
    set_b: set,
    label_a: str,
    label_b: str,
    title: str,
) -> None:
    """Draw a 2-set Venn on *ax* and print overlap to stdout.

    Parameters
    ----------
    ax:
        Matplotlib axes to draw on.
    set_a, set_b:
        The two sets to compare.
    label_a, label_b:
        Human-readable labels for each set.
    title:
        Axes title string.
    """
    overlap = set_a & set_b
    print(f"  {title}: |A|={len(set_a)}, |B|={len(set_b)}, |A∩B|={len(overlap)}: {sorted(overlap) or '-'}")
    venn2(
        [set_a, set_b],
        set_labels=(f"{label_a}\n(n={len(set_a)})", f"{label_b}\n(n={len(set_b)})"),
        ax=ax,
    )
    ax.set_title(title, fontsize=10)


def save_venn_figure(fig: plt.Figure, path, dpi: int = 150) -> None:
    """Save *fig* to *path* and close it."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

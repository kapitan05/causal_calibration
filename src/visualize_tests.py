from typing import List

import matplotlib.pyplot as plt


def visualize_tests(
    step_frac: float,
    del_probs: List[float],
    ins_probs: List[float],
    del_auc: float,
    ins_auc: float,
) -> None:
    x_axis = [i * step_frac for i in range(len(del_probs))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    kolor_linii = "steelblue"
    kolor_wypelnienia = "lightsteelblue"

    # --- deletion
    ax1 = axes[0]
    ax1.plot(x_axis, del_probs, color=kolor_linii, linewidth=1.5)
    ax1.fill_between(x_axis, del_probs, color=kolor_wypelnienia, alpha=0.7)
    ax1.set_title("Deletion", fontsize=20)
    ax1.text(
        0.5,
        0.55,
        f"AUC={del_auc:.3f}",
        transform=ax1.transAxes,
        fontsize=22,
        ha="center",
        va="center",
    )

    ax1.set_xlim(0, max(x_axis))
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # --- insertion
    ax2 = axes[1]
    ax2.plot(x_axis, ins_probs, color=kolor_linii, linewidth=1.5)
    ax2.fill_between(x_axis, ins_probs, color=kolor_wypelnienia, alpha=0.7)
    ax2.set_title("Insertion", fontsize=20)
    ax2.text(
        0.5,
        0.55,
        f"AUC={ins_auc:.3f}",
        transform=ax2.transAxes,
        fontsize=22,
        ha="center",
        va="center",
    )

    ax2.set_xlim(0, max(x_axis))
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()
    plt.show()

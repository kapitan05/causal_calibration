from typing import List

import matplotlib.pyplot as plt


def visualize_tests(
    del_x_axis: List[float],
    ins_x_axis: List[float],
    del_probs: List[float],
    ins_probs: List[float],
    del_auc: float,
    ins_auc: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    kolor_linii = "steelblue"
    kolor_wypelnienia = "lightsteelblue"

    # --- deletion
    ax1 = axes[0]
    ax1.plot(del_x_axis, del_probs, color=kolor_linii, linewidth=1.5)
    ax1.fill_between(del_x_axis, del_probs, color=kolor_wypelnienia, alpha=0.7)
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

    ax1.set_xlim(0, max(del_x_axis))
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # --- insertion
    ax2 = axes[1]
    ax2.plot(ins_x_axis, ins_probs, color=kolor_linii, linewidth=1.5)
    ax2.fill_between(ins_x_axis, ins_probs, color=kolor_wypelnienia, alpha=0.7)
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

    ax2.set_xlim(0, max(ins_x_axis))
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()
    plt.show()


def plot_ece_curves(
    x_axis: list[float],
    ece_base: list[float],
    ece_calibrated: list[float],
    title: str = "Calibration Curves (ECE) for Deletion Test",
) -> None:
    """Plot comparison of ECE for base and calibrated models."""
    plt.figure(figsize=(8, 5))
    plt.plot(
        x_axis,
        ece_base,
        color="darkred",
        linestyle="--",
        marker="o",
        label="Base Model (Uncalibrated)",
    )
    plt.plot(
        x_axis,
        ece_calibrated,
        color="darkgreen",
        linestyle="-",
        marker="s",
        label="ReCalX Model (Calibrated)",
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Percentage of Damaged Images", fontsize=12)
    plt.ylabel("Expected Calibration Error (ECE)", fontsize=12)
    plt.ylim(0.0, max(max(ece_base), max(ece_calibrated)) * 1.2)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.show()

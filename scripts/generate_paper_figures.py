#!/usr/bin/env python3
"""Generate publication-style quantitative figures from tracked evidence.

Conceptual schematics in ``figures/research_*.png`` are explanatory assets.
This script is deliberately limited to quantitative plots so generated artwork
can never invent or alter experimental measurements.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRS = (ROOT / "figures", ROOT / "docs" / "figures")

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green": "#8BCF8B",
    "red": "#B64342",
    "red_light": "#F6CFCB",
    "neutral": "#CFCECE",
    "gray": "#767676",
}

LOG_PATTERN = re.compile(r"\[Iter\s+(\d+).*?vs_qt=([0-9.]+)")


def configure_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 13,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 1.8,
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })
    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, stem: str) -> None:
    for output_dir in OUTPUT_DIRS:
        fig.savefig(output_dir / f"{stem}.png", dpi=300,
                    bbox_inches="tight", facecolor="white")
        fig.savefig(output_dir / f"{stem}.pdf",
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_q_learning_curve() -> tuple[np.ndarray, np.ndarray]:
    data = json.loads((ROOT / "results" / "training_curve_real.json").read_text())
    return np.asarray(data["episodes"]), np.asarray(data["win_rates"])


def load_ppo_logs() -> list[tuple[str, np.ndarray]]:
    specs = [
        ("V3: same-model self-play", "training_selfplay_v3_nofrozen_20260506_135726.log"),
        ("V4: wider clipping", "training_selfplay_v4_aggressive_20260506_162016.log"),
        ("V5: 2 PPO epochs", "training_selfplay_v5_onpolicy_20260506_174102.log"),
        ("V6: 1 PPO epoch", "training_selfplay_v6_pure_onpolicy_20260506_175139.log"),
    ]
    curves = []
    for label, filename in specs:
        pairs = [(int(i), float(v)) for i, v in LOG_PATTERN.findall(
            (ROOT / "logs" / filename).read_text()
        )]
        if pairs:
            curves.append((label, np.asarray(pairs)))
    return curves


def make_training_figure() -> None:
    """Plot Q-learning and PPO evidence without mixing their benchmarks."""
    fig, (ax_q, ax_ppo) = plt.subplots(1, 2, figsize=(14, 4.8))

    episodes, win_rates = load_q_learning_curve()
    x = episodes / 1_000_000
    ax_q.scatter(x, win_rates, s=36, color=PALETTE["blue_secondary"],
                 alpha=0.55, edgecolor="none", label="evaluation")
    window = 5
    smooth = np.convolve(win_rates, np.ones(window) / window, mode="valid")
    ax_q.plot(x[window - 1:], smooth, color=PALETTE["blue_main"],
              linewidth=2.8, label="5-point moving average")
    ax_q.set_title("a   Tabular Q-learning", loc="left", fontweight="bold")
    ax_q.set_xlabel("Training games (millions)")
    ax_q.set_ylabel("Win rate vs random (%)")
    ax_q.set_ylim(45, 100)
    ax_q.legend(loc="lower right")

    colors = [PALETTE["gray"], PALETTE["green"],
              PALETTE["red"], PALETTE["blue_main"]]
    for (label, curve), color in zip(load_ppo_logs(), colors):
        ax_ppo.plot(curve[:, 0], 100 * curve[:, 1], marker="o",
                    markersize=4.5, linewidth=2.2, color=color, label=label)
    ax_ppo.set_title("b   PPO iteration study", loc="left", fontweight="bold")
    ax_ppo.set_xlabel("Global training iteration")
    ax_ppo.set_ylabel("Win rate vs fixed Q-table (%)")
    ax_ppo.set_ylim(0, 100)
    ax_ppo.legend(loc="lower right", fontsize=9)

    for ax in (ax_q, ax_ppo):
        ax.tick_params(direction="out", length=5)

    fig.text(
        0.5, -0.03,
        "The panels use different opponents and are not directly comparable. "
        "Each PPO evaluation contains 800 games with alternating roles.",
        ha="center", fontsize=10, color=PALETTE["gray"],
    )
    fig.tight_layout(pad=2.0)
    save(fig, "research_training_results")


def make_matchup_figure() -> None:
    """Plot the role-conditioned matchup with an explicit scope warning."""
    fig, (ax_count, ax_rate) = plt.subplots(1, 2, figsize=(14, 4.8))

    counts = np.asarray([1394, 455, 151])
    labels = ["PPO wins\nas P0 and P1", "PPO wins\nonly as P0", "PPO wins\nonly as P1"]
    colors = [PALETTE["green"], PALETTE["red_light"], PALETTE["blue_secondary"]]
    bars = ax_count.bar(np.arange(3), counts, width=0.66, color=colors,
                        edgecolor="black", linewidth=1.3)
    ax_count.set_xticks(np.arange(3), labels)
    ax_count.set_ylabel("Starting positions")
    ax_count.set_ylim(0, 1575)
    ax_count.set_title("a   Deterministic PPO vs Q-table", loc="left", fontweight="bold")
    for bar, count in zip(bars, counts):
        ax_count.text(bar.get_x() + bar.get_width() / 2, count + 35,
                      f"{count}\n({100 * count / counts.sum():.1f}%)",
                      ha="center", va="bottom", fontsize=11)

    rates = np.asarray([92.4, 77.2, 84.8])
    role_labels = ["PPO as P0", "PPO as P1", "Balanced mean"]
    bars = ax_rate.barh(np.arange(3), rates, height=0.52,
                        color=[PALETTE["red_light"], PALETTE["blue_secondary"],
                               PALETTE["blue_main"]],
                        edgecolor="black", linewidth=1.3)
    ax_rate.set_yticks(np.arange(3), role_labels)
    ax_rate.invert_yaxis()
    ax_rate.set_xlim(0, 100)
    ax_rate.set_xlabel("Win rate vs fixed Q-table (%)")
    ax_rate.set_title("b   Role-conditioned performance", loc="left", fontweight="bold")
    for bar, rate in zip(bars, rates):
        ax_rate.text(rate + 1.2, bar.get_y() + bar.get_height() / 2,
                     f"{rate:.1f}%", va="center", fontsize=11)

    fig.text(
        0.5, -0.03,
        "n = 2,000 fixed starting idioms, two games per start. "
        "Head-to-head outcomes do not certify minimax values.",
        ha="center", fontsize=10, color=PALETTE["red"],
    )
    fig.tight_layout(pad=2.0)
    save(fig, "research_matchup_results")


def main() -> None:
    configure_style()
    make_training_figure()
    make_matchup_figure()
    print("Generated quantitative figures in figures/ and docs/figures/.")


if __name__ == "__main__":
    main()

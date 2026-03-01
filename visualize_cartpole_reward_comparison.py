"""
Plot CartPole reward curves from multiple PKL recordings on one figure.
Style matches plot_rewards.py used by run_viz_lql_gridworld.py.
"""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from utils import smooth


SOURCES = {
    "QL": Path("recordings/cartpole_ql/rewards_raw.pkl"),
    "QL2": Path("recordings/cartpole_ql2/rewards_raw.pkl"),
    "LQL": Path("recordings/cartpole_lql/rewards_raw.pkl"),
    "LQL2": Path("recordings/cartpole_lql2/rewards_raw.pkl"),
}

OUTPUT_DIR = Path("recordings/cartpole_compare")
OUTPUT_PATH = OUTPUT_DIR / "rewards_compare.png"
SMOOTH_WINDOW = 50


def load_rewards(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        rewards = pickle.load(f)
    return np.asarray(rewards, dtype=float)


def main() -> None:
    for name, path in SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing rewards file for {name}: {path}")

    # Match style from plot_rewards.py
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    colors = {
        "LQL": "#E74C3C",   # red
        "LQL2": "#E67E22",  # orange
        "QL": "#3498DB",    # blue
        "QL2": "#2ECC71",   # green
    }

    fig, ax = plt.subplots(figsize=(5, 3))
    series: dict[str, tuple[np.ndarray, np.ndarray, range]] = {}

    for name, path in SOURCES.items():
        rewards = load_rewards(path)
        rewards_smooth = smooth(rewards, window=SMOOTH_WINDOW)
        x_smooth = range(len(rewards) - len(rewards_smooth), len(rewards))
        series[name] = (rewards, rewards_smooth, x_smooth)

    # Draw all raw curves first so they stay below the smoothed curves.
    for name in SOURCES:
        rewards, _, _ = series[name]
        ax.plot(
            rewards,
            color=colors[name],
            alpha=0.18,
            linewidth=0.8,
            zorder=1,
        )

    # Draw all smoothed curves last so they always appear on top.
    for name in SOURCES:
        _, rewards_smooth, x_smooth = series[name]
        ax.plot(
            x_smooth,
            rewards_smooth,
            color=colors[name],
            linewidth=2,
            label=f"{name} (MA{SMOOTH_WINDOW})",
            zorder=3,
        )

    ax.set_xlabel("Episode", fontweight="normal")
    ax.set_ylabel("Total Reward", fontweight="normal")
    # ax.set_title("CartPole Reward Comparison", fontweight="normal", pad=8)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.3)
    ax.legend(frameon=False, loc="lower right")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

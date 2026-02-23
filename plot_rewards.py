from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import numpy as np
from utils import smooth


def plot_rewards(rewards_path, smooth_window=50):
    rewards = pickle.load(open(rewards_path, "rb"))
    rewards_smooth = smooth(np.array(rewards), window=smooth_window)

    # Nature/Science style
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

    # Colors (consistent with visualization)
    COLOR_PRIMARY = "#E74C3C"  # Red (arrows, goal)
    COLOR_SECONDARY = "#3498DB"  # Blue
    COLOR_AGENT = "#E67E22"  # Orange (agent)

    fig, ax = plt.subplots(figsize=(5, 3))

    # Raw rewards
    ax.plot(rewards, color=COLOR_SECONDARY, alpha=0.3, linewidth=1, label="Raw rewards")

    # Smoothed curve
    ax.plot(
        range(len(rewards_smooth)),
        rewards_smooth,
        color=COLOR_PRIMARY,
        linewidth=2,
        label="Moving average (window=50)",
    )

    ax.set_xlabel("Episode", fontweight="normal")
    ax.set_ylabel("Total Reward", fontweight="normal")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.3)
    ax.legend(frameon=False, loc="lower right")
    # ax.set_title("Training Progress", fontweight="normal", pad=10)

    plt.tight_layout()
    plt.savefig("recordings/rewards.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()

    print("Saved: recordings/rewards.png")


if __name__ == "__main__":
    rewards_path = Path("recordings/rewards.pkl")
    plot_rewards(rewards_path, smooth_window=50)

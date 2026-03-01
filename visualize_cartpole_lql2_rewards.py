"""
Visualize CartPole LQL training rewards from a PKL recording.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt


RECORDING_DIR = Path("recordings/cartpole_lql2")
REWARD_PKL_PATH = RECORDING_DIR / "rewards_raw.pkl"
REWARD_PLOT_PATH = RECORDING_DIR / "rewards_raw.png"


def main() -> None:
    if not REWARD_PKL_PATH.exists():
        raise FileNotFoundError(f"Missing rewards file: {REWARD_PKL_PATH}")

    with REWARD_PKL_PATH.open("rb") as f:
        rewards_raw = pickle.load(f)

    plt.figure(figsize=(10, 4))
    plt.plot(rewards_raw, label="Raw episode reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward (raw env reward)")
    plt.title("LQL CartPole Training Reward")
    plt.tight_layout()
    plt.savefig(REWARD_PLOT_PATH)
    plt.close()

    print(f"Loaded rewards PKL: {REWARD_PKL_PATH}")
    print(f"Saved rewards plot: {REWARD_PLOT_PATH}")


if __name__ == "__main__":
    main()

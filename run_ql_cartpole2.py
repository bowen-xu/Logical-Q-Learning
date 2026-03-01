"""
CartPole Q-learning training script using ql.agent.Agent.

Design aligned with run_1_round.py:
- split into run_episode/train/visualize helpers
- use Agent for epsilon-greedy action selection and Q updates
- train with tqdm progress bar
"""

from __future__ import annotations

import math
import pickle
import random
from pathlib import Path
from typing import Callable, Tuple

import gymnasium as gym
import numpy as np
from ql.agent import Agent
from sklearn.preprocessing import KBinsDiscretizer

try:
    from tqdm import trange
except Exception:

    def trange(n, **kwargs):  # type: ignore
        return range(n)


# Fixed experiment config (no CLI/env overrides)
SEED = 42
N_EPISODES = 1500
DEMO_MAX_STEPS = 500
ALPHA = 0.1
GAMMA = 0.98
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
FINAL_EVAL_EPSILON = 0.0
TERMINAL_PENALTY = -5.0
STEP_REWARD = 0.1
UPRIGHT_BONUS = 5.0
UPRIGHT_ANGLE_THRESHOLD = math.radians(3.0)
RECORDING_DIR = Path("recordings/cartpole_ql2")
REWARD_PKL_PATH = RECORDING_DIR / "rewards_raw.pkl"
DEMO_GIF_PATH = RECORDING_DIR / "cartpole_ql_demo.gif"


def make_discretizer(env: gym.Env) -> Callable[[np.ndarray], Tuple[int, ...]]:
    """Discretize CartPole observation into (angle_bin, pole_velocity_bin)."""
    n_bins = (6, 12)
    lower_bounds = [env.observation_space.low[2], -math.radians(50)]
    upper_bounds = [env.observation_space.high[2], math.radians(50)]

    est = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    est.fit([lower_bounds, upper_bounds])

    def discretize(obs: np.ndarray) -> Tuple[int, ...]:
        _, _, angle, pole_velocity = obs
        values = est.transform([[angle, pole_velocity]])[0]
        return tuple(map(int, values))

    return discretize


def run_episode(
    env: gym.Env,
    agent: Agent,
    discretize: Callable[[np.ndarray], Tuple[int, ...]],
) -> float:
    """Run one training episode and update Q values online."""
    obs, _ = env.reset()
    state = discretize(obs)
    done = False
    total_reward = 0.0

    while not done:
        action = agent.select_action(state)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize(next_obs)

        # Reward shaping aligned with run_lql_cartpole.py:
        # 1) failure termination -> negative reward
        # 2) normal states -> small positive reward
        # 3) near-upright and stable states -> extra positive bonus
        if terminated:
            ql_reward = TERMINAL_PENALTY
        else:
            ql_reward = STEP_REWARD
            _, _, angle, _ = next_obs
            if abs(angle) <= UPRIGHT_ANGLE_THRESHOLD:
                ql_reward += UPRIGHT_BONUS

        agent.update_q_state_action(state, action, ql_reward, next_state)
        state = next_state
        total_reward += reward

    agent.decay_epsilon()
    return total_reward


def train() -> tuple[Agent, list[float]]:
    random.seed(SEED)
    np.random.seed(SEED)

    env = gym.make("CartPole-v1")
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    env.reset(seed=SEED)

    discretize = make_discretizer(env)
    agent = Agent(
        actions=[0, 1],
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
    )
    rewards_raw: list[float] = []

    progress = trange(N_EPISODES, desc="Training", unit="ep")
    for episode in progress:
        agent.alpha = max(0.01, min(1.0, 1.0 - math.log10((episode + 1) / 25)))
        total_reward = run_episode(env, agent, discretize)
        rewards_raw.append(total_reward)
        if hasattr(progress, "set_postfix"):
            if episode % 50 == 0 or episode == N_EPISODES - 1:
                progress.set_postfix(
                    {
                        "reward": f"{total_reward:.0f}",
                        "eps": f"{agent.epsilon:.3f}",
                        "lr": f"{agent.alpha:.3f}",
                    }
                )

    env.close()
    return agent, rewards_raw


def save_rewards(rewards_raw: list[float]) -> None:
    RECORDING_DIR.mkdir(parents=True, exist_ok=True)
    with REWARD_PKL_PATH.open("wb") as f:
        pickle.dump(rewards_raw, f)
    print(f"Saved rewards PKL: {REWARD_PKL_PATH}")


def visualize(agent: Agent) -> None:
    """Record one greedy demo episode to recordings/cartpole_ql."""
    random.seed(SEED)
    np.random.seed(SEED)
    RECORDING_DIR.mkdir(parents=True, exist_ok=True)
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    discretize = make_discretizer(env)
    old_epsilon = agent.epsilon
    agent.epsilon = FINAL_EVAL_EPSILON  # pure exploitation for final demo

    captured_frames = []
    obs, _ = env.reset(seed=SEED + 10000)
    frame = env.render()
    if frame is not None:
        captured_frames.append(frame)

    done = False
    steps = 0
    while not done and steps < DEMO_MAX_STEPS:
        state = discretize(obs)
        action = agent.select_action(state)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
        frame = env.render()
        if frame is not None:
            captured_frames.append(frame)

    agent.epsilon = old_epsilon
    env.close()

    if captured_frames:
        try:
            import imageio.v2 as imageio

            imageio.mimsave(DEMO_GIF_PATH, captured_frames, fps=30)
            print(f"Saved demo GIF: {DEMO_GIF_PATH}")
        except Exception:
            print("Failed to save demo GIF (missing imageio).")


def main() -> None:
    print("=== CartPole Q-learning (ql.agent.Agent) ===")
    agent, rewards_raw = train()
    save_rewards(rewards_raw)
    print("Training finished. Start visualization...")
    visualize(agent)


if __name__ == "__main__":
    main()

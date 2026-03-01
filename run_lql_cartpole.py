"""
CartPole LQL training script using lql.agent.Agent.

Design aligned with run_1_round.py:
- split into run_episode/train/visualize helpers
- use Agent for epsilon-greedy action selection and updates
- train with tqdm progress bar
"""

from __future__ import annotations

import math
import random
import time
from typing import Callable, Tuple

import gymnasium as gym
import numpy as np
from lql.agent import Agent
from sklearn.preprocessing import KBinsDiscretizer

try:
    from tqdm import trange
except Exception:
    def trange(n, **kwargs):  # type: ignore
        return range(n)

# Fixed experiment config (no CLI/env overrides)
SEED = 42
N_EPISODES = 1000
DEMO_MAX_STEPS = 500
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
FINAL_EVAL_EPSILON = 0.0
TERMINAL_PENALTY = -10.0
STEP_REWARD = 0.05
UPRIGHT_BONUS = 1.0
UPRIGHT_ANGLE_THRESHOLD = math.radians(3.0)
UPRIGHT_ANGVEL_THRESHOLD = 0.5
UPRIGHT_CART_POS_THRESHOLD = 0.5
UPRIGHT_CART_VEL_THRESHOLD = 0.8

TERMINAL_FAILURE_PENALTY = TERMINAL_PENALTY


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
    """Run one training episode and update Agent online."""
    obs, _ = env.reset()
    state = discretize(obs)
    done = False
    total_reward = 0.0

    while not done:
        action = agent.select_action(state)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize(next_obs)

        # Reward shaping:
        # 1) failure termination -> negative reward
        # 2) normal states -> small positive reward
        # 3) near-upright and stable states -> extra positive bonus
        if terminated:
            lql_reward = TERMINAL_FAILURE_PENALTY
        else:
            lql_reward = STEP_REWARD
            cart_pos, cart_vel, angle, ang_vel = next_obs
            if (
                abs(angle) <= UPRIGHT_ANGLE_THRESHOLD
                and abs(ang_vel) <= UPRIGHT_ANGVEL_THRESHOLD
                and abs(cart_pos) <= UPRIGHT_CART_POS_THRESHOLD
                and abs(cart_vel) <= UPRIGHT_CART_VEL_THRESHOLD
            ):
                lql_reward += UPRIGHT_BONUS
        agent.update_q_state_action(state, action, lql_reward, next_state)
        state = next_state
        total_reward += reward

    agent.decay_epsilon()
    return total_reward


def train() -> Agent:
    random.seed(SEED)
    np.random.seed(SEED)

    env = gym.make("CartPole-v1")
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    env.reset(seed=SEED)

    discretize = make_discretizer(env)
    agent = Agent(
        actions=[0, 1],
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
    )

    progress = trange(N_EPISODES, desc="Training", unit="ep")
    for episode in progress:
        total_reward = run_episode(env, agent, discretize)
        if hasattr(progress, "set_postfix"):
            if episode % 50 == 0 or episode == N_EPISODES - 1:
                progress.set_postfix(
                    {
                        "reward": f"{total_reward:.0f}",
                        "eps": f"{agent.epsilon:.3f}",
                    }
                )

    env.close()
    return agent


def visualize(agent: Agent) -> None:
    """Visualize learned policy, fallback to GIF in headless environments."""
    random.seed(SEED)
    np.random.seed(SEED)

    use_human = True
    try:
        env = gym.make("CartPole-v1", render_mode="human")
    except Exception:
        use_human = False
        env = gym.make("CartPole-v1", render_mode="rgb_array")

    discretize = make_discretizer(env)
    old_epsilon = agent.epsilon
    agent.epsilon = FINAL_EVAL_EPSILON  # pure exploitation for final demo

    captured_frames = []
    for demo_idx in range(1):
        try:
            obs, _ = env.reset(seed=SEED + 10000 + demo_idx)
        except Exception:
            env.close()
            use_human = False
            env = gym.make("CartPole-v1", render_mode="rgb_array")
            discretize = make_discretizer(env)
            obs, _ = env.reset(seed=SEED + 10000 + demo_idx)

        done = False
        steps = 0
        while not done and steps < DEMO_MAX_STEPS:
            state = discretize(obs)
            action = agent.select_action(state)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            if use_human:
                time.sleep(0.02)
            else:
                frame = env.render()
                if frame is not None:
                    captured_frames.append(frame)

    agent.epsilon = old_epsilon
    env.close()

    if not use_human and captured_frames:
        output_path = "cartpole_demo.gif"
        try:
            import imageio.v2 as imageio

            imageio.mimsave(output_path, captured_frames, fps=30)
            print(f"No display detected. Saved visualization to {output_path}")
        except Exception:
            print("No display detected and failed to save GIF (missing imageio).")


def main() -> None:
    print("=== CartPole LQL (lql.agent.Agent) ===")
    agent = train()
    print("Training finished. Start visualization...")
    visualize(agent)


if __name__ == "__main__":
    main()

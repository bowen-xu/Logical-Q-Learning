import yaml
import random
from pathlib import Path

from lql.agent import Agent

import matplotlib.pyplot as plt
from tqdm import tqdm
from grid_world import GridWorld

import numpy as np

def smooth(rewards, window=10):
    return np.convolve(rewards, np.ones(window)/window, mode='valid')


def run_episode(env: GridWorld, agent: Agent, max_steps: int = 20) -> float:
    """Run one episode, return total reward"""
    state = env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        # Select action
        action = agent.select_action(state)

        # Execute action
        next_state, reward = env.step(state, action)

        # Update Agent
        agent.update_q_state_action(state, action, reward, next_state)

        total_reward += reward
        state = next_state

        # Reached goal, end early
        if state == env.goal:
            break

    # Decay epsilon
    agent.decay_epsilon()

    return total_reward


def print_policy(env: GridWorld, agent: Agent):
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Disable exploration to show learned policy
    for y in range(env.grid_size - 1, -1, -1):
        row = []
        for x in range(env.grid_size):
            state = (x, y)
            if state == env.goal:
                row.append(" G")
            elif state == env.start:
                row.append(" S")
            elif state in env.obstacles:
                row.append(" ■")
            elif not env.state_is_valid(state):
                row.append("  ")
            else:
                action = agent.select_action(state)
                arrow = {0: "↑", 1: "→", 2: "↓", 3: "←"}[action]
                row.append(f" {arrow}")
        print("  " + " ".join(row))
    agent.epsilon = old_epsilon  # Restore original epsilon


def main():
    print("=== GridWorld Environment Test ===\n")

    env = GridWorld(
        grid_size=7,
        obstacle_probability=0.2,
        step_reward=-0.1,
        goal_reward=10.0,
        invalid_move_penalty=-1.0,
    )

    agent = Agent(
        actions=env.actions, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995
    )

    print("Environment config:")
    print(f"  Grid: {env.grid_size}x{env.grid_size}")
    print(f"  Start: {env.start}, Goal: {env.goal}")
    print(f"  Step reward: {env.step_reward}")
    print(f"  Goal reward: {env.goal_reward}")
    print(f"  Invalid move penalty: {env.invalid_move_penalty}")
    print()

    print("Agent config:")
    print(f"  Epsilon: {agent.epsilon}")
    print(f"  Epsilon min: {agent.epsilon_min}")
    print(f"  Epsilon decay: {agent.epsilon_decay}")
    print()

    # Store initial epsilon for reset
    initial_epsilon = agent.epsilon

    # Run training in two phases
    rewards = []
    num_episodes_phase1 = 1500
    num_episodes_phase2 = 1500

    # ============ PHASE 1 ============
    env.start = (0, env.grid_size - 1)  # top-left (0, 2)
    env.goal = (env.grid_size - 1, 0)  # bottom-right (2, 0)

    print(f"=== PHASE 1: Training (Episodes 1-{num_episodes_phase1}) ===\n")
    for episode in tqdm(range(num_episodes_phase1), desc="Phase 1 Training"):
        total_reward = run_episode(env, agent, max_steps=200)
        if (episode) % 500 == 0:
            tqdm.write(
                f"[{episode + 1}/{num_episodes_phase1}] Reward: {total_reward:.1f}"
            )
        rewards.append(total_reward)

    # Show final learned policy after Phase 1
    print("=== Learned Phase 1 Policy ===")
    print("Optimal action from each state:")
    print_policy(env, agent)

    # ============ GOAL CHANGE & EPSILON RESET ============
    print("\n=== Changing Goal and Resetting Epsilon ===")
    old_goal = env.goal
    excluded = {env.start, old_goal}
    new_goal = select_random_valid_goal(env, excluded)
    env.goal = new_goal

    print(f"Old goal: {old_goal}")
    print(f"New goal: {new_goal}")

    agent.epsilon = initial_epsilon
    # agent.epsilon_decay = 0.999
    print(f"Epsilon reset to: {agent.epsilon}")
    print()

    # ============ PHASE 2 ============
    env.start = (0, 0)  # bottom-left
    env.goal = (env.grid_size - 1, env.grid_size - 1)  # top-right

    print(f"=== PHASE 2: Training (Episodes {num_episodes_phase2+1}-{num_episodes_phase1 + num_episodes_phase2}) ===\n")
    for episode in tqdm(range(num_episodes_phase2), desc="Phase 2 Training"):
        total_reward = run_episode(env, agent, max_steps=200)
        if (episode) % 500 == 0:
            tqdm.write(
                f"[{episode + 1}/{num_episodes_phase2}] Reward: {total_reward:.1f}"
            )
        rewards.append(total_reward)
    

    # Show final learned policy after Phase 2
    print("=== Learned Phase 2 Policy ===")
    print("Optimal action from each state:")
    print_policy(env, agent)

    print("\nTest complete!")
    smoothed_rewards = smooth(rewards, window=50)
    plt.plot(rewards)
    plt.axvline(num_episodes_phase1, color="red", linestyle="--", label="Phase 1/2 Switch")
    plt.plot(range(len(rewards) - len(smoothed_rewards), len(rewards)), smoothed_rewards, color="blue", label="Smoothed Reward")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.savefig("rewards_plot_gridworld.png")


if __name__ == "__main__":
    main()

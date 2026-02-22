import yaml
import random
from pathlib import Path

from lql.agent import Agent

import matplotlib.pyplot as plt
from tqdm import tqdm
from grid_world import GridWorld


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
        actions=env.actions, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999
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

    # Run training in two phases
    rewards = []
    num_episodes = 3000

    for episode in tqdm(range(num_episodes)):
        total_reward = run_episode(env, agent, max_steps=100)
        if (episode) % 500 == 0:
            tqdm.write(f"[{episode + 1}/{num_episodes}] Reward: {total_reward:.1f}")
        rewards.append(total_reward)

    # Show final learned policy
    print("=== Learned Policy ===")
    print("Optimal action from each state:")
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

    print("\nTest complete!")
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.savefig("rewards_plot.png")


if __name__ == "__main__":
    main()

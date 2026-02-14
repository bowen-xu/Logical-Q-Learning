"""
Goal-Switching Experiment
Uses Q-learning agent in a linear chain environment.

Phase 1: G1 is the goal (has reward)
Phase 2: G2 is the goal (G1 no reward, G2 has reward)

Observes how the agent adapts to goal switching.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

from q_nal.linear_chain import LinearChain, State
from q_nal.agent_q import AgentQ


def run_episode(env: LinearChain, agent, max_steps: int = 20) -> tuple[float, int]:
    """Run a single episode, return (total_reward, steps)"""
    state = env.reset()
    total_reward = 0.0
    steps = 0

    for _ in range(max_steps):
        action = agent.select_action(state)
        next_state, reward = env.step(state, action)
        agent.update_q_state_action(state, action, reward, next_state)
        total_reward += reward
        steps += 1

        if env.is_terminal(next_state):
            break
        state = next_state

    return total_reward, steps


def print_q_table(agent_q: AgentQ, states: list[State], actions: list[int]):
    """Print Q-table for Q-learning agent"""
    print("\nQ-Table (Q-learning Agent):")
    print("-" * 50)
    header = "State  | " + " | ".join(f"A{a}" for a in actions) + " | Best"
    print(header)
    print("-" * 50)

    for state in states:
        q_values = agent_q._get_q(state)
        best_action = int(np.argmax(q_values))
        row = (
            f"{state:6} | "
            + " | ".join(f"{q:5.2f}" for q in q_values)
            + f" | A{best_action}"
        )
        print(row)


def plot_results(results: dict, phase1_episodes: int, phase2_episodes: int):
    """Plot reward curves and Q-value curves"""

    # ========== Plot 1: Reward Curve ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Reward curve
    ax1 = axes[0, 0]
    all_rewards = results["phase1"] + results["phase2"]
    episodes = range(1, len(all_rewards) + 1)

    ax1.plot(episodes, all_rewards, alpha=0.6, linewidth=1, color="blue")

    # Moving average
    window = 10
    if len(all_rewards) >= window:
        ma = np.convolve(all_rewards, np.ones(window) / window, mode="valid")
        ma_episodes = range(window, len(all_rewards) + 1)
        ax1.plot(ma_episodes, ma, linewidth=2, color="red", label=f"MA({window})")

    ax1.axvline(
        x=phase1_episodes,
        color="green",
        linestyle="--",
        linewidth=2,
        label="Goal Switch",
    )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Reward per Episode")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Phase 1 and Phase 2 separate
    ax2 = axes[0, 1]
    ax2.plot(
        range(1, phase1_episodes + 1),
        results["phase1"],
        label="Phase 1 (G1)",
        color="blue",
        alpha=0.7,
    )
    ax2.plot(
        range(phase1_episodes + 1, phase1_episodes + phase2_episodes + 1),
        results["phase2"],
        label="Phase 2 (G2)",
        color="orange",
        alpha=0.7,
    )
    ax2.axvline(
        x=phase1_episodes,
        color="green",
        linestyle="--",
        linewidth=2,
        label="Goal Switch",
    )
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Reward")
    ax2.set_title("Reward by Phase")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ========== Plot 2: Q-Value Curves ==========
    q_history = results.get("q_history", {})

    if q_history:
        # Get non-terminal states (exclude G1, G2)
        non_terminal_states = [s for s in q_history.keys() if s not in ["G1", "G2"]]

        # Plot Q-values for each state (all actions)
        ax3 = axes[1, 0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(non_terminal_states)))

        for idx, state in enumerate(non_terminal_states):
            state_q_history = np.array(
                q_history[state]
            )  # shape: (episodes, num_actions)
            for action in range(state_q_history.shape[1]):
                label = f"{state}-A{action}" if len(non_terminal_states) <= 6 else None
                ax3.plot(
                    range(1, len(state_q_history) + 1),
                    state_q_history[:, action],
                    label=label,
                    alpha=0.7,
                    linewidth=1,
                )

        ax3.axvline(
            x=phase1_episodes,
            color="green",
            linestyle="--",
            linewidth=2,
            label="Goal Switch",
        )
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Q-Value")
        ax3.set_title("Q-Values per State-Action (All)")
        if len(non_terminal_states) <= 6:
            ax3.legend(loc="best", fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Plot max Q-value per state
        ax4 = axes[1, 1]
        for idx, state in enumerate(non_terminal_states):
            state_q_history = np.array(q_history[state])
            max_q = np.max(state_q_history, axis=1)
            ax4.plot(
                range(1, len(max_q) + 1),
                max_q,
                label=state,
                color=colors[idx],
                alpha=0.8,
                linewidth=1.5,
            )

        ax4.axvline(
            x=phase1_episodes,
            color="green",
            linestyle="--",
            linewidth=2,
            label="Goal Switch",
        )
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Max Q-Value")
        ax4.set_title("Max Q-Value per State")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiment_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nPlots saved to 'experiment_results.png'")


def run_experiment(
    phase1_episodes: int = 100,
    phase2_episodes: int = 100,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.3,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.99,
):
    """Run the goal-switching experiment"""

    # Create environment
    env = LinearChain(step_reward=-1.0, goal_reward=10.0, invalid_move_penalty=-5.0)

    # Create Q-learning agent
    agent_q = AgentQ(
        actions=env.actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    # Results storage
    results = {
        "phase1": [],
        "phase2": [],
        "q_history": {},
    }

    # Track Q-values every episode
    for state in env.states:
        results["q_history"][state] = []

    # ========== Phase 1: G1 is the goal ==========
    print("=" * 60)
    print("PHASE 1: Goal is G1")
    print("=" * 60)
    env.set_goal("G1")

    for ep in range(phase1_episodes):
        reward_q, steps_q = run_episode(env, agent_q)
        results["phase1"].append(reward_q)
        agent_q.decay_epsilon()

        # Track Q-values
        for state in env.states:
            q_values = agent_q._get_q(state).copy()
            results["q_history"][state].append(q_values)

        if (ep + 1) % 20 == 0:
            print(f"Episode {ep + 1}: reward={reward_q:.1f}")

    print("\n--- Phase 1 Results ---")
    print(f"Q-learning avg reward (last 20): {np.mean(results['phase1'][-20:]):.2f}")

    print_q_table(agent_q, env.states, env.actions)

    # ========== Phase 2: G2 is the goal ==========
    print("\n" + "=" * 60)
    print("PHASE 2: Goal switched to G2")
    print("=" * 60)
    env.set_goal("G2")

    # Reset epsilon for exploration in new phase
    agent_q.epsilon = epsilon

    for ep in range(phase2_episodes):
        reward_q, steps_q = run_episode(env, agent_q)
        results["phase2"].append(reward_q)
        agent_q.decay_epsilon()

        # Track Q-values
        for state in env.states:
            q_values = agent_q._get_q(state).copy()
            results["q_history"][state].append(q_values)

        if (ep + 1) % 20 == 0:
            print(f"Episode {ep + 1}: reward={reward_q:.1f}")

    print("\n--- Phase 2 Results ---")
    print(f"Q-learning avg reward (last 20): {np.mean(results['phase2'][-20:]):.2f}")

    print_q_table(agent_q, env.states, env.actions)

    # Plot results
    plot_results(results, phase1_episodes, phase2_episodes)

    return results, agent_q


if __name__ == "__main__":
    results, agent_q = run_experiment(
        phase1_episodes=100,
        phase2_episodes=100,
    )

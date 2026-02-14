"""
Goal-Switching Experiment with AgentNAL
Uses NAL (Neural-Symbolic Agent with Logic) in a linear chain environment.

Phase 1: G1 is the goal (has reward)
Phase 2: G2 is the goal (G1 no reward, G2 has reward)

Observes how the NAL agent adapts to goal switching.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

from q_nal.linear_chain import LinearChain, State
from q_nal.agent_nal import AgentNAL


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


def get_nal_values(agent_nal: AgentNAL, states: list[State]) -> dict:
    """
    Extract 'Q-like' values from NAL agent.
    For each state, get the max desire value across all sequences.
    Returns dict: {state: {'desire_values': array, 'best_action': int}}
    """
    result = {}
    for state in states:
        sequences = agent_nal.sequence_table.get(state, [])
        if not sequences:
            # No sequences learned yet, return default
            result[state] = {
                "desire_values": np.zeros(len(agent_nal.actions)),
                "best_action": 0,
                "max_desire": 0.5,
            }
            continue

        # Get desire values for each action
        desire_values = []
        for action in agent_nal.actions:
            # Find sequence with this action
            seq_with_action = None
            for seq in sequences:
                if seq.components[1].value == action:
                    seq_with_action = seq
                    break
            if seq_with_action:
                desire_values.append(seq_with_action.desire.desirev.e)
            else:
                desire_values.append(0.5)  # default

        desire_values = np.array(desire_values)
        best_action = int(np.argmax(desire_values))
        max_desire = np.max(desire_values)

        result[state] = {
            "desire_values": desire_values,
            "best_action": best_action,
            "max_desire": max_desire,
        }

    return result


def print_nal_table(agent_nal: AgentNAL, states: list[State], actions: list[int]):
    """Print NAL desire table for NAL agent"""
    values = get_nal_values(agent_nal, states)

    print("\nNAL Desire Table:")
    print("-" * 50)
    header = "State  | " + " | ".join(f"A{a}" for a in actions) + " | Best"
    print(header)
    print("-" * 50)

    for state in states:
        v = values[state]
        desire_values = v["desire_values"]
        best_action = v["best_action"]
        row = (
            f"{state:6} | "
            + " | ".join(f"{d:5.2f}" for d in desire_values)
            + f" | A{best_action}"
        )
        print(row)


def plot_results(results: dict, phase1_episodes: int, phase2_episodes: int):
    """Plot reward curves and NAL desire-value curves"""

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

    # ========== Plot 2: NAL Desire-Value Curves ==========
    nal_history = results.get("nal_history", {})

    if nal_history:
        # Get non-terminal states (exclude G1, G2)
        non_terminal_states = [s for s in nal_history.keys() if s not in ["G1", "G2"]]

        # Plot desire values for each state-action
        ax3 = axes[1, 0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(non_terminal_states)))

        for idx, state in enumerate(non_terminal_states):
            state_history = np.array(
                [v["desire_values"] for v in nal_history[state]]
            )  # shape: (episodes, num_actions)
            for action in range(state_history.shape[1]):
                label = f"{state}-A{action}" if len(non_terminal_states) <= 6 else None
                ax3.plot(
                    range(1, len(state_history) + 1),
                    state_history[:, action],
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
        ax3.set_ylabel("Desire Value (e)")
        ax3.set_title("NAL Desire Values per State-Action")
        if len(non_terminal_states) <= 6:
            ax3.legend(loc="best", fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Plot max desire value per state
        ax4 = axes[1, 1]
        for idx, state in enumerate(non_terminal_states):
            state_history = [v["max_desire"] for v in nal_history[state]]
            ax4.plot(
                range(1, len(state_history) + 1),
                state_history,
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
        ax4.set_ylabel("Max Desire Value (e)")
        ax4.set_title("Max Desire Value per State")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiment_nal_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nPlots saved to 'experiment_nal_results.png'")


def run_experiment(
    phase1_episodes: int = 100,
    phase2_episodes: int = 100,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.3,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.99,
):
    """Run the goal-switching experiment with NAL agent"""

    # Create environment
    env = LinearChain(step_reward=-1.0, goal_reward=10.0, invalid_move_penalty=-5.0)

    # Create NAL agent
    agent_nal = AgentNAL(
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
        "nal_history": {},
    }

    # Track NAL values every episode
    for state in env.states:
        results["nal_history"][state] = []

    # ========== Phase 1: G1 is the goal ==========
    print("=" * 60)
    print("PHASE 1: Goal is G1 (NAL Agent)")
    print("=" * 60)
    env.set_goal("G1")

    for ep in range(phase1_episodes):
        reward_nal, steps_nal = run_episode(env, agent_nal)
        results["phase1"].append(reward_nal)
        agent_nal.decay_epsilon()

        # Track NAL desire values
        nal_values = get_nal_values(agent_nal, env.states)
        for state in env.states:
            results["nal_history"][state].append(nal_values[state])

        if (ep + 1) % 20 == 0:
            print(f"Episode {ep + 1}: reward={reward_nal:.1f}")

    print("\n--- Phase 1 Results ---")
    print(f"NAL avg reward (last 20): {np.mean(results['phase1'][-20:]):.2f}")

    print_nal_table(agent_nal, env.states, env.actions)

    # ========== Phase 2: G2 is the goal ==========
    print("\n" + "=" * 60)
    print("PHASE 2: Goal switched to G2 (NAL Agent)")
    print("=" * 60)
    env.set_goal("G2")

    # Reset epsilon for exploration in new phase
    agent_nal.epsilon = epsilon

    for ep in range(phase2_episodes):
        reward_nal, steps_nal = run_episode(env, agent_nal)
        results["phase2"].append(reward_nal)
        agent_nal.decay_epsilon()

        # Track NAL desire values
        nal_values = get_nal_values(agent_nal, env.states)
        for state in env.states:
            results["nal_history"][state].append(nal_values[state])

        if (ep + 1) % 20 == 0:
            print(f"Episode {ep + 1}: reward={reward_nal:.1f}")

    print("\n--- Phase 2 Results ---")
    print(f"NAL avg reward (last 20): {np.mean(results['phase2'][-20:]):.2f}")

    print_nal_table(agent_nal, env.states, env.actions)

    # Plot results
    plot_results(results, phase1_episodes, phase2_episodes)

    return results, agent_nal


if __name__ == "__main__":
    results, agent_nal = run_experiment(
        phase1_episodes=100,
        phase2_episodes=100,
    )

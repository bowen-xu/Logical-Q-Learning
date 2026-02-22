import yaml
import random
from pathlib import Path

from lql.agent import Agent

import matplotlib.pyplot as plt
from tqdm import tqdm
from linear_chain import LinearChain


def run_episode(env: LinearChain, agent: Agent, max_steps: int = 20) -> float:
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
    print("=== LinearChain Environment Test ===\n")

    env = LinearChain(
        step_reward=-1.0,
        goal_reward=10.0,
    )

    agent = Agent(
        actions=env.actions,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99
    )

    print("Environment config:")
    print(f"  States: {env.states}")
    print(f"  Actions: {env.actions}")
    print(f"  Step reward: {env.step_reward}")
    print(f"  Goal reward: {env.goal_reward}")
    print()

    print("Agent config:")
    print(f"  Epsilon: {agent.epsilon}")
    print(f"  Epsilon min: {agent.epsilon_min}")
    print(f"  Epsilon decay: {agent.epsilon_decay}")
    print()


    # Run training in two phases
    rewards = []
    num_episodes = 300

    for episode in tqdm(range(num_episodes)):
        total_reward = run_episode(env, agent, max_steps=8)
        if (episode) % 100 == 0:
            tqdm.write(
                f"[{episode + 1}/{num_episodes}] Reward: {total_reward:.1f}"
            )
        rewards.append(total_reward)


    print("\nTest complete!")
    plt.ioff()
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.savefig("rewards_plot_linearchain.png")
    distrib_desirev = dict([(c.term_str(), c.desire.desirev) for c in agent.conet.sequences.values()])

    distrib_e_desirev = [(c.term_str(), c.desire.desirev.e) for c in agent.conet.sequences.values()]
    distrib_e_desirev = sorted(distrib_e_desirev, key=lambda x: x[1], reverse=True)
    plt.figure()
    plt.plot([x[1] for x in distrib_e_desirev])
    plt.xticks(ticks=range(len(distrib_e_desirev)), labels=[x[0] for x in distrib_e_desirev], rotation=30)

    plt.xlabel("Sequence Index")
    plt.ylabel("DesireV.e")
    plt.title("DesireV.e for Learned Sequences")
    plt.savefig("desirev_e_plot_linearchain.png")

if __name__ == "__main__":
    main()
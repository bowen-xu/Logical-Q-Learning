"""
Test AgentNAL behavior in simplified environment
"""

import yaml
from pathlib import Path

from q_nal.environment_nal import EnvironmentNal
from q_nal.agent_nal import AgentNAL

import matplotlib.pyplot as plt
from tqdm import tqdm


def load_config():
    """Load configuration from config.yml"""
    config_path = Path(__file__).parent / "config.yml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_episode(env: EnvironmentNal, agent: AgentNAL, max_steps: int = 20) -> float:
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


def print_step(
    env: EnvironmentNal, agent: AgentNAL, state, action, reward, next_state, step: int
):
    """Print step information"""
    action_names = {0: "up", 1: "right", 2: "down", 3: "left"}
    print(f"Step {step}:")
    print(f"  State: {state}")
    print(f"  Action: {action} ({action_names.get(action, '?')})")
    print(f"  Reward: {reward}")
    print(f"  Next state: {next_state}")
    print(f"  Epsilon: {agent.epsilon:.4f}")
    print(f"  Sequences: {len(agent.sequences)}")
    print()


def main():
    print("=== AgentNAL Simplified Environment Test ===\n")

    # Load configuration from YAML
    config = load_config()
    env_config = config["environment"]
    agent_config = config["agent"]
    trainer_config = config["trainer"]

    # Create environment and Agent
    env = EnvironmentNal(
        grid_size=env_config["grid_size"],
        obstacle_probability=env_config.get("obstacle_probability", 0.2),
        step_reward=env_config["step_reward"],
        goal_reward=env_config["goal_reward"],
        invalid_move_penalty=env_config["invalid_move_penalty"],
    )

    agent = AgentNAL(
        actions=env.actions,
        alpha=agent_config["alpha"],
        gamma=agent_config["gamma"],
        epsilon=agent_config["epsilon"],
        epsilon_min=agent_config["epsilon_min"],
        epsilon_decay=agent_config["epsilon_decay"],
    )

    print("Environment config:")
    print(f"  Grid: {env.grid_size}x{env.grid_size}")
    print(f"  Start: {env.start}, Goal: {env.goal}")
    print(f"  Step reward: {env.step_reward}")
    print(f"  Goal reward: {env.goal_reward}")
    print(f"  Invalid move penalty: {env.invalid_move_penalty}")
    print()

    print("Agent config:")
    print(f"  Alpha (learning rate): {agent.alpha}")
    print(f"  Gamma (discount): {agent.gamma}")
    print(f"  Epsilon (exploration): {agent.epsilon}")
    print()

    # Run multiple episodes
    num_episodes = trainer_config.get("max_episodes", 3000)

    rewards = []
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
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
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.savefig('rewards_plot.png')
    

if __name__ == "__main__":
    main()

"""
Linear Chain Environment
7 states: S0, S1, S2, S3, S4, S5, S6
Two paths:
  - S0 -> S1 -> S2 -> S3
  - S0 -> S4 -> S5 -> S6
"""

from typing import Literal


type State = str


class LinearChain:
    r"""
    Linear chain environment with two branches:
    
         S1 -- S2 -- S3
        /
    S0 
        \
         S4 -- S5 -- S6
    
    Actions:
    - 1: forward
    - 2: backward
    """

    def __init__(
        self,
        step_reward: float = -0.1,
        goal_reward: float = 10.0,
    ):
        self.step_reward = step_reward
        self.goal_reward = goal_reward

        # All states
        self.states: list[State] = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]

        # Actions: 1=forward, 2=backward
        self.actions: list[int] = [1, 2]

        # Start state
        self.start: State = "S0"

        # Current goal
        self.goal: State = "S3"

        # Transition table: (state, action) -> next_state
        # Actions: 0=forward, 1=invalid (stay), 2=backward
        self.transitions: dict[tuple[State, int], State] = {
            # From S0 (junction)
            ("S0", 1): "S1",  # forward to upper path
            ("S0", 2): "S4",  # backward to lower path
            # Upper path: S0 -> S1 -> S2 -> S3
            ("S1", 1): "S2",  # forward
            ("S1", 2): "S0",  # backward
            ("S2", 1): "S3",  # forward
            ("S2", 2): "S1",  # backward
            # Lower path: S0 -> S4 -> S5 -> S6
            ("S4", 1): "S5",  # forward
            ("S4", 2): "S0",  # backward
            ("S5", 1): "S6",  # forward
            ("S5", 2): "S4",  # backward
            # Terminal states (self-loop)
            ("S3", 1): "S3",
            ("S3", 2): "S2",
            ("S6", 1): "S6",
            ("S6", 2): "S5",
        }

    def reset(self) -> State:
        """Reset environment, return start state"""
        return self.start

    def set_goal(self, goal: State):
        """Set the current goal state"""
        assert goal in self.states, f"Goal must be one of {self.states}"
        self.goal = goal

    def is_terminal(self, state: State) -> bool:
        """Check if state is terminal"""
        return state in ["S3", "S6"]

    def step(self, state: State, action: int) -> tuple[State, float]:
        """
        Execute action, return (next_state, reward)
        """
        next_state = self.transitions.get((state, action), state)

        # Check if move was invalid (stayed in same state but not terminal)
        if next_state == state and not self.is_terminal(state):
            reward = self.step_reward
            return next_state, reward

        # Normal move
        reward = self.step_reward

        # Check if reached goal
        if next_state == self.goal:
            reward += self.goal_reward

        return next_state, reward

    def render(self, agent_state: State) -> str:
        """Text rendering of the environment"""
        lines = [
            f"Current Goal: {self.goal}",
            "",
            "    S1 → S2 → S3",
            "   ↗",
            f" S0     (Agent: {agent_state})",
            "   ↘",
            "    S4 → S5 → S6",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    # Simple test
    env = LinearChain()

    print("=== Linear Chain Environment Test ===")
    print(f"States: {env.states}")
    print(f"Actions: {env.actions}")
    print(f"Start: {env.start}")
    print(f"Goal: {env.goal}")
    print()
    print(env.render("S0"))
    print()

    # Test path to S3
    state = env.reset()
    print(f"Testing path to S3:")
    print(f"  Start: {state}")
    state, reward = env.step(state, 1)  # S0 -> S1 (upper path)
    print(f"  Action 1 (upper): {state}, reward={reward}")
    state, reward = env.step(state, 1)  # S1 -> S2 (forward)
    print(f"  Action 1 (forward): {state}, reward={reward}")
    state, reward = env.step(state, 1)  # S2 -> S3 (forward)
    print(f"  Action 1 (forward): {state}, reward={reward}")
    state, reward = env.step(state, 1)  # S3 -> S3 (forward, invalid)
    print(f"  Action 1 (forward): {state}, reward={reward}")
    print()

    # Test path to S6
    env.set_goal("S6")
    state = env.reset()
    print(f"Testing path to S6:")
    print(f"  Start: {state}")
    state, reward = env.step(state, 0)  # S0 -> S3 (lower path)
    print(f"  Action 0 (lower): {state}, reward={reward}")
    state, reward = env.step(state, 1)  # S3 -> S4 (forward)
    print(f"  Action 1 (forward): {state}, reward={reward}")
    state, reward = env.step(state, 1)  # S4 -> S5 (forward)
    print(f"  Action 1 (forward): {state}, reward={reward}")
    state, reward = env.step(state, 1)  # S5 -> S6 (forward)
    print(f"  Action 1 (forward): {state}, reward={reward}")
    print()

    # Test backward movement
    print("Testing backward movement from S2:")
    state = "S2"
    print(f"  Current: {state}")
    state, reward = env.step(state, 2)  # S2 -> S1 (backward)
    print(f"  Action 2 (backward): {state}, reward={reward}")
    state, reward = env.step(state, 2)  # S1 -> S0 (backward)
    print(f"  Action 2 (backward): {state}, reward={reward}")

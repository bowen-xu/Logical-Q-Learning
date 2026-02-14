"""
Linear Chain Environment for goal-switching experiment
7 states: S0, S1, S2, S3, S4, G1, G2
Two paths:
  - S0 -> S1 -> S2 -> G1
  - S0 -> S3 -> S4 -> G2
"""

from typing import Literal


type State = str


class LinearChain:
    """
    Linear chain environment with two branches:
    
         S1 -> S2 -> G1
        /
    S0 
        \
         S3 -> S4 -> G2
    
    Actions:
    - 0: forward (towards goal) or take upper path from S0
    - 1: take lower path from S0, or invalid (stay) on other states
    - 2: backward (towards S0) or invalid at S0
    """
    
    def __init__(
        self,
        step_reward: float = -1.0,
        goal_reward: float = 10.0,
        invalid_move_penalty: float = -5.0,
    ):
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.invalid_move_penalty = invalid_move_penalty
        
        # All states
        self.states: list[State] = ["S0", "S1", "S2", "S3", "S4", "G1", "G2"]
        
        # Actions: 0=upper path, 1=lower path, 2=forward
        self.actions: list[int] = [0, 1, 2]
        
        # Start state
        self.start: State = "S0"
        
        # Current goal (can be switched)
        self.goal: State = "G1"
        
        # Transition table: (state, action) -> next_state
        # Actions: 0=forward, 1=invalid (stay), 2=backward
        self.transitions: dict[tuple[State, int], State] = {
            # From S0 (junction)
            ("S0", 0): "S1",  # forward to upper path
            ("S0", 1): "S3",  # forward to lower path
            ("S0", 2): "S0",  # backward (invalid, no previous state)
            
            # Upper path: S0 -> S1 -> S2 -> G1
            ("S1", 0): "S2",  # forward
            ("S1", 1): "S1",  # invalid (stay)
            ("S1", 2): "S0",  # backward
            
            ("S2", 0): "G1",  # forward to G1
            ("S2", 1): "S2",  # invalid (stay)
            ("S2", 2): "S1",  # backward
            
            # Lower path: S0 -> S3 -> S4 -> G2
            ("S3", 0): "S4",  # forward
            ("S3", 1): "S3",  # invalid (stay)
            ("S3", 2): "S0",  # backward
            
            ("S4", 0): "G2",  # forward to G2
            ("S4", 1): "S4",  # invalid (stay)
            ("S4", 2): "S3",  # backward
            
            # Terminal states (self-loop)
            ("G1", 0): "G1",
            ("G1", 1): "G1",
            ("G1", 2): "S2",  # can go backward from G1
            
            ("G2", 0): "G2",
            ("G2", 1): "G2",
            ("G2", 2): "S4",  # can go backward from G2
        }
    
    def reset(self) -> State:
        """Reset environment, return start state"""
        return self.start
    
    def set_goal(self, goal: State):
        """Set the current goal state"""
        assert goal in ["G1", "G2"], "Goal must be G1 or G2"
        self.goal = goal
    
    def is_terminal(self, state: State) -> bool:
        """Check if state is terminal"""
        return state in ["G1", "G2"]
    
    def step(self, state: State, action: int) -> tuple[State, float]:
        """
        Execute action, return (next_state, reward)
        """
        next_state = self.transitions.get((state, action), state)
        
        # Check if move was invalid (stayed in same state but not terminal)
        if next_state == state and not self.is_terminal(state):
            reward = self.step_reward + self.invalid_move_penalty
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
            "    S1 → S2 → G1",
            "   ↗",
            f" S0     (Agent: {agent_state})",
            "   ↘",
            "    S3 → S4 → G2",
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
    
    # Test path to G1
    state = env.reset()
    print(f"Testing path to G1:")
    print(f"  Start: {state}")
    state, reward = env.step(state, 0)  # S0 -> S1 (upper path)
    print(f"  Action 0 (upper): {state}, reward={reward}")
    state, reward = env.step(state, 0)  # S1 -> S2 (forward)
    print(f"  Action 0 (forward): {state}, reward={reward}")
    state, reward = env.step(state, 0)  # S2 -> G1 (forward)
    print(f"  Action 0 (forward): {state}, reward={reward}")
    print()
    
    # Test path to G2
    env.set_goal("G2")
    state = env.reset()
    print(f"Testing path to G2:")
    print(f"  Start: {state}")
    state, reward = env.step(state, 1)  # S0 -> S3 (lower path)
    print(f"  Action 1 (lower): {state}, reward={reward}")
    state, reward = env.step(state, 0)  # S3 -> S4 (forward)
    print(f"  Action 0 (forward): {state}, reward={reward}")
    state, reward = env.step(state, 0)  # S4 -> G2 (forward)
    print(f"  Action 0 (forward): {state}, reward={reward}")
    print()
    
    # Test backward movement
    print("Testing backward movement from S2:")
    state = "S2"
    print(f"  Current: {state}")
    state, reward = env.step(state, 2)  # S2 -> S1 (backward)
    print(f"  Action 2 (backward): {state}, reward={reward}")
    state, reward = env.step(state, 2)  # S1 -> S0 (backward)
    print(f"  Action 2 (backward): {state}, reward={reward}")

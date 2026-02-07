"""
RL Agent: Q-learning Agent
- stores a Q-table with a Q-value for each state-action pair
- selects actions based on epsilon-greedy policy
- updates the Q-table based on observed rewards and state transitions
- decays epsilon

Classes:
- Agent
    - __init__(self, actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay)
    - _get_q(self, state)
    - select_action(self, state)
    - update_q_state_action(self, state, action, reward, next_state)
    - decay_epsilon(self)
"""

import random
import numpy as np

from .grid_world import State
from .agent import Agent


class AgentQ(Agent):

    def __init__(
        self,
        actions: list[int],
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
    ):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: dict[State, np.ndarray] = {}

    def _get_q(self, state: State) -> np.ndarray:
        """Get the Q-value for all actions for the given state."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions), dtype=np.float32)
        return self.q_table[state]

    def select_action(self, state: State) -> int:
        """Select an action based on epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q = self._get_q(state)
        return int(np.argmax(q))

    def update_q_state_action(
        self, state: State, action: int, reward: float, next_state: State
    ):
        """Update the Q-value for the given state-action pair."""
        max_q_next_state = np.max(self._get_q(next_state))
        td_target = reward + self.gamma * max_q_next_state
        td_error = td_target - self._get_q(state)[action]
        self._get_q(state)[action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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

from .environment_nal import State
from weakref import WeakSet, WeakValueDictionary


from .nal import DesireV, TruthV


class Concept:
    def __init__(self, value):
        self.value = value
        self.desirev: DesireV = DesireV(0.5, 0.0)
        self.truthv: TruthV = TruthV(0.5, 0.0)

    def __eq__(self, other):
        if not isinstance(other, Concept):
            return False
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return f"{self.value}"


class Sequence:
    def __init__(self, *args: Concept):
        self.components = tuple(args)
        self.desirev: DesireV = DesireV(0.5, 0.0)
        self.truthv: TruthV = TruthV(0.5, 0.0)

    def __len__(self):
        return len(self.components)

    def __hash__(self):
        return hash(self.components)

    def __eq__(self, other):
        if not isinstance(other, Sequence):
            return False
        if hash(self) != hash(other):
            return False
        return self.components == other.components

    def __repr__(self):
        return (
            f"(&/, {', '.join(str(c.value) for c in self.components)}) {self.desirev}!"
        )


class AgentNAL:
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
        self.sequences: dict[int, Sequence] = dict()
        self.sequence_table: dict[State, list[Sequence]] = {}
        self.goal = Concept("G")

    def select_action(self, state: State) -> int:
        """Select an action based on epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        sequences = self.sequence_table.get(state, [])
        if not sequences:
            return random.choice(self.actions)
        es = [seq.desirev.e for seq in sequences]
        idx = int(np.argmax(es))
        best_sequence = sequences[idx]
        best_action = best_sequence.components[1].value
        return best_action

    def update_q_state_action(
        self, state: State, action: int, reward: float, next_state: State
    ):
        """Update the Q-value for the given state-action pair."""
        if state == next_state:
            return  # invalid move, do not update

        seq = self._add_sequence(Sequence(Concept(state), Concept(action)))

        if reward > 0:
            seq.desirev.choose(DesireV(1.0, 0.99))  # w+ = w = 99

        # update seq.desirev
        next_seqs = self.sequence_table.get(next_state, [])
        max_desire = max(
            (seq.desirev for seq in next_seqs),
            key=lambda dv: dv.e,
            default=DesireV(0.5, 0.0),
        )
        max_desire = DesireV(max_desire.f, max_desire.c * 0.95)
        seq.desirev.choose(max_desire)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _add_sequence(self, sequence: Sequence):
        if hash(sequence) not in self.sequences:
            self.sequences[hash(sequence)] = sequence
            state: State = sequence.components[0].value
            self.sequence_table.setdefault(state, [])
            self.sequence_table[state].append(sequence)
            return sequence
        else:
            return self.sequences[hash(sequence)]

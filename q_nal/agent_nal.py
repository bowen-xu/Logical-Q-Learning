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
from weakref import WeakSet, WeakValueDictionary


from .nal import DesireV, TruthV, Desire_deduction

from .concepts import Concept, Schema, Sequence


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
        self.schemas: dict[int, Schema] = dict()
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
        schema = self._add_schema(Schema(seq, Concept(next_state)))
        # schema
        schema.truth.revise(1.0, 0.5)  # w+ = w = 1.0

        if reward > 0:
            schema_g = self._add_schema(Schema(Concept(next_state), self.goal))
            schema_g.truth.revise(1.0, 0.5)  # w+ = w = 1.0
            seq.desirev.choose(DesireV(1.0, 0.99))  # w+ = w = 99

        # update seq.desirev
        next_seqs = self.sequence_table.get(next_state, [])
        max_desire = max(
            (seq.desirev for seq in next_seqs),
            key=lambda dv: dv.e,
            default=DesireV(0.5, 0.0),
        )
        max_desire = DesireV(max_desire.f, max_desire.c * 0.95)
        desirev = Desire_deduction(schema.truth, max_desire)
        seq.desirev.choose(desirev)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _add_schema(self, schema: Schema):
        if hash(schema) not in self.schemas:
            self.schemas[hash(schema)] = schema
            return schema
        else:
            return self.schemas[hash(schema)]

    def _add_sequence(self, sequence: Sequence):
        if hash(sequence) not in self.sequences:
            self.sequences[hash(sequence)] = sequence
            state: State = sequence.components[0].value
            self.sequence_table.setdefault(state, [])
            self.sequence_table[state].append(sequence)
            return sequence
        else:
            return self.sequences[hash(sequence)]

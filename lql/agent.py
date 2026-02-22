from .nal import DesireV, TruthV, Desire_deduction, w_to_c
from .elements import Desire, Belief
from .concepts import Concept, Schema, Sequence
from .network import ConceptNetwork
import random
import numpy as np
import warnings
from copy import copy


class Agent:
    def __init__(
        self,
        actions: list[int],
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.conet = ConceptNetwork()
        self.actions = list(actions)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            return random.choice(self.actions)

        concept = self.conet.ensure_concept(state)
        sequences = concept.upper_sequences
        if not sequences:
            return random.choice(self.actions)

        best_sequence = max(
            sequences, key=lambda seq: (seq.evaluate_desire() if len(seq) >= 2 else -1)
        )
        if len(best_sequence) < 2:
            warnings.warn(
                f"No valid action sequences found for state {state}, selecting random action."
            )
            return random.choice(self.actions)

        best_action = best_sequence.components[1].value
        return best_action

    def update_q_state_action(self, last_state, last_action, reward, current_state):
        current_concept = self.conet.ensure_concept(current_state)
        last_concept = self.conet.ensure_concept(last_state)
        last_action_concept = self.conet.ensure_concept(last_action)
        last_seq = self.conet.ensure_sequence(last_concept, last_action_concept)
        schema = self.conet.ensure_schema(last_seq, current_concept)
        if schema.belief is None:
            schema.belief = Belief(TruthV(1.0, 0.999))

        # desirev_reward <-> R(s′)
        if reward > 1e-4:
            c = w_to_c(reward, 1)
            desirev_reward = DesireV(1.0, c)
        elif reward < -1e-4:
            c = w_to_c(-reward, 1)
            desirev_reward = DesireV(0.0, c)
        else:
            desirev_reward = DesireV(0.5, 0.0)

        current_concept.desire.overwrite(desirev_reward)

        # desirev_max_next_seq <-> max_{a'}{Q(s', a')}
        if not current_concept.upper_sequences:
            desirev_max_next_seq = None
        else:
            max_next_seq = max(
                current_concept.upper_sequences,
                key=lambda next_seq: next_seq.evaluate_desire(),
            )
            desirev_max_next_seq = max_next_seq.desire.desirev
        # desirev_last_seq <-> {Q(s, a)}_{obs}
        desirev_current = copy(current_concept.desire.desirev)
        if desirev_max_next_seq is not None:
            desirev_current.revise(
                desirev_max_next_seq.f, desirev_max_next_seq.c, c_max=0.9
            )

        # (s, a) =/> s', s'! |- (s, a)!
        desirev_last_seq: DesireV = copy(
            Desire_deduction(schema.belief.truthv, desirev_current)
        )
        last_seq.desire.overwrite(desirev_last_seq)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

from .grid_world import State


class Agent:
    def select_action(self, state: State) -> int: ...

    def update_q_state_action(
        self, state: State, action: int, reward: float, next_state: State
    ): ...

    def decay_epsilon(self): ...

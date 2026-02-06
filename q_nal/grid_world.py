"""
Simplified 3x3 GridWorld environment
Used for testing AgentNAL
Supports random obstacles (ensuring path exists)
"""

import random
from collections import deque


type State = tuple[int, int]


class GridWorld:
    def __init__(
        self,
        grid_size: int = 3,
        obstacle_probability: float = 0.2,
        step_reward: float = -1.0,
        goal_reward: float = 10.0,
        invalid_move_penalty: float = -5.0,
    ):
        self.grid_size = grid_size
        self.obstacle_probability = obstacle_probability
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.invalid_move_penalty = invalid_move_penalty

        # Action definitions: 0=up, 1=right, 2=down, 3=left
        self.actions: list[int] = [0, 1, 2, 3]
        self.action_to_delta: dict[int, State] = {
            0: (0, 1),  # up
            1: (1, 0),  # right
            2: (0, -1),  # down
            3: (-1, 0),  # left
        }

        # Fixed start and goal positions
        self.start: State = (0, grid_size - 1)  # bottom-left (0, 2)
        self.goal: State = (grid_size - 1, 0)  # top-right (2, 0)

        # Generate obstacles
        self.obstacles: set[State] = self._generate_obstacles()

    def _generate_obstacles(self) -> set[State]:
        """Generate random obstacles and ensure path exists from start to goal"""
        max_attempts = 1000
        for _ in range(max_attempts):
            obstacles = set()
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if (x, y) == self.start or (x, y) == self.goal:
                        continue
                    if random.random() < self.obstacle_probability:
                        obstacles.add((x, y))
            if self._exists_path(obstacles):
                return obstacles
        return set()

    def _exists_path(self, obstacles: set[State]) -> bool:
        """BFS to check if path exists from start to goal"""
        visited = {self.start}
        queue = deque([self.start])

        while queue:
            visiting_state = queue.popleft()

            # Reached goal
            if visiting_state == self.goal:
                return True

            for dx, dy in self.action_to_delta.values():
                next_x = visiting_state[0] + dx
                next_y = visiting_state[1] + dy
                next_state = (next_x, next_y)

                # Check bounds
                if not (0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size):
                    continue

                # Check obstacles
                if next_state in obstacles:
                    continue

                # Check visited
                if next_state in visited:
                    continue

                visited.add(next_state)
                queue.append(next_state)

        return False

    def reset(self) -> State:
        """Reset environment, return start state"""
        return self.start

    def state_is_valid(self, state: State) -> bool:
        """Check if state is valid (within grid and not an obstacle)"""
        x, y = state
        if x < 0 or x >= self.grid_size:
            return False
        if y < 0 or y >= self.grid_size:
            return False
        if state in self.obstacles:
            return False
        return True

    def step(self, state: State, action: int) -> tuple[State, float]:
        """
        Execute action, return (next_state, reward)

        Args:
            state: current state
            action: action (0=up, 1=right, 2=down, 3=left)

        Returns:
            (next_state, reward)
        """
        dx, dy = self.action_to_delta[action]
        next_state = (state[0] + dx, state[1] + dy)

        # Check if hit wall or obstacle
        if not self.state_is_valid(next_state):
            reward = self.step_reward + self.invalid_move_penalty
            next_state = state  # position unchanged
            return next_state, reward

        # Normal move
        reward = self.step_reward

        # Check if reached goal
        if next_state == self.goal:
            reward += self.goal_reward

        return next_state, reward

    def render(self, agent_state: State) -> str:
        """Simple text rendering"""
        grid = []
        for y in range(self.grid_size - 1, -1, -1):  # top to bottom
            row = []
            for x in range(self.grid_size):
                pos = (x, y)
                if pos == self.start:
                    row.append("S")
                elif pos == self.goal:
                    row.append("G")
                elif pos == agent_state:
                    row.append("A")
                elif pos in self.obstacles:
                    row.append("X")
                else:
                    row.append(".")
            grid.append(" ".join(row))
        return "\n".join(grid)


if __name__ == "__main__":
    # Simple test
    env = GridWorld(
        grid_size=3,
        obstacle_probability=0.2,
        step_reward=-1.0,
        goal_reward=10.0,
        invalid_move_penalty=-5.0,
    )

    print("=== Simplified 3x3 GridWorld Test (with obstacles) ===")
    print(f"Grid size: {env.grid_size}x{env.grid_size}")
    print(f"Start: {env.start}, Goal: {env.goal}")
    print(f"Obstacle probability: {env.obstacle_probability}")
    print(f"Obstacles: {env.obstacles}")
    print(f"Actions: {env.actions}")
    print()

    # Test movement
    state = env.reset()
    print(env.render(state))
    print()
    print("Legend: S=Start, G=Goal, A=Agent, X=Obstacle, .=Empty")

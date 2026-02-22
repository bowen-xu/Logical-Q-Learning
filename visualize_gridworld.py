"""
GridWorld Visualization Script
Visualizes the agent's movement in the GridWorld environment step by step.
Shows the optimal path with directional arrows at the end.
"""

import pygame
import sys
import yaml
import random
from pathlib import Path
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from lql.agent import Agent
from grid_world import GridWorld


scale_apple = 3 / 5
scale_carpet = 3 / 5
scale_stone = 0.7
scale_agent = 1.8


class GridWorldVisualizer:
    def __init__(self, env: GridWorld, agent: Agent, cell_size: int = 64):
        pygame.init()

        self.env = env
        self.agent = agent
        self.cell_size = cell_size
        self.grid_size = env.grid_size

        # Calculate window size
        self.width = self.grid_size * cell_size
        self.height = self.grid_size * cell_size

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.ORANGE = (255, 165, 0)

        # Load sprites
        self.sprites = self.load_sprites()

        # Create window
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("GridWorld Visualization")

        # Agent position
        self.agent_pos = env.start

        # Animation state
        self.animating = False
        self.animation_step = 0
        self.target_pos = None

        # Control mode
        self.show_optimal_path = False
        self.optimal_path = []

    def load_sprites(self):
        sprites = {}
        assets_dir = Path(__file__).parent / "assets"

        grass_path = assets_dir / "Grass.png"
        if grass_path.exists():
            pil_image = Image.open(grass_path)
            grass_size = (pil_image.width // 11, pil_image.height // 7)
            print(f"[DEBUG] Grass.png: {pil_image.size}, cell: {grass_size}")
            sprites["grass_top_left"] = self.pil_to_pygame(pil_image, 0, 0, grass_size)
            sprites["grass_top"] = self.pil_to_pygame(pil_image, 0, 1, grass_size)
            sprites["grass_top_right"] = self.pil_to_pygame(pil_image, 0, 2, grass_size)
            sprites["grass_left"] = self.pil_to_pygame(pil_image, 1, 0, grass_size)
            sprites["grass_center"] = self.pil_to_pygame(pil_image, 1, 1, grass_size)
            sprites["grass_right"] = self.pil_to_pygame(pil_image, 1, 2, grass_size)
            sprites["grass_bottom_left"] = self.pil_to_pygame(
                pil_image, 2, 0, grass_size
            )
            sprites["grass_bottom"] = self.pil_to_pygame(pil_image, 2, 1, grass_size)
            sprites["grass_bottom_right"] = self.pil_to_pygame(
                pil_image, 2, 2, grass_size
            )
            print(
                f"[DEBUG] Loaded {len([k for k in sprites if 'grass' in k])} grass sprites"
            )

        things_path = assets_dir / "Basic_Grass_Biom_things.png"
        if things_path.exists():
            pil_image = Image.open(things_path)
            things_size = (pil_image.width // 9, pil_image.height // 5)
            print(f"[DEBUG] Things.png: {pil_image.size}, cell: {things_size}")
            sprites["stone"] = self.pil_to_pygame(pil_image, 4, 5, things_size)
            sprites["apple"] = self.pil_to_pygame(pil_image, 2, 2, things_size)
            print(f"[DEBUG] Stone (5,6) and Apple (3,3) loaded")

        char_path = assets_dir / "Basic Charakter Spritesheet.png"
        if char_path.exists():
            pil_image = Image.open(char_path)
            char_size = (pil_image.width // 4, pil_image.height // 4)
            print(f"[DEBUG] Character.png: {pil_image.size}, cell: {char_size}")
            sprites["agent"] = self.pil_to_pygame(pil_image, 0, 0, char_size)
            print(f"[DEBUG] Agent (1,1) loaded")

        furniture_path = assets_dir / "Basic Furniture.png"
        if furniture_path.exists():
            pil_image = Image.open(furniture_path)
            furniture_size = (pil_image.width // 9, pil_image.height // 6)
            print(f"[DEBUG] Furniture.png: {pil_image.size}, cell: {furniture_size}")
            sprites["carpet"] = self.pil_to_pygame(pil_image, 5, 2, furniture_size)
            print(f"[DEBUG] Carpet (6,3) loaded")

        return sprites

    def pil_to_pygame(self, pil_image, grid_row, grid_col, cell_size):
        x = grid_col * cell_size[0]
        y = grid_row * cell_size[1]
        cropped = pil_image.crop((x, y, x + cell_size[0], y + cell_size[1]))
        if cropped.mode != "RGBA":
            cropped = cropped.convert("RGBA")
        return pygame.image.frombytes(cropped.tobytes(), cropped.size, "RGBA")

    def extract_sprite(self, sheet, col, row, total_cols, total_rows, cell_size):
        """Extract a single sprite from a sprite sheet"""
        x = col * cell_size[0]
        y = row * cell_size[1]
        return sheet.subsurface(pygame.Rect(x, y, cell_size[0], cell_size[1]))

    def draw_grid(self):
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                is_top = y == 0
                is_bottom = y == self.grid_size - 1
                is_left = x == 0
                is_right = x == self.grid_size - 1

                if is_top and is_left:
                    sprite_key = "grass_top_left"
                elif is_top and is_right:
                    sprite_key = "grass_top_right"
                elif is_bottom and is_left:
                    sprite_key = "grass_bottom_left"
                elif is_bottom and is_right:
                    sprite_key = "grass_bottom_right"
                elif is_top:
                    sprite_key = "grass_top"
                elif is_bottom:
                    sprite_key = "grass_bottom"
                elif is_left:
                    sprite_key = "grass_left"
                elif is_right:
                    sprite_key = "grass_right"
                else:
                    sprite_key = "grass_center"

                if sprite_key in self.sprites:
                    sprite = pygame.transform.scale(
                        self.sprites[sprite_key], (self.cell_size, self.cell_size)
                    )
                    self.screen.blit(sprite, rect.topleft)
                else:
                    pygame.draw.rect(self.screen, (34, 139, 34), rect)

    def draw_obstacles(self):
        if "stone" in self.sprites:
            for obs in self.env.obstacles:
                x, y = obs
                # 2/3 of cell size
                obj_size = int(self.cell_size * scale_stone)
                offset = (self.cell_size - obj_size) // 2
                sprite = pygame.transform.scale(
                    self.sprites["stone"], (obj_size, obj_size)
                )
                self.screen.blit(
                    sprite, (x * self.cell_size + offset, y * self.cell_size + offset)
                )
        else:
            # Fallback: draw gray squares
            for obs in self.env.obstacles:
                x, y = obs
                rect = pygame.Rect(
                    x * self.cell_size + 4,
                    y * self.cell_size + 4,
                    self.cell_size - 8,
                    self.cell_size - 8,
                )
                pygame.draw.rect(self.screen, (128, 128, 128), rect)

    def draw_goal(self):
        x, y = self.env.goal
        rect = pygame.Rect(
            x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size
        )

        if "apple" in self.sprites:
            # 2/3 of cell size
            obj_size = int(self.cell_size * scale_apple)
            offset = (self.cell_size - obj_size) // 2
            sprite = pygame.transform.scale(self.sprites["apple"], (obj_size, obj_size))
            self.screen.blit(
                sprite, (x * self.cell_size + offset, y * self.cell_size + offset)
            )
        else:
            # Fallback: draw green G
            font = pygame.font.Font(None, 36)
            text = font.render("G", True, self.GREEN)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)

    def draw_start(self):
        x, y = self.env.start
        rect = pygame.Rect(
            x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size
        )

        if "carpet" in self.sprites:
            # 2/3 of cell size
            obj_size = int(self.cell_size * scale_carpet)
            offset = (self.cell_size - obj_size) // 2
            sprite = pygame.transform.scale(
                self.sprites["carpet"], (obj_size, obj_size)
            )
            self.screen.blit(
                sprite, (x * self.cell_size + offset, y * self.cell_size + offset)
            )
        else:
            font = pygame.font.Font(None, 36)
            text = font.render("S", True, self.BLUE)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)

    def draw_agent(self, position=None):
        if position is None:
            position = self.agent_pos

        x, y = position

        if "agent" in self.sprites:
            # Scale by 2x, then crop center to fit cell
            scaled_size = self.cell_size * scale_agent
            scaled_sprite = pygame.transform.scale(
                self.sprites["agent"], (scaled_size, scaled_size)
            )

            # Crop center
            offset = (scaled_size - self.cell_size) // 2
            cropped = scaled_sprite.subsurface(
                pygame.Rect(offset, offset, self.cell_size, self.cell_size)
            )

            self.screen.blit(cropped, (x * self.cell_size, y * self.cell_size))
        else:
            rect = pygame.Rect(
                x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size
            )
            center = rect.center
            pygame.draw.circle(self.screen, self.RED, center, self.cell_size // 3)
            pygame.draw.circle(self.screen, self.WHITE, center, self.cell_size // 3, 2)

    def draw_arrow(self, position, direction):
        """Draw a directional arrow at the given position"""
        x, y = position
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2

        # Arrow direction mapping
        arrows = {
            0: (0, -1),  # up
            1: (1, 0),  # right
            2: (0, 1),  # down
            3: (-1, 0),  # left
        }

        dx, dy = arrows.get(direction, (0, 0))

        # Draw arrow
        arrow_size = self.cell_size // 3
        points = []

        if dx == 0 and dy == -1:  # up
            points = [
                (center_x, center_y - arrow_size),
                (center_x - arrow_size // 2, center_y),
                (center_x + arrow_size // 2, center_y),
            ]
        elif dx == 1 and dy == 0:  # right
            points = [
                (center_x + arrow_size, center_y),
                (center_x, center_y - arrow_size // 2),
                (center_x, center_y + arrow_size // 2),
            ]
        elif dx == 0 and dy == 1:  # down
            points = [
                (center_x, center_y + arrow_size),
                (center_x - arrow_size // 2, center_y),
                (center_x + arrow_size // 2, center_y),
            ]
        elif dx == -1 and dy == 0:  # left
            points = [
                (center_x - arrow_size, center_y),
                (center_x, center_y - arrow_size // 2),
                (center_x, center_y + arrow_size // 2),
            ]

        if points:
            pygame.draw.polygon(self.screen, self.YELLOW, points)

    def draw_optimal_path(self):
        """Draw the optimal path with arrows"""
        for i, (pos, action) in enumerate(self.optimal_path):
            self.draw_arrow(pos, action)

            # Draw path number
            x, y = pos
            rect = pygame.Rect(
                x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size
            )
            font = pygame.font.Font(None, 24)
            text = font.render(str(i + 1), True, self.WHITE)
            text_rect = text.get_rect(
                center=(rect.centerx, rect.centery - self.cell_size // 4)
            )
            self.screen.blit(text, text_rect)

    def compute_optimal_path(self):
        """Compute the optimal path from start to goal using the learned policy"""
        self.agent.epsilon = 0.0  # Disable exploration

        path = []
        current = self.env.start

        # Track visited to avoid infinite loops
        visited = set()
        max_steps = self.grid_size * self.grid_size * 2

        while current != self.env.goal and len(path) < max_steps:
            if current in visited:
                break
            visited.add(current)

            action = self.agent.select_action(current)
            path.append((current, action))

            # Get next state
            dx, dy = self.env.action_to_delta[action]
            next_state = (current[0] + dx, current[1] + dy)

            if not self.env.state_is_valid(next_state):
                break

            current = next_state

        self.optimal_path = path

    def run_episode(self, delay: int = 500):
        """Run one episode with visualization"""
        state = self.env.reset()
        self.agent_pos = state
        self.draw()

        pygame.display.flip()
        pygame.time.delay(delay)

        max_steps = 100

        for step in range(max_steps):
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Select action
            action = self.agent.select_action(state)

            # Execute action
            next_state, reward = self.env.step(state, action)

            # Update agent
            self.agent.update_q_state_action(state, action, reward, next_state)

            # Update position
            self.agent_pos = next_state
            self.draw()

            pygame.display.flip()
            pygame.time.delay(delay)

            state = next_state

            # Reached goal
            if state == self.env.goal:
                print(
                    f"Reached goal! Total reward: {sum([self.env.step_reward] * (step + 1))}"
                )
                break

        # Decay epsilon
        self.agent.decay_epsilon()

    def run_training(self, num_episodes: int = 100, delay: int = 100):
        """Run training episodes with visualization"""
        for episode in range(num_episodes):
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Run one episode
            state = self.env.reset()
            self.agent_pos = state
            self.draw()

            pygame.display.set_caption(
                f"GridWorld - Episode {episode + 1}/{num_episodes}"
            )
            pygame.display.flip()

            max_steps = 100
            episode_reward = 0

            for step in range(max_steps):
                # Handle events during episode
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                # Select action
                action = self.agent.select_action(state)

                # Execute action
                next_state, reward = self.env.step(state, action)

                # Update agent
                self.agent.update_q_state_action(state, action, reward, next_state)

                episode_reward += reward

                # Update position
                self.agent_pos = next_state
                self.draw()

                pygame.display.flip()
                pygame.time.delay(delay)

                state = next_state

                # Reached goal
                if state == self.env.goal:
                    break

            # Decay epsilon
            self.agent.decay_epsilon()

            if (episode + 1) % 10 == 0:
                print(
                    f"Episode {episode + 1}/{num_episodes}, Epsilon: {self.agent.epsilon:.4f}"
                )

    def show_result(self):
        self.compute_optimal_path()
        self.show_optimal_path = True
        self.draw()
        pygame.display.flip()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type in (pygame.KEYDOWN,):
                    if event.key in (
                        pygame.K_ESCAPE,
                        pygame.K_q,
                        pygame.K_SPACE,
                        pygame.K_RETURN,
                    ):
                        running = False

    def draw(self):
        self.screen.fill((255, 255, 255))
        self.draw_grid()
        self.draw_obstacles()
        self.draw_goal()
        self.draw_start()

        if self.show_optimal_path and self.optimal_path:
            self.draw_optimal_path()
        else:
            self.draw_agent()


def main():
    print("=== GridWorld Visualization ===\n")

    # Create environment
    env = GridWorld(
        grid_size=7,
        obstacle_probability=0.2,
        step_reward=-0.1,
        goal_reward=10.0,
        invalid_move_penalty=-1.0,
    )

    # Create agent
    agent = Agent(
        actions=env.actions, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999
    )

    print("Environment config:")
    print(f"  Grid: {env.grid_size}x{env.grid_size}")
    print(f"  Start: {env.start}, Goal: {env.goal}")
    print(f"  Obstacles: {env.obstacles}")
    print(f"  Step reward: {env.step_reward}")
    print(f"  Goal reward: {env.goal_reward}")
    print()

    # Create visualizer
    visualizer = GridWorldVisualizer(env, agent, cell_size=80)

    # Run training
    print("Training agent (this will show a window, close it to continue)...")
    visualizer.run_training(num_episodes=20, delay=30)

    print("\nShowing optimal path...")
    print("Close the window or press ESC/q/Space to exit")
    visualizer.show_result()

    pygame.quit()
    print("Done!")


if __name__ == "__main__":
    main()

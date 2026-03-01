"""
GridWorld Visualization Script
Visualizes the agent's movement in the GridWorld environment step by step.
Shows the optimal path with directional arrows at the end.
"""

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import sys
import yaml
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from plot_rewards import plot_rewards


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from lql.agent import Agent
from grid_world import GridWorld


scale_apple = 3 / 5
scale_carpet = 3 / 5
scale_stone = 0.7
scale_agent = 1.8

obstacle_probability = 0.35
seed = 20260228

rewards_path = None

RECORDING_DIR = Path("recordings/gridworld_lql")

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

    def grid_to_pygame(self, pos):
        """Convert grid coordinates (origin at bottom-left) to pygame coordinates (origin at top-left)"""
        x, y = pos
        return (x, self.grid_size - 1 - y)

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
        for grid_y in range(self.grid_size):
            for x in range(self.grid_size):
                # Convert grid coords to pygame coords
                pygame_y = self.grid_size - 1 - grid_y
                rect = pygame.Rect(
                    x * self.cell_size,
                    pygame_y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                is_top = grid_y == self.grid_size - 1  # top in grid = max y
                is_bottom = grid_y == 0  # bottom in grid = y=0
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
                x, y = self.grid_to_pygame(obs)
                # 2/3 of cell size
                obj_size = int(self.cell_size * 2 / 3)
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
                x, y = self.grid_to_pygame(obs)
                rect = pygame.Rect(
                    x * self.cell_size + 4,
                    y * self.cell_size + 4,
                    self.cell_size - 8,
                    self.cell_size - 8,
                )
                pygame.draw.rect(self.screen, (128, 128, 128), rect)

    def draw_goal(self):
        x, y = self.grid_to_pygame(self.env.goal)
        rect = pygame.Rect(
            x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size
        )

        if "apple" in self.sprites:
            # 2/3 of cell size
            obj_size = int(self.cell_size * 2 / 3)
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
        x, y = self.grid_to_pygame(self.env.start)
        rect = pygame.Rect(
            x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size
        )

        if "carpet" in self.sprites:
            # 2/3 of cell size
            obj_size = int(self.cell_size * 2 / 3)
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

        x, y = self.grid_to_pygame(position)

        if "agent" in self.sprites:
            # Scale by 2x, then crop center to fit cell
            scaled_size = self.cell_size * 2
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

    def draw_arrow(self, position, direction, alpha=255):
        import math

        x, y = self.grid_to_pygame(position)
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2

        arrows = {
            0: (0, 1),
            1: (1, 0),
            2: (0, -1),
            3: (-1, 0),
        }

        dx, dy = arrows.get(direction, (0, 0))

        # Equilateral triangle height
        h = self.cell_size // 3
        # Side length for equilateral triangle: s = 2 * h / sqrt(3)
        s = 2 * h / math.sqrt(3)

        # Distance from centroid to tip = 2h/3
        # Distance from centroid to base midpoint = h/3
        tip_dist = 2 * h / 3
        base_dist = h / 3

        tip = base_left = base_right = None

        if dx == 0 and dy == 1:  # up
            tip = (center_x, center_y - tip_dist)
            base_center = (center_x, center_y + base_dist)
            base_left = (base_center[0] - s / 2, base_center[1])
            base_right = (base_center[0] + s / 2, base_center[1])
        elif dx == 1 and dy == 0:  # right
            tip = (center_x + tip_dist, center_y)
            base_center = (center_x - base_dist, center_y)
            base_left = (base_center[0], base_center[1] - s / 2)
            base_right = (base_center[0], base_center[1] + s / 2)
        elif dx == 0 and dy == -1:  # down
            tip = (center_x, center_y + tip_dist)
            base_center = (center_x, center_y - base_dist)
            base_left = (base_center[0] - s / 2, base_center[1])
            base_right = (base_center[0] + s / 2, base_center[1])
        elif dx == -1 and dy == 0:  # left
            tip = (center_x - tip_dist, center_y)
            base_center = (center_x + base_dist, center_y)
            base_left = (base_center[0], base_center[1] - s / 2)
            base_right = (base_center[0], base_center[1] + s / 2)

        if tip is not None:
            points = [tip, base_left, base_right]

            if alpha < 255:
                surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                color = (255, 0, 0, alpha)
                pygame.draw.polygon(
                    surf,
                    color,
                    [
                        (p[0] - x * self.cell_size, p[1] - y * self.cell_size)
                        for p in points
                    ],
                )
                self.screen.blit(surf, (x * self.cell_size, y * self.cell_size))
            else:
                pygame.draw.polygon(self.screen, (255, 0, 0), points)

    def draw_optimal_path(self):
        # Compute optimal path if not already done
        if not self.optimal_path:
            self.compute_optimal_path()

        # Get set of optimal path states
        optimal_states = set(pos for pos, _ in self.optimal_path)

        # First, draw all non-optimal arrows with 0.5 alpha
        for grid_y in range(self.grid_size):
            for x in range(self.grid_size):
                state = (x, grid_y)
                if state == self.env.start:
                    continue
                if state == self.env.goal:
                    continue
                if state in self.env.obstacles:
                    continue
                if not self.env.state_is_valid(state):
                    continue

                action = self.agent.select_action(state)
                if state in optimal_states:
                    # Optimal path: opaque
                    self.draw_arrow(state, action, alpha=255)
                else:
                    # Non-optimal: 0.5 alpha
                    self.draw_arrow(state, action, alpha=128)

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

    def run_training(
        self,
        num_episodes: int = 100,
        delay: int = 100,
        record_interval: int = 0,
        output_dir: str = ".",
    ):
        global rewards_path
        import os
        import pickle
        from PIL import Image

        os.makedirs(output_dir, exist_ok=True)

        rewards_history = []
        self.total_reward = 0.0

        is_recording = record_interval > 0

        for episode in tqdm(range(num_episodes), desc="Training"):
            should_record = is_recording and (
                episode % record_interval == 0 or episode == num_episodes - 1
            )

            state = self.env.reset()
            self.agent_pos = state

            max_steps = 100
            episode_reward = 0.0

            for step in range(max_steps):
                action = self.agent.select_action(state)
                next_state, reward = self.env.step(state, action)
                self.agent.update_q_state_action(state, action, reward, next_state)
                self.agent_pos = next_state

                episode_reward += reward
                state = next_state

                if state == self.env.goal:
                    break

            self.agent.decay_epsilon()
            rewards_history.append(episode_reward)
            self.total_reward = episode_reward

            if should_record:
                if not pygame.display.get_init():
                    pygame.display.init()

                episode_frames = []
                state = self.env.reset()
                self.agent_pos = state
                self.draw()
                pygame.display.flip()

                for step in range(max_steps):
                    action = self.agent.select_action(state)
                    next_state, reward = self.env.step(state, action)
                    self.agent_pos = next_state
                    self.draw()
                    pygame.display.flip()

                    img_str = pygame.image.tostring(self.screen, "RGB")
                    img = Image.frombytes("RGB", self.screen.get_size(), img_str)
                    episode_frames.append(img)

                    state = next_state
                    if state == self.env.goal:
                        break

                gif_path = os.path.join(output_dir, f"episode_{episode:04d}.gif")
                episode_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=episode_frames[1:],
                    duration=delay,
                    loop=0,
                )
                tqdm.write(f"Recorded: {gif_path}")

                # Save conet sequences desirev with term_str as key
                desirev_data = {}
                for seq in self.agent.conet.sequences.values():
                    term = seq.term_str()
                    desirev = seq.desire.desirev if seq.desire else None
                    if desirev:
                        desirev_data[term] = {
                            "f": desirev.f,
                            "c": desirev.c,
                        }

                pickle_path = os.path.join(output_dir, f"episode_{episode:04d}.pkl")
                with open(pickle_path, "wb") as f:
                    pickle.dump(desirev_data, f)
                tqdm.write(f"Saved: {pickle_path}")

        # Save rewards history
        rewards_path = os.path.join(output_dir, "rewards.pkl")
        with open(rewards_path, "wb") as f:
            pickle.dump(rewards_history, f)
        print(f"Saved rewards: {rewards_path}")

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

    def record_final(
        self, output_path: str = "final.png", trajectory_path: str = "trajectory.png"
    ):
        from PIL import Image

        self.agent.epsilon = 0.0
        self.compute_optimal_path()
        self.show_optimal_path = True

        # Draw the final state with arrows
        self.draw()
        pygame.display.flip()

        # Save as PNG
        img_str = pygame.image.tostring(self.screen, "RGB")
        img = Image.frombytes("RGB", self.screen.get_size(), img_str)

        # Ensure output_path ends with .png
        if not output_path.endswith(".png"):
            output_path = output_path.rsplit(".", 1)[0] + ".png"

        img.save(output_path)
        print(f"Saved final result: {output_path}")

        # Draw trajectory image (without arrows)
        self.show_optimal_path = False
        self.draw()

        # Draw agents along the optimal path with fading alpha
        path_len = len(self.optimal_path)
        if path_len > 0:
            for i, (pos, action) in enumerate(self.optimal_path):
                # Skip start and end
                if pos == self.env.start or pos == self.env.goal:
                    continue
                # Alpha from 0.1 to 1.0
                alpha = int(0.1 * 255 + (0.9 * i / (path_len - 1)) * 255)
                self.draw_agent_at_position(pos, alpha)

        pygame.display.flip()

        img_str = pygame.image.tostring(self.screen, "RGB")
        img = Image.frombytes("RGB", self.screen.get_size(), img_str)

        if not trajectory_path.endswith(".png"):
            trajectory_path = trajectory_path.rsplit(".", 1)[0] + ".png"

        img.save(trajectory_path)
        print(f"Saved trajectory: {trajectory_path}")

    def draw_agent_at_position(self, position, alpha=255):
        x, y = self.grid_to_pygame(position)

        if "agent" in self.sprites:
            scaled_size = self.cell_size * 2
            scaled_sprite = pygame.transform.scale(
                self.sprites["agent"], (scaled_size, scaled_size)
            )
            offset = (scaled_size - self.cell_size) // 2
            cropped = scaled_sprite.subsurface(
                pygame.Rect(offset, offset, self.cell_size, self.cell_size)
            )

            if alpha < 255:
                surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                orig_alpha = cropped.get_alpha()
                cropped.set_alpha(alpha)
                surf.blit(cropped, (0, 0))
                self.screen.blit(surf, (x * self.cell_size, y * self.cell_size))
            else:
                self.screen.blit(cropped, (x * self.cell_size, y * self.cell_size))

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
    import random

    random.seed(seed)

    print("=== GridWorld Visualization ===\n")

    # Create environment
    env = GridWorld(
        grid_size=7,
        obstacle_probability=obstacle_probability,
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

    # Run training with recording
    print("Training agent with GIF recording...")
    print("Recording at episode 0, 500, 1000, 1500, 2000, 2500, 3000")
    visualizer.run_training(
        num_episodes=3000, delay=50, record_interval=500, output_dir=RECORDING_DIR
    )

    # Record final result with epsilon=0
    print("\nRecording final result...")
    visualizer.record_final(
        output_path=str(RECORDING_DIR / "final.png"), trajectory_path=str(RECORDING_DIR / "trajectory.png")
    )

    # Plot rewards
    if rewards_path is not None:
        plot_rewards(rewards_path, smooth_window=50, output_path=str(RECORDING_DIR / "rewards.png"))

    pygame.quit()
    print(f"\nDone! All recordings saved to '{RECORDING_DIR}' directory")


if __name__ == "__main__":
    main()

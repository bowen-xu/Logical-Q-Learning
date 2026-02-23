import pygame
from pathlib import Path
from PIL import Image
import sys

pygame.init()

assets_dir = Path("assets")


def pil_to_pygame(pil_image, grid_row, grid_col, cell_size):
    x = grid_col * cell_size[0]
    y = grid_row * cell_size[1]
    cropped = pil_image.crop((x, y, x + cell_size[0], y + cell_size[1]))
    if cropped.mode != "RGBA":
        cropped = cropped.convert("RGBA")
    return pygame.image.frombytes(cropped.tobytes(), cropped.size, "RGBA")


grass_path = assets_dir / "Grass.png"
pil_image = Image.open(grass_path)
grass_size = (pil_image.width // 11, pil_image.height // 7)
print(f"Grass.png: {pil_image.size}, cell: {grass_size}")

grass_sprites = {}
positions = {
    "top_left": (0, 0),
    "top": (0, 1),
    "top_right": (0, 2),
    "left": (1, 0),
    "center": (1, 1),
    "right": (1, 2),
    "bottom_left": (2, 0),
    "bottom": (2, 1),
    "bottom_right": (2, 2),
}
for name, (row, col) in positions.items():
    grass_sprites[name] = pil_to_pygame(pil_image, row, col, grass_size)
    print(f"  {name}: loaded")

stone_path = assets_dir / "Basic_Grass_Biom_things.png"
pil_image = Image.open(stone_path)
sheet_size = (pil_image.width // 9, pil_image.height // 5)
print(f"\nBasic_Grass_Biom_things.png: {pil_image.size}, cell: {sheet_size}")

apple = pil_to_pygame(pil_image, 2, 2, sheet_size)
stone = pil_to_pygame(pil_image, 4, 5, sheet_size)
print(f"  Apple (3,3): loaded")
print(f"  Stone (5,6): loaded")

char_path = assets_dir / "Basic Charakter Spritesheet.png"
pil_image = Image.open(char_path)
char_sheet_size = (pil_image.width // 4, pil_image.height // 4)
print(f"\nBasic Charakter Spritesheet.png: {pil_image.size}, cell: {char_sheet_size}")

agent = pil_to_pygame(pil_image, 0, 0, char_sheet_size)
print(f"  Agent (1,1): loaded")

furniture_path = assets_dir / "Basic Furniture.png"
pil_image = Image.open(furniture_path)
furniture_size = (pil_image.width // 9, pil_image.height // 6)
print(f"\nBasic Furniture.png: {pil_image.size}, cell: {furniture_size}")

carpet = pil_to_pygame(pil_image, 5, 2, furniture_size)  # row 6, col 3 = row 5, col 2
print(f"  Carpet (6,3): loaded")

cell_size = 64
screen = pygame.display.set_mode((cell_size * 4, cell_size * 4))
pygame.display.set_caption("Sprite Debug - Close window to exit")

clock = pygame.time.Clock()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    screen.fill((40, 40, 40))

    grass_grid = [
        ["top_left", "top", "top_right"],
        ["left", "center", "right"],
        ["bottom_left", "bottom", "bottom_right"],
    ]
    for y, row in enumerate(grass_grid):
        for x, name in enumerate(row):
            sprite = pygame.transform.scale(grass_sprites[name], (cell_size, cell_size))
            screen.blit(sprite, (x * cell_size, y * cell_size))

    items = [("Stone", stone), ("Apple", apple), ("Agent", agent), ("Carpet", carpet)]
    for y, (name, sprite) in enumerate(items):
        screen.blit(
            pygame.transform.scale(sprite, (cell_size, cell_size)),
            (3 * cell_size, y * cell_size),
        )
        font = pygame.font.Font(None, 20)
        text = font.render(name, True, (255, 255, 0))
        screen.blit(text, (3 * cell_size + 4, y * cell_size + 4))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
print("\nDone")
sys.exit()

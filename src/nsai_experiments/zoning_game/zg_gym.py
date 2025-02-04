from enum import Enum
import logging
import io

import numpy as np

import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Tile(Enum):
    EMPTY = 0
    RESIDENTIAL = 1
    COMMERCIAL = 2
    INDUSTRIAL = 3
    DOWNTOWN = 4
    PARK = 5

DEFAULT_OCCURRENCES = {
    Tile.RESIDENTIAL: 14/36,
    Tile.COMMERCIAL: 6/36,
    Tile.INDUSTRIAL: 7/36,
    Tile.DOWNTOWN: 5/36,
    Tile.PARK: 4/36
}

class ZoningGameEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, grid_size = 6, initially_filled_frac = 0.4, occurrences = DEFAULT_OCCURRENCES, render_mode = "ansi"):
        """
        Arguments (all have sensible defaults):
          `grid_size = 6`: side length of the square grid
          `initially_filled_frac = 0.4`: fraction of cells that start the game filled
          `occurrences = DEFAULT_OCCURRENCES`: dict from Tile to how commonly that tile occurs; will be normalized
          `render_mode = "ansi"`: gymnasium render mode: `"ansi"` returns a string-like output, `None` does no rendering
        """
        self.grid_size = grid_size
        self.initially_filled_frac = initially_filled_frac
        self.occurrences = {k: v/sum(occurrences.values()) for k, v in occurrences.items()}
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.tile_grid, self.tile_queue = None, None
        self.observation_space = spaces.MultiDiscrete([[len(Tile)]*self.grid_size]*self.grid_size)
        self.action_space = spaces.Discrete(self.grid_size*self.grid_size)

    def reset(self, seed = None):
        super().reset(seed=seed)
        self.tile_grid, self.tile_queue = self._generate_problem()
        logger.debug(f"tile_grid:\n{self.tile_grid}")
        logger.debug(f"tile_queue:\n{self.tile_queue}")

    def _generate_problem(self):
        "Create and return random `tile_grid` and `tile_queue` given instance config"
        tile_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        tile_options = list(self.occurrences.keys())  # Tile enum instances
        tile_values = [tile.value for tile in tile_options]  # ints
        tile_p = list(self.occurrences.values())

        n_filled = 0
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.np_random.random() < self.initially_filled_frac:
                    selected_tile = self.np_random.choice(tile_values, p=tile_p)
                    logger.debug(f"Filling ({row}, {col}) with {Tile(selected_tile)}")
                    tile_grid[row, col] = selected_tile
                    n_filled += 1
        
        n_unfilled = self.grid_size*self.grid_size - n_filled
        tile_queue = self.np_random.choice(tile_values, p=tile_p, size=n_unfilled)
        return tile_grid, tile_queue
    
    def render(self):
        "Render given `self.render_mode`. For `render_mode=ansi`, can print the results like `print(my_env.render().read())`."
        if self.render_mode is None: return
        assert self.render_mode == "ansi"
        buf = io.StringIO()
        print(f"Tile grid:\n{self.tile_grid}", file=buf)
        print(f"Tile queue (leftmost next): {self.tile_queue}", file=buf)
        print(f"where {', '.join([f'{x.value} = {x.name}' for x in Tile])}", file=buf)
        buf.seek(0)
        return buf

gym.register(
    id="zg/ZoningGameEnv-v0",
    entry_point=ZoningGameEnv
)

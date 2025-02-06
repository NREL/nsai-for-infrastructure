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

def orthogonal_neighbors(padded_grid, my_row, my_col):
    "Given a zero-padded tile grid and the row and column in unpadded coordinates of a particular tile, return a list of orthogonal (non-diagonal) neighbors"
    return padded_grid[[my_row, my_row+2, my_row+1, my_row+1], [my_col+1, my_col+1, my_col+2, my_col]]  # north, south, east, west

def neighbor_score(padded_grid, my_row, my_col, neighbor_spec):
    "Calculate a score based on a weighted count of neighbors with weights specified in neighbor_spec"
    score = 0
    neighbors = orthogonal_neighbors(padded_grid, my_row, my_col)
    for key, weight in neighbor_spec:
        score += weight * sum(neighbors == key.value)
    return score

def eval_tile_indiv_score(padded_grid, my_row, my_col):
    "Given a padded tile grid and the row and column in unpadded coordinates the of a particular tile, evaluate how well the grid satisfies that tile's objectives."
    # TODO could use some more testing
    my_tile = Tile(padded_grid[my_row+1, my_col+1])
    match my_tile:
        case Tile.EMPTY:
            # Rules for EMPTY: no objectives, score is always zero
            return 0
        case Tile.RESIDENTIAL:
            # Rules for RESIDENTIAL: +1 for adjacent RESIDENTIAL, +2 for adjacent PARK, -3 for adjacent INDUSTRIAL
            return neighbor_score(padded_grid, my_row, my_col, [(Tile.RESIDENTIAL, +1), (Tile.PARK, +2), (Tile.INDUSTRIAL, -3)])
        case Tile.COMMERCIAL:
            # Rules for COMMERCIAL: +1 for adjacent RESIDENTIAL, +4 for adjacent DOWNTOWN
            return neighbor_score(padded_grid, my_row, my_col, [(Tile.RESIDENTIAL, +1), (Tile.DOWNTOWN, +4)])
        case Tile.INDUSTRIAL:
            # Rules for INDUSTRIAL: +1 for being within grid_size/6 of either the x-center line or the y-center line of the board (suppose there are railroads there)
            dx2 = (my_row*2 - (padded_grid.shape[0]-3))**2
            dy2 = (my_col*2 - (padded_grid.shape[1]-3))**2
            distance_criterion = min(dx2, dy2) * 3**2 * 4 <= (sum(padded_grid.shape) - 4)**2
            return 1*distance_criterion
        case Tile.DOWNTOWN:
            # Rules for DOWNTOWN: +2 for being within (grid_size/6) Euclidean distance of the center of the grid, +4 for adjacent DOWNTOWN, -2 for adjacent INDUSTRIAL
            dx2 = (my_row*2 - (padded_grid.shape[0]-3))**2
            dy2 = (my_col*2 - (padded_grid.shape[1]-3))**2
            distance_criterion = (dx2 + dy2) * 3**2 * 4 <= (sum(padded_grid.shape) - 4)**2
            return 2*distance_criterion + neighbor_score(padded_grid, my_row, my_col, [(Tile.DOWNTOWN, +4), (Tile.INDUSTRIAL, -2)])
        case Tile.PARK:
            # Rules for PARK: +1 for adjacent RESIDENTIAL, +3 for adjacent DOWNTOWN
            return neighbor_score(padded_grid, my_row, my_col, [(Tile.DOWNTOWN, +4), (Tile.INDUSTRIAL, -2)])
        case other:
            raise ValueError(f"Invalid tile: {other}")

def pad_grid(unpadded_grid):
    "Pad the grid with a one-width border of Tile.EMPTY"
    return np.pad(unpadded_grid, 1, mode="constant", constant_values=Tile.EMPTY.value)

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
        for row, col in np.ndindex(tile_grid.shape):
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
        print(f"where {', '.join([f'{x.value} = {x.name}' for x in Tile])}.", file=buf)
        print(f"Current grid score is {self._eval_tile_grid_score()}.", file=buf)
        buf.seek(0)
        return buf
    
    def _eval_tile_grid_score(self):
        "Use `eval_tile_indiv_score` to compute the sum score across the whole tile grid"
        padded_grid = pad_grid(self.tile_grid)
        total_score = 0
        print(type(self.tile_grid))
        for row, col in np.ndindex(self.tile_grid.shape):
            score_incr = eval_tile_indiv_score(padded_grid, row, col)
            total_score += score_incr
            current_tile = Tile(self.tile_grid[row, col])
            if current_tile is not Tile.EMPTY:
                logger.info(f"Adding {score_incr} to score for tile {current_tile.name} at {(row, col)}")
        return total_score

gym.register(
    id="zg/ZoningGameEnv-v0",
    entry_point=ZoningGameEnv
)

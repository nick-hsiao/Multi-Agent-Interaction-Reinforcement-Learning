import numpy as np
from environment.modules.module import EnvModule
from gym.envs.classic_control import rendering

class Wall:
    def __init__(self, pt1, pt2):
        assert pt1[0] == pt2[0] or pt1[1] == pt2[1]

        # If x coordinates are the same it is vertical wall
        self.is_vertical = pt1[0] == pt2[0]

        # Make sure pt2 is the top/right point
        if np.any(np.array(pt2) - np.array(pt1) < 0):
            self.pt1 = np.array(pt2)
            self.pt2 = np.array(pt1)
        else:
            self.pt1 = np.array(pt1)
            self.pt2 = np.array(pt2)


        # Length of wall.
        if self.is_vertical:
            self.length = pt2[1] - pt1[1]
        else:
            self.length = pt2[0] - pt1[0]

    def is_touching(self, pt):
        if self.is_vertical:
            return pt[0] == self.pt1[0] and pt[1] >= self.pt1[1] and pt[1] <= self.pt2[1]
        else:
            return pt[1] == self.pt1[1] and pt[0] >= self.pt1[0] and pt[0] <= self.pt2[0]
    
def add_walls_to_grid(grid, walls):
    '''
        Draw walls onto a grid.
        Args:
            grid (np.ndarray): 2D occupancy grid
            walls (Wall list): walls
    '''
    for wall in walls:
        if wall.is_vertical:
            grid[wall.pt1[0], wall.pt1[1]:wall.pt2[1] + 1] = 1
        else:
            grid[wall.pt1[0]:wall.pt2[0] + 1, wall.pt1[1]] = 1

def build_outside_walls(grid_size):
    """ 
        Returns list of four walls for perimeter of grid.
        Args:
            grid_size (int): Size of grid to fill.
     """
    return [Wall([0, 0], [0, grid_size - 1]),
            Wall([0, 0], [grid_size - 1, 0]),
            Wall([grid_size - 1, 0], [grid_size - 1, grid_size - 1]),
            Wall([0, grid_size - 1], [grid_size - 1, grid_size - 1])]


class Walls(EnvModule):
    """ Currently only supports empty (only outside walls)"""

    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.walls = []
    
    def build_world_step(self, world):
        self.walls.extend(build_outside_walls(self.grid_size))
        add_walls_to_grid(world.placement_grid, self.walls)
        return True

    def observation_step(self, world):
        return {}

    def build_render(self, viewer, block_size):
        l = 0
        b = 0
        r = block_size
        t = block_size

        
        for wall in self.walls:
            l = wall.pt1[0] * block_size
            r = l + block_size
            b = wall.pt1[1] * block_size
            t = b + block_size

            if wall.is_vertical:
                for _ in range(wall.pt2[1] - wall.pt1[1] + 1):
                    current_block = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                    viewer.add_geom(current_block)
                    b += block_size
                    t += block_size
            else:
                for _ in range(wall.pt2[0] - wall.pt1[0] + 1):
                    current_block = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                    viewer.add_geom(current_block)
                    l += block_size
                    r += block_size

        return True



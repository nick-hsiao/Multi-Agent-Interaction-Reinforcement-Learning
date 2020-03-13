import numpy as np
from environment.modules.module import EnvModule
from environment.modules.util import placement_fn
from gym.envs.classic_control import rendering

class Flag(EnvModule):

    def __init__(self, flag_size=2):
        self.flag_size = flag_size
        self.pos = []

    def build_world_step(self, world):
        self.initial_pos = placement_fn(world.grid_size, world.placement_grid, obj_size=(self.flag_size, self.flag_size))

        for i in range(self.flag_size):
            for j in range(self.flag_size):
                world.placement_grid[self.initial_pos[0] + i][self.initial_pos[1]+ j] = 1
                self.pos.append((self.initial_pos[0] + i, self.initial_pos[1]+ j))
        
        return True

    def build_render(self, viewer, block_size):
        l = self.initial_pos[0] * block_size
        r = l + (block_size * self.flag_size)
        b = self.initial_pos[1] * block_size
        t = b + (block_size * self.flag_size)

        current_block = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        current_block.set_color(0, 255, 0)
        viewer.add_geom(current_block)

        return True

    def observation_step(self, world):
        obs = {
            'flag_pos': []
        }

        obs['flag_pos'] = self.pos

        return obs


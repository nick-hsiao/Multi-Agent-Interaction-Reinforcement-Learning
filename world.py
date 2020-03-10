import numpy as np
from modules.walls import Walls
from modules.agents import Agents
from modules.flag import Flag

class World():
    def __init__(self, grid_size=32, horizon=250, n_agents=1, flag_size=2):

        self.grid_size = grid_size
        self.horizon = horizon

        self.placement_grid = np.zeros((self.grid_size, self.grid_size))

        self.walls = Walls(self.grid_size)
        self.agents = Agents(n_agents=n_agents, grid_size=self.grid_size, colors=[(0,0,255)])
        self.flag = Flag(flag_size=flag_size)

        self.modules = [self.walls, self.agents, self.flag]

    def get_observation(self):
        obs = {}

        for module in self.modules:
            obs.update(module.observation_step(self))

        return obs

    # Instantiate environement.
    def get_world(self):

        for module in self.modules:
            module.build_world_step(self)

        return self.placement_grid

    # Asks each module to render it's components.
    def get_render(self, viewer):
        block_size = viewer.width // self.grid_size
        for module in self.modules:
            module.build_render(viewer, block_size)

        return viewer

    def set_action(self, action):        
        for index, agent in enumerate(self.agents.agents):
            curr_action = action[index]
            next_position = (agent.pos[0] + curr_action[0], agent.pos[1] + curr_action[1])

            if(next_position in self.flag.pos or self.placement_grid[next_position[0]][next_position[1]] == 0):
                self.placement_grid[agent.pos[0]][agent.pos[1]] = 0
                agent.move(curr_action)
                self.placement_grid[agent.pos[0]][agent.pos[1]] = 1


    def reset(self):
        self.get_world()
        self.state = self.get_observation()
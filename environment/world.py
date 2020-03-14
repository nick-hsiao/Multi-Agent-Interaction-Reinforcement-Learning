import numpy as np
from environment.modules.walls import Walls
from environment.modules.agents import Agents
from environment.modules.flag import Flag

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
        '''
            Each agents has its own observation. 
            The observation is a placement grid but only obstacles that are in that agents line of sight are filled in.
            Line of sight is defined as a box around the agent of radius grid_size // 8
        '''
        obs = {}

        for module in self.modules:
            obs.update(module.observation_step(self))

        agent_obs = []

        for agent_pos in obs['agent_pos']:
            curr = np.zeros((self.grid_size, self.grid_size))
            self.get_agent_obs(agent_pos, curr)
            agent_obs.append(curr)

        obs['agent_obs'] = agent_obs

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
        return self.get_observation()

    
    def get_agent_obs(self, pos, grid):
        radius = self.grid_size // 8
        for x in range((pos[0] - radius), (pos[0] + radius + 1)):
            for y in range((pos[1] - radius), (pos[1] + radius + 1)):
                if x >= 0 and x < self.grid_size and y >= 0 and y < self.grid_size:
                    grid[x][y] = self.placement_grid[x][y]
                
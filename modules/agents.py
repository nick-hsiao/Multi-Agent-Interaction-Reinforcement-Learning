import numpy as np
from modules.module import EnvModule
from modules.util import placement_fn
from gym.envs.classic_control import rendering


class Agent:

    def __init__(self, pos, color):
        self.pos = pos
        self.color = color
        self.size = 0
        self.render = None
        self.translation = None
    
    def move(self, vec):
        self.pos[0] += vec[0]
        self.pos[1] += vec[1]

        if self.translation is not None:
            self.translation.set_translation(self.pos[0] * self.size, self.pos[1] * self.size)


class Agents(EnvModule):
    
    def __init__(self, n_agents, grid_size, colors=None):
        self.n_agents = n_agents
        self.colors = colors
        self.agents = []

    def build_world_step(self, world):
        for i in range(self.n_agents):
            pos = placement_fn(world.grid_size, world.placement_grid, obj_size=(1, 1))

            color = (self.colors[i]
                          if isinstance(self.colors[0], (list, tuple, np.ndarray))
                          else self.colors)

            agent = Agent(pos, color)
            self.agents.append(agent)
            world.placement_grid[pos[0]][pos[1]] = 1
    
    def build_render(self, viewer, block_size):
        for agent in self.agents:
            l,r,t,b = 0, block_size, block_size, 0
            agent.size = block_size
            agent.render = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            agent.render.set_color(agent.color[0], agent.color[1], agent.color[2])

            agent.translation = rendering.Transform()
            agent.render.add_attr(agent.translation)
            agent.translation.set_translation(agent.pos[0] * block_size, agent.pos[1] * block_size)

            viewer.add_geom(agent.render)
    
    def observation_step(self, world):
        obs = {}
        obs['agent_pos'] = []
        for agent in self.agents:
            obs['agent_pos'].append(tuple(agent.pos))

        return obs
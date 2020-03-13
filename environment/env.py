import gym
import numpy as np
from environment.world import World
from gym import spaces
from gym.envs.classic_control import rendering

class Env(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, world=World(), horizon=250, screen_size=512):
        self.vec_actions = {
            0 : (0, 0),
            1: (0, 1),
            2: (0, -1),
            3: (-1, 0),
            4: (1, 0)
        }
        self.horizon = horizon
        self.screen_size = screen_size

        self.world = world
        self.viewer = None
        self.action_space = spaces.Discrete(5)

    def step(self, action):
        actions = []
        for agent_action in action:
            assert self.action_space.contains(agent_action)

            vec_action = self.vec_actions[agent_action]
            actions.append(vec_action)

        self.world.set_action(actions)
        obs = self.world.get_observation()
        reward = 0
        done = False
        

        for pos in obs['agent_pos']:
            if(pos in obs['flag_pos']):
                done = True

        return obs, reward, done, None

    def render(self, mode='human'):
        if mode == 'human':
            if self.viewer is None:
                self.viewer = rendering.Viewer(self.screen_size, self.screen_size)
            
                self.world.get_render(self.viewer)
            self.viewer.render()
        else:
            raise ValueError("Unsupported mode.")

    def reset(self):
        self.world.reset()
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

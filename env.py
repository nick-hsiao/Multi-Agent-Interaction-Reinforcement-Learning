import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

class CTFEnv(py_environment.PyEnvironment):
    def __init__(self, grid_size=16, initial_agent_pos=(0, 0), screen_size=512):
        self.grid_size = grid_size
        self.placement_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.initial_agent_pos = initial_agent_pos
        self.flag_pos = self.get_flag_pos()

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4,), dtype=np.int32, minimum=[0,0,0,0],maximum=[grid_size-1,grid_size-1,grid_size-1,grid_size-1], name='observation')
        
        self._state=[
            initial_agent_pos[0],
            initial_agent_pos[1],
            self.flag_pos[0],
            self.flag_pos[1]
        ] #represent the (row, col, frow, fcol) of the player and the finish
        
        self._episode_ended = False

        self.screen_size = screen_size
        self.block_size = screen_size // grid_size
        self.step_count = 0
        
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        self.step_count = 0
        self.flag_pos = self.get_flag_pos()
        self.reset_grid()
        self._state=[self.initial_agent_pos[0],self.initial_agent_pos[1],
                        self.flag_pos[0],self.flag_pos[1]] 
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):
        self.step_count += 1
        if self._episode_ended:
            return self.reset()

        self.move(action)

        if self.game_over():
            self._episode_ended = True

        if self._episode_ended:
            if self.game_over():
                reward = 100 - self.step_count
            else:
                reward = 0
            return ts.termination(np.array(self._state, dtype=np.int32), reward)
        else:
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=0, discount=0.9)
    
    def move(self, action):
        row, col = self._state[0],self._state[1]
        self.placement_grid[row, col] = 0
        if action == 0: #down
            if row - 1 >= 0:
                self._state[0] -= 1
        if action == 1: #up
            if row + 1 < self.grid_size:
                self._state[0] += 1
        if action == 2: #left
            if col - 1 >= 0:
                self._state[1] -= 1
        if action == 3: #right
            if col + 1  < self.grid_size:
                self._state[1] += 1
        self.placement_grid[self._state[0], self._state[1]] = 1
    
    def game_over(self):
        row, col, frow, fcol = self._state[0],self._state[1],self._state[2],self._state[3]
        return row==frow and col==fcol
    
    def reset_grid(self):
        self.placement_grid[self._state[0]][self._state[1]] = 0
        self.placement_grid[self._state[2]][self._state[3]] = 0
        self.placement_grid[self.initial_agent_pos[0]][self.initial_agent_pos[1]] = 1
        self.placement_grid[self.flag_pos[0]][self.flag_pos[1]] = 2
    
    def render(self, mode='rgb_array'):
        frame = np.full((self.screen_size, self.screen_size, 3), 255, dtype=np.uint8)
        for ix,iy in np.ndindex(self.placement_grid.shape):
            current = self.placement_grid[ix, iy]            
            if(current == 1):
                self.fill_color(frame, ix, iy, [0, 0, 255])
            elif(current == 2):
                self.fill_color(frame, ix, iy, [0, 255, 0])
        return np.flipud(frame)
    
    def fill_color(self, frame, x, y, color):
        l = x * self.block_size
        r = l + self.block_size
        b = y * self.block_size
        t = b + self.block_size
        for i in range(b, t):
            for j in range(l, r):
                frame[i, j] = color
    
    def get_flag_pos(self):
        # return (np.random.randint(self.grid_size), self.grid_size - 1)
        return (15, 15)

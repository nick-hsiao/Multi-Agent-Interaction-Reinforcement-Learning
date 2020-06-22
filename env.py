import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

class CTFEnv(py_environment.PyEnvironment):
    def __init__(self, grid_size=16 , screen_size=512, num_walls=4):
        self.grid_size = grid_size
        self.placement_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        self.num_walls = num_walls
        self.agent_pos = self.get_agent_pos()
        self.flag_pos = self.get_flag_pos()
        # wall_pos is a list because of multiple walls
        self.wall_pos = self.get_wall_pos()

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=7, name='action')

        # total params:
        # x,y agent = 2 +
        # x,y flag = 2 +
        # x,y of each wall = 2 * num_walls
        total_params = 4 + (self.num_walls * 2)
        
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(total_params,), dtype=np.int32, 
            minimum= np.zeros((total_params,), dtype=np.int32),
            maximum= np.full((total_params,), self.grid_size-1, dtype=np.int32), 
            name='observation')

        self._state= self.get_state()
        
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
        self.agent_pos = self.get_agent_pos()
        self.flag_pos = self.get_flag_pos()
        self.wall_pos = self.get_wall_pos()
        self.reset_grid()
        self._state= self.get_state()
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        self.step_count += 1
        if self._episode_ended:
            return self.reset()

        self.move(action)

        if self.game_over():
            self._episode_ended = True

        if self._episode_ended:
            if self.game_over():
                reward = 100
            else:
                reward = 0
            return ts.termination(self._state, reward)
        else:
            return ts.transition(
                self._state, reward=0, discount=0.9)
    
    def move(self, action):
        # Get the current position of the agent.
        row, col = self._state[0],self._state[1]
        # Set the position in the grid to 0 because we are about to move away from it.
        self.placement_grid[row, col] = 0
        if action == 0: #down
            if row - 1 >= 0 and self.placement_grid[row-1][col] != 3:
                self._state[0] -= 1
        if action == 1: #up
            if row + 1 < self.grid_size and self.placement_grid[row+1][col] != 3:
                self._state[0] += 1
        if action == 2: #left
            if col - 1 >= 0 and self.placement_grid[row][col-1] != 3:
                self._state[1] -= 1
        if action == 3: #right
            if col + 1  < self.grid_size and self.placement_grid[row][col+1] != 3:
                self._state[1] += 1
        if action == 4:
            if row + 1 < self.grid_size and col + 1 < self.grid_size and self.placement_grid[row+1][col+1] != 3:
                self._state[0] += 1
                self._state[1] += 1
        if action == 5:
            if row - 1 >= 0 and col + 1 < self.grid_size and self.placement_grid[row-1][col+1] != 3:
                self._state[0] -= 1
                self._state[1] += 1
        if action == 6:
            if row - 1 >= 0 and col - 1 >= 0 and self.placement_grid[row-1][col-1] != 3:
                self._state[0] -= 1
                self._state[1] -= 1
        if action == 7:
            if row + 1 < self.grid_size and col - 1 >= 0 and self.placement_grid[row+1][col-1] != 3:
                self._state[0] += 1
                self._state[1] -= 1
        self.placement_grid[self._state[0], self._state[1]] = 1
    
    def game_over(self):
        row, col, frow, fcol = self._state[0],self._state[1],self._state[2],self._state[3]
        return row==frow and col==fcol
    
    def reset_grid(self):
        for i in range(0, len(self._state), 2):
            self.placement_grid[self._state[i]][self._state[i+1]] = 0

        self.placement_grid[self.agent_pos[0]][self.agent_pos[1]] = 1
        self.placement_grid[self.flag_pos[0]][self.flag_pos[1]] = 2
        for i in range(0, len(self.wall_pos), 2):
            self.placement_grid[self.wall_pos[i]][self.wall_pos[i+1]] = 3
    
    def render(self, mode='rgb_array'):
        frame = np.full((self.screen_size, self.screen_size, 3), 255, dtype=np.uint8)
        for ix,iy in np.ndindex(self.placement_grid.shape):
            current = self.placement_grid[ix, iy]            
            if(current == 1):
                self.fill_color(frame, ix, iy, [0, 0, 255])
            elif(current == 2):
                self.fill_color(frame, ix, iy, [0, 255, 0])
            elif(current == 3):
                self.fill_color(frame, ix, iy, [0, 0, 0])
        return np.flipud(frame)
    
    def fill_color(self, frame, x, y, color):
        l = x * self.block_size
        r = l + self.block_size
        b = y * self.block_size
        t = b + self.block_size
        for i in range(b, t):
            for j in range(l, r):
                frame[i, j] = color
    
    def get_state(self):
        state = [
            self.agent_pos[0],
            self.agent_pos[1],
            self.flag_pos[0],
            self.flag_pos[1],
        ]
        state.extend(self.wall_pos)

        return np.array(state, dtype=np.int32)

    def get_agent_pos(self):
        return (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        # return (0, 0)

    def get_flag_pos(self):
        return (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        # return (15, 15)

    def get_wall_pos(self):
        wall_pos = []
        for _ in range(self.num_walls):
            found = False
            while not found:
                x = np.random.randint(self.grid_size)
                y = np.random.randint(self.grid_size)
                if x != self.agent_pos[0] and y != self.agent_pos[1] and x != self.flag_pos[0] and y != self.flag_pos[1]:
                    wall_pos.append(x)
                    wall_pos.append(y)
                    found = True
        return wall_pos

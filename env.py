import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

class CTFEnv(py_environment.PyEnvironment):
    def __init__(self, grid_size=16 , screen_size=512, num_walls=4, num_agents=4):
        #Set grid
        self.grid_size = grid_size
        self.placement_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)


        self.num_agents = num_agents


        #Set walls, agent, and flag
        self.num_walls = num_walls# (num_walls +self.num_agents)*2

        self.agent_pos = self.get_agent_pos()
        self.flag_pos = self.get_flag_pos()
        # wall_pos is a list because of multiple walls
        self.wall_pos = self.get_wall_pos()

        # agent2_pos to store multiple agents
        self.agent2_pos = self.get_agent2_pos()
        #self.num_agents = 4

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=7, name='action')

        # total params:
        # x,y agent = 2 +
        # x,y flag = 2 +
        # x,y of each wall = 2 * num_walls
        total_params = 4 + (self.num_walls * 2)  + (self.num_agents*2) #the extra 4 represented 4 extra agents
        
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(total_params,), dtype=np.int32, 
            minimum= np.zeros((total_params,), dtype=np.int32),
            maximum= np.full((total_params,), self.grid_size-1, dtype=np.int32), 
            name='observation')

        self._state= self.get_state()
        
        #Start episode
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

        #Collect positions of agent, flag, and wall. Then reset grid.
        self.agent_pos = self.get_agent_pos()
        self.flag_pos = self.get_flag_pos()
        self.wall_pos = self.get_wall_pos()
        self.reset_grid()
        self._state= self.get_state()
        self._episode_ended = False


        self.agent2_pos = self.get_agent2_pos()


        return ts.restart(self._state)


    def _step(self, action):

        #Increase "step" by 1
        self.step_count += 1

        #Reset if episode ended
        if self._episode_ended:
            return self.reset()

        self.move(action)
        
        #Move separate agents
        for i in range(0,self.num_agents):
            self.move2(action,4+(self.num_walls*2)+i)
            #add 1 to i so i can increase twice per loop
            i = i+1

        #If game is over, end episode.
        if self.game_over():
            self._episode_ended = True

        #stop when step_count is 200
        #if self.step_count==200:
        #    self._episode_ended = True
        #    print("Step 200 is step ")
        #    print(self.step_count)

        if self._episode_ended:
            #If episode ended:
            # (i) If game over, then give reward
            # (ii) If game not over, no reward
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
        if action == 4: #diagonal. go right and up
            if row + 1 < self.grid_size and col + 1 < self.grid_size and self.placement_grid[row+1][col+1] != 3:
                self._state[0] += 1
                self._state[1] += 1
        if action == 5: #diagonal. go right and down
            if row - 1 >= 0 and col + 1 < self.grid_size and self.placement_grid[row-1][col+1] != 3:
                self._state[0] -= 1
                self._state[1] += 1
        if action == 6: #diagonal. go left and down
            if row - 1 >= 0 and col - 1 >= 0 and self.placement_grid[row-1][col-1] != 3:
                self._state[0] -= 1
                self._state[1] -= 1
        if action == 7: #diagonal. go left and up
            if row + 1 < self.grid_size and col - 1 >= 0 and self.placement_grid[row+1][col-1] != 3:
                self._state[0] += 1
                self._state[1] -= 1
        self.placement_grid[self._state[0], self._state[1]] = 1

    #Allows separate agents to move
    def move2(self, action, index):
    # Get the current position of the agent.
        row, col = self._state[index],self._state[index+1]

        #Stop agent if agent reaches flag
        if self._state[index] == self._state[2] and self._state[index+1] == self._state[3]:
            return

        # Set the position in the grid to 0 because we are about to move away from it.
        #self.placement_grid[self._state[index], self._state[index+1]] = 0
        self.placement_grid[row, col] = 0

        if action == 0: #down
            if row - 1 >= 0 and self.placement_grid[row-1][col] != 3:
                self._state[index] -= 1
        if action == 1: #up
            if row + 1 < self.grid_size and self.placement_grid[row+1][col] != 3:
                self._state[index] += 1
        if action == 2: #left
            if col - 1 >= 0 and self.placement_grid[row][col-1] != 3:
                self._state[index+1] -= 1
        if action == 3: #right
            if col + 1  < self.grid_size and self.placement_grid[row][col+1] != 3:
                self._state[index+1] += 1
        if action == 4: #diagonal. go right and up
            if row + 1 < self.grid_size and col + 1 < self.grid_size and self.placement_grid[row+1][col+1] != 3:
                self._state[index] += 1
                self._state[index+1] += 1
        if action == 5: #diagonal. go right and down
            if row - 1 >= 0 and col + 1 < self.grid_size and self.placement_grid[row-1][col+1] != 3:
                self._state[index] -= 1
                self._state[index+1] += 1
        if action == 6: #diagonal. go left and down
            if row - 1 >= 0 and col - 1 >= 0 and self.placement_grid[row-1][col-1] != 3:
                self._state[index] -= 1
                self._state[index+1] -= 1
        if action == 7: #diagonal. go left and up
            if row + 1 < self.grid_size and col - 1 >= 0 and self.placement_grid[row+1][col-1] != 3:
                self._state[index] += 1
                self._state[index+1] -= 1
        self.placement_grid[self._state[index], self._state[index+1]] = 4
    
    #Game over when agent reaches flag
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

        for i in range(0, len(self.agent2_pos), 2):
            self.placement_grid[self.agent2_pos[i]][self.agent2_pos[i+1]] = 4
    
    #Set color
    def render(self, mode='rgb_array'):
        frame = np.full((self.screen_size, self.screen_size, 3), 255, dtype=np.uint8)
        for ix,iy in np.ndindex(self.placement_grid.shape):
            current = self.placement_grid[ix, iy]       
            #Agent     
            if(current == 1):
                self.fill_color(frame, ix, iy, [0, 0, 255])
            #Flag
            elif(current == 2):
                self.fill_color(frame, ix, iy, [0, 255, 0])
            #Wall
            elif(current == 3):
                self.fill_color(frame, ix, iy, [0, 0, 0])
            #New agents
            elif(current == 4):
                self.fill_color(frame, ix, iy, [0, 0, 255])
        return np.flipud(frame)
    
    #Fill color
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
        state.extend(self.agent2_pos)
        state.extend(self.wall_pos)
        #state.extend(self.agent2_pos)

        return np.array(state, dtype=np.int32)

    #Agent position
    def get_agent_pos(self):
        return (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        # return (0, 0)

    #Flag position
    def get_flag_pos(self):
        return (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        # return (15, 15)

    #Wall position
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

    #TempAgent position
    #Currently these agents can land on the same space as walls. 
    #Code to check for this condition has been commented below but has not officially been implemented as there is an error
    def get_agent2_pos(self):
        agent2_pos = []
        for _ in range(self.num_agents):
            found = False
            while not found:
                x = np.random.randint(self.grid_size)
                y = np.random.randint(self.grid_size)
                if x != self.agent_pos[0] and y != self.agent_pos[1] and x != self.flag_pos[0] and y != self.flag_pos[1]:
                    agent2_pos.append(x)
                    agent2_pos.append(y)
                    found = True
        return agent2_pos

                    #for i in range(0,len(self.wall_pos)):
                        #if x != self.wall_pos[i] and y != self.wall_pos[i+1]
                        #    agent2_pos.append(x)
                        #    agent2_pos.append(y)
                        #    found = True
                        #    i= i+1
                        #return agent2_pos

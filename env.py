import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

#Code from the following links were used: https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial and https://towardsdatascience.com/tf-agents-tutorial-a63399218309
class CTFEnv(py_environment.PyEnvironment):
    def __init__(self, grid_size=16, screen_size=512, num_walls=5, num_sagents=4, num_dagents=1):
        #Set grid
        self.grid_size = grid_size
        self.placement_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        #Set walls, agent, and flag
        self.num_walls = num_walls
        #self.agent_pos = self.get_agent_pos()
        self.flag_pos = self.get_flag_pos()

        #set stealer agents
        self.num_sagents = num_sagents

        #set defender agents
        self.num_dagents = num_dagents

        # wall_pos is a list because of multiple walls
        self.wall_pos = self.get_wall_pos()

        # sagent_pos to store multiple stealer agents
        self.sagent_pos = self.get_sagent_pos()

        # dagent_pos to store multiple defender agents
        self.dagent_pos = self.get_dagent_pos()

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=7, name='action')

        # total params:
        # x,y flag = 2 +
        # x,y of each wall = 2 * num_walls
        total_params = 2 + (self.num_walls * 2)  + (self.num_sagents*2) + (self.num_dagents*2) 
        
        #Observation spec, written by Josh Gendein, borrowed and modified from https://towardsdatascience.com/tf-agents-tutorial-a63399218309
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
        self.flag_pos = self.get_flag_pos()
        self.wall_pos = self.get_wall_pos()
        self.sagent_pos = self.get_sagent_pos()
        self.dagent_pos = self.get_dagent_pos()
        self.reset_grid()
        self._state= self.get_state()
        self._episode_ended = False


        return ts.restart(self._state)


    #Step method, from https://towardsdatascience.com/tf-agents-tutorial-a63399218309. Modified by Josh Gendein and Richard Pham
    def _step(self, action):

        #Increase "step" by 1. Written by Josh Gendein
        self.step_count += 1

        #Reset if episode ended. Written by Josh Gendein
        if self._episode_ended:
            return self.reset()

        #Move separate defender agents. Written by Richard Pham
        for i in range(0,(self.num_dagents*2),2):
            self.move3(np.random.randint(7),2+(self.num_walls*2)+(self.num_sagents*2)+i)
        
        #Move separate stealer agents. Written by Richard Pham
        #This was added after move3 because there is a condition that checks if space has been occupied by defender. If you move this before move3, then the condition will not work
        for i in range(0,(self.num_sagents*2),2):
            self.move2(action,2+(self.num_walls*2)+i)
            

        #If game is over, end episode. Written by Josh Gendein
        if self.game_over():
            self._episode_ended = True

        #Written by Josh Gendein
        if self._episode_ended:

            #If episode ended:
            # (i) If game over, then give reward
            # (ii) If game not over, no reward
            if self.game_over():
                #reward = 25000
                reward = 100
            else:
                reward = 0
            return ts.termination(self._state, reward)
        else:
            return ts.transition(
                #self._state, reward=0, discount=50.00)
                self._state, reward=0, discount=0.9)

    #Check if all agents were captured. Written by Richard Pham
    def all_agents_captured(self):

        over = False

        #total by adding a bunch of -1 coordinates.
        total = 0

        num_coordinates = 0

        #End game if all stealer agents are stopped
        #for i in range(4+(self.num_walls*2), 4+(self.num_walls*2)+(self.num_sagents*2),2):
        for i in range(2+(self.num_walls*2), 2+(self.num_walls*2)+(self.num_sagents*2),2):

            num_coordinates = num_coordinates + 2

            total = total + self._state[i] + self._state[i+1]

        total = total * -1

        if num_coordinates == total:
            over = True

            
        if over == True:
            #End game if all stealer agents are stopped
            self._episode_ended = True       

    #Allows separate stealer agents to move. Method was originally written by Josh Gendein but modified by Richard Pham
    def move2(self, action, index):
    # Get the current position of the agent.
        # Get the current position of the agent.
        row, col = self._state[index],self._state[index+1]
        # Set the position in the grid to 0 because we are about to move away from it.
        if self.placement_grid[row, col] != 2:
            self.placement_grid[row, col] = 0

        #Stop agent if stealer agent touches defender. Check all defenders
        for i in range(0,(self.num_dagents*2),2):
            if self._state[2+(self.num_walls*2)+(self.num_sagents*2)+i] == self._state[index] and self._state[2+(self.num_walls*2)+(self.num_sagents*2)+i+1] == self._state[index+1]:

                #Vanish from map
                self._state[index] = -1
                self._state[index+1] = -1

                self.placement_grid[self._state[index], self._state[index+1]] = 0

                self.placement_grid[self._state[0], self._state[1]] = 2

                self.all_agents_captured()
                return


           

        #If agent is outside map, agent should not move
        if self._state[index] == -1 and self._state[index+1] == -1:
            return

        #state[0] and state[1] are now the flag positions instead of state[2] and state[3]
        if self._state[index] == self._state[0] and self._state[index+1] == self._state[1]:
            #Reset flag color
            self.placement_grid[row, col] = 2

            #End episode
            self._episode_ended = True
            return
        
        #Stop if agent is not in map
        if self._state[index+1] == -1 or self._state[index] == -1:
            return

        if action == 0: #down
            if row - 1 >= 0 and self.placement_grid[row-1][col] != 3 and self.placement_grid[row-1][col] != 1 and self.placement_grid[row-1][col] != 4:
                self._state[index] -= 1
        if action == 1: #up
            if row + 1 < self.grid_size and self.placement_grid[row+1][col] != 3 and self.placement_grid[row+1][col] != 1 and self.placement_grid[row+1][col] != 4:
                self._state[index] += 1
        if action == 2: #left
            if col - 1 >= 0 and self.placement_grid[row][col-1] != 3 and self.placement_grid[row][col-1] != 1 and self.placement_grid[row][col-1] != 4:
                self._state[index+1] -= 1
        if action == 3: #right
            if col + 1  < self.grid_size and self.placement_grid[row][col+1] != 3 and self.placement_grid[row][col+1] != 1 and self.placement_grid[row][col+1] != 4:
                self._state[index+1] += 1
        if action == 4: #diagonal. go right and up
            if row + 1 < self.grid_size and col + 1 < self.grid_size and self.placement_grid[row+1][col+1] != 3 and self.placement_grid[row+1][col+1] != 1 and self.placement_grid[row+1][col+1] != 4:
                self._state[index] += 1
                self._state[index+1] += 1
        if action == 5: #diagonal. go right and down
            if row - 1 >= 0 and col + 1 < self.grid_size and self.placement_grid[row-1][col+1] != 3 and self.placement_grid[row-1][col+1] != 1 and self.placement_grid[row-1][col+1] != 4:
                self._state[index] -= 1
                self._state[index+1] += 1
        if action == 6: #diagonal. go left and down
            if row - 1 >= 0 and col - 1 >= 0 and self.placement_grid[row-1][col-1] != 3 and self.placement_grid[row-1][col-1] != 1 and self.placement_grid[row-1][col-1] != 4:
                self._state[index] -= 1
                self._state[index+1] -= 1
        if action == 7: #diagonal. go left and up
            if row + 1 < self.grid_size and col - 1 >= 0 and self.placement_grid[row+1][col-1] != 3 and self.placement_grid[row+1][col-1] != 1 and self.placement_grid[row+1][col-1] != 4:
                self._state[index] += 1
                self._state[index+1] -= 1

        self.placement_grid[row, col] = 0

        
        #Stop agent if stealer agent touches defender
        if self.placement_grid[self._state[index], self._state[index+1]] == 5:

            #Vanish from map
            self._state[index] = -1
            self._state[index+1] = -1

            self.placement_grid[self._state[index], self._state[index+1]] = 0

            self.placement_grid[self._state[0], self._state[1]] = 2

            self.all_agents_captured()
            

        else:
            self.placement_grid[self._state[index], self._state[index+1]] = 4
    
    #Allows separate defender agents to move. Method was originally written by Josh Gendein but modified by Richard Pham
    def move3(self, action, index):
    # Get the current position of the agent.
        # Get the current position of the agent.
        row, col = self._state[index],self._state[index+1]
        # Set the position in the grid to 0 because we are about to move away from it.
        self.placement_grid[row, col] = 0

        if action == 0: #down
            if row - 1 >= 0 and self.placement_grid[row-1][col] != 3 and self.placement_grid[row-1][col] != 2 and self.placement_grid[row-1][col] != 5:
                self._state[index] -= 1
        if action == 1: #up
            if row + 1 < self.grid_size and self.placement_grid[row+1][col] != 3 and self.placement_grid[row+1][col] != 2 and self.placement_grid[row+1][col] != 5:
                self._state[index] += 1
        if action == 2: #left
            if col - 1 >= 0 and self.placement_grid[row][col-1] != 3 and self.placement_grid[row][col-1] != 2 and self.placement_grid[row][col-1] != 5:
                self._state[index+1] -= 1
        if action == 3: #right
            if col + 1  < self.grid_size and self.placement_grid[row][col+1] != 3 and self.placement_grid[row][col+1] != 2 and self.placement_grid[row][col+1] != 5:
                self._state[index+1] += 1
        if action == 4: #diagonal. go right and up
            if row + 1 < self.grid_size and col + 1 < self.grid_size and self.placement_grid[row+1][col+1] != 3 and self.placement_grid[row+1][col+1] != 2 and self.placement_grid[row+1][col+1] != 5:
                self._state[index] += 1
                self._state[index+1] += 1
        if action == 5: #diagonal. go right and down
            if row - 1 >= 0 and col + 1 < self.grid_size and self.placement_grid[row-1][col+1] != 3 and self.placement_grid[row-1][col+1] != 2 and self.placement_grid[row-1][col+1] != 5:
                self._state[index] -= 1
                self._state[index+1] += 1
        if action == 6: #diagonal. go left and down
            if row - 1 >= 0 and col - 1 >= 0 and self.placement_grid[row-1][col-1] != 3 and self.placement_grid[row-1][col-1] != 2 and self.placement_grid[row-1][col-1] != 5:
                self._state[index] -= 1
                self._state[index+1] -= 1
        if action == 7: #diagonal. go left and up
            if row + 1 < self.grid_size and col - 1 >= 0 and self.placement_grid[row+1][col-1] != 3 and self.placement_grid[row+1][col-1] != 2 and self.placement_grid[row+1][col-1] != 5:
                self._state[index] += 1
                self._state[index+1] -= 1

        self.placement_grid[row, col] = 0
        self.placement_grid[self._state[index], self._state[index+1]] = 5

    #Game over when agent reaches flag
    #def game_over(self):
    #    row, col, frow, fcol = self._state[0],self._state[1],self._state[2],self._state[3]
    #    return row==frow and col==fcol

    #Game over when agent reaches flag
    def game_over(self):
        over = False

        #Check if at least one agent reached flag
        for i in range(2+(self.num_walls*2), 2+(self.num_walls*2)+(self.num_sagents),2):
            #state[0] and state[1] are flag positions
            if(self._state[i] == self._state[0] and self._state[i+1] == self._state[1]):
                over = True

        return over
    
    #Grid reset, written by Josh Gendein
    def reset_grid(self):
        for i in range(0, len(self._state), 2):
            self.placement_grid[self._state[i]][self._state[i+1]] = 0

        #self.placement_grid[self.agent_pos[0]][self.agent_pos[1]] = 1
        self.placement_grid[self.flag_pos[0]][self.flag_pos[1]] = 2

        for i in range(0, len(self.wall_pos), 2):
            self.placement_grid[self.wall_pos[i]][self.wall_pos[i+1]] = 3

        for i in range(0, len(self.sagent_pos), 2):
            self.placement_grid[self.sagent_pos[i]][self.sagent_pos[i+1]] = 4
        
        for i in range(0, len(self.dagent_pos), 2):
            self.placement_grid[self.dagent_pos[i]][self.dagent_pos[i+1]] = 5
    
    #Set color, written by Josh Gendein
    def render(self, mode='rgb_array'):
        frame = np.full((self.screen_size, self.screen_size, 3), 255, dtype=np.uint8)
        for ix,iy in np.ndindex(self.placement_grid.shape):
            current = self.placement_grid[ix, iy]       
            #Main agent (all code for main agent has been removed)     
            if(current == 1):
                self.fill_color(frame, ix, iy, [0, 0, 255])
            #Flag
            elif(current == 2):
                self.fill_color(frame, ix, iy, [0, 255, 0])
            #Wall
            elif(current == 3):
                self.fill_color(frame, ix, iy, [0, 0, 0])
            #New stealer agents
            elif(current == 4):
                self.fill_color(frame, ix, iy, [0, 0, 255])
            #New defender agents
            elif(current == 5):
                self.fill_color(frame, ix, iy, [255, 0, 0])
        return np.flipud(frame)
    
    #Fill color, written by Josh Gendein
    def fill_color(self, frame, x, y, color):
        l = x * self.block_size
        r = l + self.block_size
        b = y * self.block_size
        t = b + self.block_size
        for i in range(b, t):
            for j in range(l, r):
                frame[i, j] = color
    
    #State, written by Josh Gendein
    def get_state(self):
        state = [
            #self.agent_pos[0],
            #self.agent_pos[1],
            self.flag_pos[0],
            self.flag_pos[1],
        ]
        state.extend(self.wall_pos)
        state.extend(self.sagent_pos)
        state.extend(self.dagent_pos)

        return np.array(state, dtype=np.int32)

    #Agent position
    #def get_agent_pos(self):
    #    return (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
    #    # return (0, 0)

    #Flag position, written by Josh Gendein
    def get_flag_pos(self):
        return (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        #return (15, 15)

    #Wall position. Written by Aliaksandr Nenartovich
    def get_wall_pos(self):
        wall_pos = []
        for _ in range(self.num_walls):
            found = False
            while not found:
                x = np.random.randint(self.grid_size)
                y = np.random.randint(self.grid_size)
                #if x != self.agent_pos[0] and y != self.agent_pos[1] and x != self.flag_pos[0] and y != self.flag_pos[1]:
                if self.placement_grid[x, y] == 0 and x != self.flag_pos[0] and y != self.flag_pos[1]:
                    wall_pos.append(x)
                    wall_pos.append(y)
                    found = True
        return wall_pos

    #Stealer Agent position. Written by Richard Pham
    def get_sagent_pos(self):
        sagent_pos = []
        for _ in range(self.num_sagents):
            found = False
            while not found:
                x = np.random.randint(self.grid_size)
                y = np.random.randint(self.grid_size)

                #Spawn in empty location
                if self.placement_grid[x, y] == 0 and x != self.flag_pos[0] and y != self.flag_pos[1]:
                    sagent_pos.append(x)
                    sagent_pos.append(y)
                    found = True
        return sagent_pos

    #Defender Agent position. Written by Richard Pham
    def get_dagent_pos(self):
        dagent_pos = []
        for _ in range(self.num_dagents):
            found = False
            while not found:
                x = np.random.randint(self.grid_size)
                y = np.random.randint(self.grid_size)

                #Spawn in empty location
                if self.placement_grid[x, y] == 0 and x != self.flag_pos[0] and y != self.flag_pos[1]:
                    dagent_pos.append(x)
                    dagent_pos.append(y)
                    found = True
        return dagent_pos
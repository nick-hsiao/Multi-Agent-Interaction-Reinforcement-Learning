import numpy as np
import time
from world import World
from env import Env


grid_size = 32

world = World(grid_size=grid_size, n_agents=1, flag_size=1)

env = Env(world=world)
env.reset()

for _ in range(env.horizon): 
    env.render()
    obs, reward, done, _ = env.step([env.action_space.sample()]) # Take a random action
    if done:
        print("Got to the flag!")
        env.close()
        break
    time.sleep(1 / env.horizon)

import numpy as np
import time
from environment.world import World
from environment.env import Env

# Enable this if you want to print out the entire grid.
# np.set_printoptions(threshold=np.inf)

grid_size = 32

env = Env(grid_size=grid_size, n_agents=1, flag_size=1)
initial = env.reset()

print(env.action_space)
print(env.observation_space)

start = time.process_time_ns()

for _ in range(env.horizon): 
    env.render()
    obs, reward, done, _ = env.step(env.action_space.sample()) # Take a random action
    if done:
        print("Got to the flag!")
        env.close()
        break
    time.sleep(1 / env.horizon)

end = time.process_time_ns()

print(end-start)
import tensorflow as tf
from tf_agents.environments import wrappers, tf_py_environment
from env import CTFEnv
import imageio
import sys

import tkinter as tk
from PIL import Image, ImageTk


#Code from the following links were used: https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial

model_path = "policies/policy_static_goal_dynamic_reward"
filename = "videos/example_training"
filename = filename + ".mp4"

if(str.isdigit(sys.argv[1])):
    grid_size = int(sys.argv[1])
else:
    grid_size = 0

if(str.isdigit(sys.argv[2])):
    num_walls = int(sys.argv[2])
else:
    num_walls = 0

if(str.isdigit(sys.argv[3])):
    num_agents = int(sys.argv[3])
else:
    num_agents = 0

if(str.isdigit(sys.argv[4])):
    def_agents = int(sys.argv[4])
else:
    def_agents = 0

py_env = wrappers.TimeLimit(CTFEnv(grid_size, 512, num_walls, num_agents, def_agents), duration=100)

env = tf_py_environment.TFPyEnvironment(py_env)

policy = tf.saved_model.load(model_path)

#Video of 5 simulations, written by Josh Gendein, borrowed and modified from https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
with imageio.get_writer(filename, fps=15) as video:
    for _ in range(5):
        time_step = env.reset()
        video.append_data(py_env.render())
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
            video.append_data(py_env.render())

root = tk.Tk()
root.geometry("300x100")
root.title("Complete")
text = tk.Label(root, text="Video has been generated! \n Please check the videos folder!")
text.pack(pady = (0,0))
root.mainloop()
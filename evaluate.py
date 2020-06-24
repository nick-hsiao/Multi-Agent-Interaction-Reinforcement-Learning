import tensorflow as tf
from tf_agents.environments import wrappers, tf_py_environment
from env import CTFEnv
import imageio

model_path = "policies/policy_static_goal_dynamic_reward"
filename = "videos/example_training"
filename = filename + ".mp4"

py_env = wrappers.TimeLimit(CTFEnv(), duration=100)

env = tf_py_environment.TFPyEnvironment(py_env)

policy = tf.saved_model.load(model_path)

#Video of 5 simulations
with imageio.get_writer(filename, fps=15) as video:
    for _ in range(5):
        time_step = env.reset()
        video.append_data(py_env.render())
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
            video.append_data(py_env.render())
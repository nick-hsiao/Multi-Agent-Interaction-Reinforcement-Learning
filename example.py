from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from env import CTFEnv
from tf_agents.environments import wrappers
import imageio

py_env = wrappers.TimeLimit(CTFEnv(), duration=100)

env = tf_py_environment.TFPyEnvironment(py_env)

policy = random_tf_policy.RandomTFPolicy(
    env.time_step_spec(), 
    env.action_spec()
)

time_step = env.reset()
print(time_step)
while not time_step.is_last():
    action_step = policy.action(time_step)
    time_step = env.step(action_step.action)
    print(time_step.observation)

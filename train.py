from __future__ import absolute_import, division, print_function
import gym
## import gym_ctf

import numpy as np
np.set_printoptions(threshold=np.inf)

import tensorflow as tf
import imageio

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.environments import wrappers
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.policies import policy_saver

import matplotlib
import matplotlib.pyplot as plt
import sys

from env import CTFEnv

#Code from the following links were used: https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial and https://towardsdatascience.com/tf-agents-tutorial-a63399218309

tf.compat.v1.enable_v2_behavior()

#9 Hyperparameters, written by Josh Gendein, borrowed and modified from the tutorial in https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
num_iterations = 30000 # @param {type:"integer"}
initial_collect_steps = 1000  # @param {type:"integer"} 
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}
batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-5  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}
num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}


#Parameters for agents and walls
grid_size = int(sys.argv[1])
num_walls = int(sys.argv[2])
num_agents = int(sys.argv[3])
def_agents = int(sys.argv[4])
c = CTFEnv(grid_size, 512, num_walls, num_agents, def_agents)

#Training environment, written by Josh Gendein, borrowed and modified from the tutorial in https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
#Simulation will last 200 steps
train_py_env = wrappers.TimeLimit(c, duration=200)
eval_py_env = wrappers.TimeLimit(c, duration=200)


#Training environment, written by Josh Gendein, borrowed and modified from the tutorial in https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

#Q Network, written by Josh Gendein, borrowed and modified from the tutorial in https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
fc_layer_params = (2000,)
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params
)

#Q Network Optimizer, written by Josh Gendein, borrowed from the tutorial in https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_step_counter = tf.compat.v2.Variable(0)
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter
)
agent.initialize()

#Policy, written by Josh Gendein, from the tutorial in https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
eval_policy = agent.policy
collect_policy = agent.collect_policy

#Training metrics, written by Josh Gendein, borrowed from the tutorial in https://towardsdatascience.com/tf-agents-tutorial-a63399218309
train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
]

#Average return method, written by Josh Gendein, from the tutorial in https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

#Replay buffer, written by Josh Gendein, from the tutorial in https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length,
)

#Replay observer, written by Josh Gendein, from the tutorial in https://towardsdatascience.com/tf-agents-tutorial-a63399218309
replay_observer = [replay_buffer.add_batch]

#Collect step method, written by Josh Gendein, from the tutorial in https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)

#Collect data method, written by Josh Gendein, from the tutorial in https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)
    
#Dataset, written by Josh Gendein, from the tutorial in https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)

#Driver, written by Josh Gendein, from the tutorial in https://towardsdatascience.com/tf-agents-tutorial-a63399218309
driver = dynamic_step_driver.DynamicStepDriver(
            train_env,
            collect_policy,
            observers=replay_observer + train_metrics,
    num_steps=1)

iterator = iter(dataset)

#The process of training the agent, written by Josh Gendein, from the tutorial in https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
# Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step, written by Josh Gendein, from the tutorial in https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
agent.train_step_counter.assign(0)

# Time step, written by Josh Gendein, from the tutorial in https://towardsdatascience.com/tf-agents-tutorial-a63399218309
final_time_step, policy_state = driver.run()
for i in range(1000):
    final_time_step, _ = driver.run(final_time_step, policy_state)

# Evaluate the agent's policy once before training, written by Josh Gendein, borrowed and modified from the tutorial in https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
# Lines 167 to 190
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]
episode_len = []
step_len = []
for _ in range(num_iterations):

    # Time step, written by Josh Gendein, from the tutorial in https://towardsdatascience.com/tf-agents-tutorial-a63399218309
    final_time_step, _ = driver.run(final_time_step, policy_state)

  # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))
        episode_len.append(train_metrics[3].result().numpy())
        step_len.append(step)
        print('Average episode length: {}'.format(train_metrics[3].result().numpy()))
    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)


#Plot graph after training, written by Josh Gendein, borrowed and modified from https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
plt.plot(step_len, episode_len)
plt.ylabel('Episodes')
plt.xlabel('Average Episode Length (Steps)')
plt.show()

#Create video method, written by Josh Gendein, from https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
def create_policy_eval_video(policy, filename, num_episodes=5, fps=30):
    filename = filename + ".mp4"
    with imageio.get_writer(f'videos/{filename}', fps=fps) as video:
        for _ in range(num_episodes):
            time_step = eval_env.reset()
            video.append_data(eval_py_env.render())
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                video.append_data(eval_py_env.render())

#Written by Josh Gendein
def save_policy(policy, filename):
    saver = policy_saver.PolicySaver(policy, batch_size=None)
    saver.save('policies/policy_%s' % filename)

filename='static_goal_dynamic_reward'

save_policy(agent.policy, filename)

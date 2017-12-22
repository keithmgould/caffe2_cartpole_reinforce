# modified from:
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

from caffe2.python import core, net_drawer, model_helper, workspace, brew
from matplotlib import pyplot
from itertools import count
import numpy as np
import argparse
import random
import pdb
import gym

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')

args = parser.parse_args()

env = gym.make('CartPole-v1')
env.seed(args.seed)

HIDDEN_SIZE = 128

# apply the discount
def apply_discount(raw_rewards):
  R = 0
  rewards = []
  for r in raw_rewards:
      # apply the discount
      R = r + args.gamma * R
      rewards.insert(0, R)
  return rewards

# give rewards a zero mean, and a std of 1
def normalize_rewards(raw_rewards):
  rewards = np.array(raw_rewards)
  rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
  return rewards

def calculate_loss(rewards, ep_predictions):
  loss = 0
  natural_log_predictions = np.log(ep_predictions)
  for i in range(np.size(rewards)):
    foo = rewards[i] * natural_log_predictions[i]
    loss += foo

  return loss

def finish_episode(ep_rewards, ep_predictions):
  rewards = apply_discount(ep_rewards)
  rewards = normalize_rewards(rewards)

  new_loss = calculate_loss(rewards, ep_predictions)
  print("loss: {}".format(new_loss))

  for index in range(np.size(ep_rewards)):
    reward = np.array([ep_rewards[index]])
    workspace.FeedBlob("loss", new_loss)
    workspace.RunNet(full_net)

  return True;

# flip the weighted coin
def select_action(prediction):
  if prediction < np.random.uniform(0, 1):
    return 0
  else:
    return 1

avg_t = np.array([])

input_data = np.random.rand(1, 4).astype(np.float32)
workspace.FeedBlob("input_data", input_data)

forward_model = model_helper.ModelHelper(name="forward")
forward_init_net = forward_model.param_init_net
forward_net = forward_model.net

brew.fc(forward_model, 'input_data', 'hidden', 4, HIDDEN_SIZE)
brew.relu(forward_model, 'hidden', 'hidden')
brew.fc(forward_model, 'hidden', 'prediction', HIDDEN_SIZE, 1)
forward_model.Sigmoid('prediction', 'prediction')

full_model = model_helper.ModelHelper(name="full")
full_init_net = full_model.param_init_net
full_net = full_model.net
loss = full_net.ConstantFill([], "loss", shape=[1], value=0.0)
ONE = full_net.ConstantFill([], "ONE", shape=[1], value=1.)

brew.fc(full_model, 'input_data', 'hidden', 4, HIDDEN_SIZE)
brew.relu(full_model, 'hidden', 'hidden')
brew.fc(full_model, 'hidden', 'prediction', HIDDEN_SIZE, 1)
full_model.Sigmoid('prediction', 'prediction')
gradient_map = full_net.AddGradientOperators(['loss'])
# full_net.WeightedSum(['prediction_w', ONE, gradient_map['prediction_w'], ONE], 'prediction_w')
# full_net.WeightedSum(['prediction_b', ONE, gradient_map['prediction_w'], ONE], 'prediction_b')

# Run the init nets once
workspace.RunNetOnce(forward_init_net)
workspace.RunNetOnce(full_init_net)

# Create the forward and full nets
workspace.CreateNet(forward_net)
workspace.CreateNet(full_net)

graph = net_drawer.GetPydotGraph(forward_net.Proto().op, "forward", rankdir="LR")
graph.write_png('forward.png', prog='dot')

graph = net_drawer.GetPydotGraph(full_net.Proto().op, "full", rankdir="LR")
graph.write_png('full.png', prog='dot')

for i_episode in count(1):
    ep_rewards = []
    ep_states = []
    ep_actions = []
    ep_predictions = []
    state = env.reset()

    for t in range(1000):
        env.render() if args.render else False
        scrubbed_state = state.reshape(1,4).astype(np.float32)
        workspace.FeedBlob("input_data", scrubbed_state)
        workspace.RunNet(forward_net)
        prediction = workspace.FetchBlob('prediction')
        action = select_action(prediction)
        state, reward, done, _ = env.step(action)

        # store things
        ep_predictions.append(prediction)
        ep_states.append(state)
        ep_rewards.append(reward)
        ep_actions.append(action)

        # if cart distance too far from x=0
        if abs(state[0]) > 1:
            break

        if done:
            break

    avg_t = np.append(avg_t, t)
    finish_episode(ep_rewards, ep_predictions)
    print("t: {}, avg_t: {}".format(t, avg_t.mean()))
    exit() # tmp!! just so I can stop after one episode

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

def normalize_rewards(raw_rewards):
  R = 0
  rewards = []
  for r in raw_rewards:
      # apply the discount
      R = r + args.gamma * R
      rewards.insert(0, R)

  # give rewards a zero mean, and a std of 1
  rewards = np.array(rewards)
  rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

  return rewards

def finish_episode(ep_rewards, ep_actions, ep_states):
  rewards = normalize_rewards(ep_rewards)
  # for index in np.size(ep_rewards):
    # workspace.RunNet(train_net.Proto().name)

  return True;

# flip the weighted coin
def select_action(prediction):
  if prediction < np.random.uniform(0, 1):
    return 0
  else:
    return 1

avg_t = np.array([])

workspace.CreateBlob('prediction')

input_data = np.random.rand(1, 4).astype(np.float32)
workspace.FeedBlob("input_data", input_data)

forward_model = model_helper.ModelHelper(name="forward")
forward_init_net = forward_model.param_init_net
forward_net = forward_model.net

brew.fc(forward_model, 'input_data', 'hidden', 4, HIDDEN_SIZE)
brew.relu(forward_model, 'hidden', 'hidden')
brew.fc(forward_model, 'hidden', 'prediction', HIDDEN_SIZE, 1)
forward_model.Sigmoid('prediction', 'prediction')

backward_model = model_helper.ModelHelper(name="backward")
backward_init_net = backward_model.param_init_net
backward_net = backward_model.net

# ITER is the iterator count.
ITER = backward_init_net.ConstantFill([], "ITER", shape=[1], value=0, dtype=core.DataType.INT32)

# Constant value ONE is used in weighted sum when updating parameters.
ONE = backward_init_net.ConstantFill([], "ONE", shape=[1], value=1.)

# Compute the learning rate that corresponds to the iteration.
LR = backward_net.LearningRate(ITER, "LR", base_lr=-0.1, policy="step", stepsize=20, gamma=0.9)

# Increment the iteration by one.
backward_net.Iter(ITER, ITER)

# Run the init nets once
workspace.RunNetOnce(forward_init_net)
workspace.RunNetOnce(backward_init_net)

# Create the forward and backward nets
workspace.CreateNet(forward_net)
workspace.CreateNet(backward_net)

print("Current network proto:\n\n{}".format(backward_net.Proto()))
print("Current blobs in the workspace: {}".format(workspace.Blobs()))

graph = net_drawer.GetPydotGraph(forward_net.Proto().op, "forward", rankdir="LR")
graph.write_png('forward.png', prog='dot')

graph = net_drawer.GetPydotGraph(backward_net.Proto().op, "backward", rankdir="LR")
graph.write_png('backward.png', prog='dot')

for i_episode in count(1):
    state = env.reset()
    ep_rewards = []
    ep_states = []
    ep_actions = []
    ep_predictions = []

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
    workspace.RunNet(backward_net)
    finish_episode(ep_rewards, ep_actions, ep_states)
    print("t: {}, avg_t: {}".format(t, avg_t.mean()))
    exit()

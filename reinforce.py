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
  for index in np.size(ep_rewards):
    workspace.RunNet(train_net.Proto().name)

  return True;

# flip the weighted coin
def select_action(prediction):
  if prediction < np.random.uniform(0, 1):
    return 0
  else:
    return 1

avg_t = np.array([])

workspace.CreateBlob('loss')

input_data = np.random.rand(1, 4).astype(np.float32)
workspace.FeedBlob("input_data", input_data)

model = model_helper.ModelHelper(name="train")
init_net = model.param_init_net
train_net = model.net

brew.fc(model, 'input_data', 'hidden', 4, HIDDEN_SIZE)
brew.relu(model, 'hidden', 'hidden')
brew.fc(model, 'hidden', 'prediction', HIDDEN_SIZE, 1)
model.Sigmoid('prediction', 'prediction')

workspace.RunNetOnce(init_net)
workspace.CreateNet(train_net)

graph = net_drawer.GetPydotGraph(train_net.Proto().op, "train", rankdir="LR")
graph.write_png('rf1.png', prog='dot')

exit()

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
        workspace.RunNet(train_net)
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
    finish_episode(ep_rewards, ep_actions, ep_states)
    print("t: {}, avg_t: {}".format(t, avg_t.mean()))

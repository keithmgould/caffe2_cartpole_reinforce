### Intro

Uses [Caffe2](https://caffe2.ai/) to implement reinforcement learning via gradient descent.

The problem solved is the [openAI Gym](https://gym.openai.com) [Cartpole problem](https://gym.openai.com/envs/CartPole-v0/).

### Status

Currently the script manages the forward pass, but I don't get how to
do the backward pass within the context of Caffe2.

### Inspiration

This script is inspired by [this Pytorch example](https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py)

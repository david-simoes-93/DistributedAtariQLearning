# !/usr/bin/python3

import gym
from gym.spaces import *
import random
import sys
import numpy as np
import pygame
from time import sleep


class GymBridge(gym.Env):
    def __init__(self, size=40):
        # super?
        self.max_actions = 4

        # GUI
        self.metadata = {'render.modes': ['human']}

        # Public GYM variables
        # The Space object corresponding to valid actions
        self.action_space = gym.spaces.Discrete(self.max_actions)
        # The Space object corresponding to valid observations
        self.observation_space = gym.spaces.Tuple([gym.spaces.Discrete(2) for _ in range(size)])

        # A tuple corresponding to the min and max possible rewards
        self.reward_range = [-1, 1]
        self.width = size
        self.agent_positions = 0

        self.terminal = True  # whether simulation is on-going
        self.screen = None

    def _render(self, mode='human', close=False):
        return

    def _reset(self):
        self.agent_positions = int(self.width/3)

        return self.get_state()

    def _step(self, action):
        reward = -0.1
        terminal = False
        if action == 1:
            self.agent_positions += 1
            reward = 10 if self.agent_positions == self.width-1 else reward
            terminal = True if self.agent_positions == self.width-1 else False
        if action == 2:
            self.agent_positions -= 1
            reward = 1 if self.agent_positions == 0 else reward
            terminal = True if self.agent_positions == 0 else False
        if action == 3:
            reward = -1
        if action == 0:
            reward = -1

        return self.get_state(), reward, terminal, {}

    def get_state(self):
        state = np.zeros(self.width)
        state[self.agent_positions]=1
        return state

    def _close(self):
        return

    def _seed(self, seed=None):
        if seed is None:
            seed = random.randrange(sys.maxsize)
        random.seed(seed)
        return [seed]

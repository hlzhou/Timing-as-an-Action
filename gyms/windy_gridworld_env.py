"""Timing-as-an-action version of windy gridworld simulator."""

import gym
from gym import spaces
from random import randint
import numpy.random as npr
import numpy as np


"""
.   .   .   .   .   .   .   .   .   .
.   .   .   .   .   .   .   .   .   .
.   .   .   .   H   H   .   .   .   .
S   .   .   .   .   .   .   G   .   .
.   .   .   .   .   .   .   .   .   .
.   .   .   .   .   .   .   .   .   .
.   .   .   .   .   .   .   .   .   .
            ^   ^   ^   ^^  ^^  ^

hazards = -5
stochastic wind w.p. 0.5
"""


class CustomWindyGridworldEnv(gym.Env):
    def __init__(self, hazards=False, start_state='random', 
                 stochastic_wind=False, goal_reward=10000):
        self.height = 7
        self.width = 10
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
                spaces.Discrete(self.height),
                spaces.Discrete(self.width)
                ))
        self.moves = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1),  # left
        }

        self.hazards = hazards
        self.start_state = start_state
        self.stochastic_wind = stochastic_wind
        self.wind_prob = 0.5  # if stochastic wind

        self.default_start_loc = (3, 0)
        self.hazard_locs = [(2, 4), (2, 5)]
        self.wind_cols = [3, 4, 5, 8]
        self.wind2_cols = [6, 7]
        self.goal_loc = (3, 7)

        self.goal_reward = goal_reward
        self.hazard_cost = 5

        self.reset()  # begin in start state

    def _construct_T(self, action_names, state_names):  # A x S x S
        assert(len(action_names) == len(self.moves))
        assert(len(state_names) == self.height * self.width)
        T = np.zeros((len(self.moves), self.height * self.width, self.height * self.width))

        for i, move in self.moves.items():
            for j, s1 in enumerate(state_names):
                s2 = (s1[0] + move[0], s1[1] + move[1])
                s2 = self._move_in_bounds(s2)
                k = state_names.index(s2)

                if self.stochastic_wind: 
                    s2_wind = self._apply_wind(s1, stochastic_wind=False)  # definitely apply wind
                    s2_wind = (s2_wind[0] + move[0], s2_wind[1] + move[1])
                    s2_wind = self._move_in_bounds(s2_wind)
                    k_wind = state_names.index(s2_wind)

                    T[i, j, k] += 1 - self.wind_prob
                    T[i, j, k_wind] += self.wind_prob
                else:
                    T[i, j, k] += 1
        T[:, state_names.index(self.goal_loc), :] = 0
        T[:, state_names.index(self.goal_loc), state_names.index(self.goal_loc)] = 1
        return T

    def _apply_wind(self, S, stochastic_wind):
        if stochastic_wind:
            if npr.uniform() > self.wind_prob:
                if S[1] in self.wind_cols:
                    S = S[0] - 1, S[1]
                elif S[1] in self.wind2_cols:
                    S = S[0] - 2, S[1]
        else:
            if S[1] in self.wind_cols:
                S = S[0] - 1, S[1]
            elif S[1] in self.wind2_cols:
                S = S[0] - 2, S[1]
        return S

    def _move_in_bounds(self, S):
        S = max(0, S[0]), max(0, S[1])
        S = (min(S[0], self.height - 1),
             min(S[1], self.width - 1))
        return S

    def reward_func(self, s, a):
        if s == self.goal_loc:
            return self.goal_reward
        elif self.hazards and (s in self.hazard_locs):
            return -self.hazard_cost
        else:
            return -1

    def step(self, action):
        if self.S == self.goal_loc:
            return self.S, self.goal_reward, True, {}
            
        move = self.moves[action]

        self.S = self._apply_wind(self.S, self.stochastic_wind)  # apply wind
        self.S = self.S[0] + move[0], self.S[1] + move[1]  # move
        self.S = self._move_in_bounds(self.S)  # move in bounds

        if self.S == self.goal_loc:
            return self.S, self.goal_reward, True, {}
        elif self.hazards and (self.S in self.hazard_locs):
            return self.S, -self.hazard_cost, False, {}
        return self.S, -1, False, {}

    def reset(self):
        if self.start_state == 'random':
            h_options = list(range(0, self.height))
            w_options = list(range(0, self.width))

            self.S = (int(npr.choice(h_options)), int(npr.choice(w_options)))
            assert(self.S[0] < self.height)
            assert(self.S[1] < self.width)
        else:
            self.S = self.default_start_loc
        print('start state: ', self.S)
        return self.S

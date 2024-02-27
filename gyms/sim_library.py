"""Functions to set up and retreive timing-as-an-action simulators."""


from gyms.simulator import OpenAISimulatorWrapper, Simulator
from gyms.glucose_env import GlucoseSimulator
import numpy as np
import numpy.random as npr
from utils import sample_simplex
import gym
from gym import spaces
from gym.envs.registration import register
from gyms.windy_gridworld_env import CustomWindyGridworldEnv


def simple_reward_function(s, a):
    lookup = {
        0: 10, 
        1: 5, 
        2: 0
    }
    state_reward = lookup[s]
    return state_reward

def get_simple_simulator(actions, states, state_p0, horizon=None, gamma=0.8, action_cost=5):
    T = np.array([
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
            [0.0, 0.0, 1.0]
        ],
        [
            [0.8, 0.15, 0.05],
            [0.1, 0.5, 0.4],
            [0.0, 0.0, 1.0]
        ],
    ])
    
    terminal_state = 2
    simulator = Simulator(
        'simple',
        actions, states,
        T, simple_reward_function, action_cost, terminal_state, state_p0, horizon=horizon, gamma=gamma)
    return simulator

def stronger_reward_function(s, a, terminal_reward):
    lookup = {
        0: 25, 
        1: 5, 
        2: terminal_reward
    }
    state_reward = lookup[s]
    return state_reward

def get_stronger_tradeoff_simulator(
    actions, states, state_p0, 
    horizon=None, gamma=0.8, action_cost=3,
    terminal_reward=0):
    T = np.array([
        [
            [0.89, 0.1, 0.01],
            [0.15, 0.8, 0.05],
            [0.0, 0.0, 1.0]
        ],
        [
            [0.1, 0.89, 0.01],
            [0.8, 0.15, 0.05],
            [0.0, 0.0, 1.0]
        ],
    ])

    terminal_state = 2
    simulator = Simulator(
        'stronger_tradeoff',
        actions, states,
        T, 
        lambda s, a: stronger_reward_function(s, a, terminal_reward=terminal_reward), 
        action_cost, terminal_state, state_p0, 
        horizon=horizon, gamma=gamma, terminal_reward=terminal_reward)
    return simulator

def get_bigspace_simulator(actions, states, state_p0, gamma, action_cost, horizon=None, seed=0):
    """Sample from simplex. Idea from Donald Rubin's The Bayesian Bootstrap.
    https://projecteuclid.org/journals/annals-of-statistics/volume-9/issue-1/The-Bayesian-Bootstrap/10.1214/aos/1176345338.full
    https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    """
    S = len(states)
    A = len(actions)

    def bigspace_reward_function(s, a):
        lookup = {s: 5 * (S - 1 - s) for s in states}
        state_reward = lookup[s]
        return state_reward
    
    npr.seed(seed)
    T = np.zeros((A, S, S))
    for a in range(A):
        for s in range(S - 1):
            T[a][s] = sample_simplex(S)
        T[a][S - 1][-1] = 1
    print(T.sum())
    print(len(states) * len(actions))
    assert(T.sum() == len(states) * len(actions))
    terminal_state = len(states) - 1
    simulator = Simulator(
        f'bigspace_{S}s_{A}a',
        actions, states, 
        T, bigspace_reward_function, action_cost, terminal_state, state_p0, horizon=horizon, gamma=gamma)
    return simulator

def get_windy_gridworld_simulator(gamma, action_cost, horizon=None, terminal_reward=0):
    name = 'WindyGridworld-v0'
    env = gym.make(name)

    action_names = ['up', 'right', 'down', 'left']
    state_names = []
    for s1 in range(env.height):
        for s2 in range(env.width):
            state_names.append((s1, s2))
    terminal_state = (3, 7)
    simulator = OpenAISimulatorWrapper(
        name, env, action_names, state_names, action_cost, terminal_state, horizon=horizon, gamma=gamma, terminal_reward=terminal_reward)
    simulator.move_to_dir = {
        'up': (-1, 0),
        'right': (0, 1),
        'down': (1, 0),
        'left': (0, -1),
    }
    return simulator

def get_glucose_simulator(gamma, action_cost, horizon=None, terminal_reward=0):
    simulator = GlucoseSimulator(
        action_cost, horizon=horizon, gamma=gamma, terminal_reward=terminal_reward)
    return simulator

def get_custom_windy_gridworld_simulator(
    gamma, action_cost, horizon=None, 
    hazards=True, start_state='random', stochastic_wind=True, goal_reward=10000):

    env = CustomWindyGridworldEnv(hazards=hazards, start_state=start_state, stochastic_wind=stochastic_wind, goal_reward=goal_reward)
    action_names = ['up', 'right', 'down', 'left']
    state_names = []
    for s1 in range(env.height):
        for s2 in range(env.width):
            state_names.append((s1, s2))
    terminal_state = env.goal_loc
    simulator = OpenAISimulatorWrapper(
        'custom_windygrid', 
        env, action_names, state_names, 
        action_cost, terminal_state, 
        horizon=horizon, gamma=gamma, terminal_reward=goal_reward)
    simulator.move_to_dir = {
        'up': (-1, 0),
        'right': (0, 1),
        'down': (1, 0),
        'left': (0, -1),
    }
    return simulator

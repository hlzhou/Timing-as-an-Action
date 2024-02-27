"""Reward models."""

import torch
import torch.nn as nn
from transition_model import OracleTransitionModel


def get_feature_vec(s, a, d, states, actions, deltas):
    s_idx = s
    a_idx = a + len(states)
    d_idx = d + len(states) + len(actions)

    vec = torch.zeros(len(states) + len(actions) + len(deltas))
    vec[s_idx] = 1
    vec[a_idx] = 1
    vec[d_idx] = 1

    return vec


class CondensedRewardModel(nn.Module):
    def __init__(self, states, actions, deltas, delta_names,
                 terminal_state, gamma, action_cost, device,
                 is_windygrid=False,
                 env=None,
                 oracle_reward_lookup=True):
        super(CondensedRewardModel, self).__init__()
        self.states = states
        self.actions = actions
        self.deltas = deltas
        self.delta_names = delta_names

        self.terminal_state = terminal_state
        self.gamma = gamma
        self.action_cost = action_cost
        self.is_windygrid = is_windygrid
        self.env = env
        self.oracle_reward_lookup = oracle_reward_lookup

        self.device = device

        if oracle_reward_lookup:
            if self.is_windygrid:
                self.reward_lookup = torch.Tensor([[env.reward_function(env.idx_to_state[st], a) for st in states] for a in actions]).to(self.device)
            else:
                self.reward_lookup = torch.Tensor([[env.reward_function(st, a) if st != self.terminal_state else 0 for st in states] for a in actions]).to(self.device)
        else:
            self.reward_lookup = torch.Tensor([[-1 if st != self.terminal_state else 0 for st in states] for a in actions]).to(self.device)
            self.reward_lookup = torch.nn.parameter.Parameter(self.reward_lookup, requires_grad=True)

        if self.is_windygrid:
            self.terminal_reward = torch.Tensor([10000]).to(self.device)
        else:
            self.terminal_reward = torch.Tensor([0]).to(self.device)

    def forward(self):
        raise NotImplementedError()

    def get_prediction(self, s, a, d, transition_model):  # d can be 0-indexed
        terminal_reward = torch.Tensor([0]).to(self.device)
        if self.is_windygrid:
            terminal_reward = torch.Tensor([10000]).to(self.device)
        if s == self.terminal_state:
            return terminal_reward

        expected_reward = torch.Tensor([0]).to(self.device)
        cur_s = torch.zeros(len(self.states)).to(self.device)
        cur_s[s] = 1
        reward_vec = self.reward_lookup[a].to(self.device)
        expected_reward -= self.action_cost
        for d_idx in range(d + 1):
            discount = self.gamma ** d_idx
            expected_reward += (discount * reward_vec * cur_s).sum()
            cur_s = transition_model.get_next_s_dist(a, d_idx, s)
        return expected_reward

    def get_prediction_batch(self, states, actions, deltas, transition_model):  # d can be 0-indexed
        expanded_T = transition_model.get_expanded_T()  # a, d, s1, s2
        I = torch.eye(len(self.states)).unsqueeze(0).unsqueeze(0).to(self.device)
        I = I.repeat(expanded_T.shape[0], 1, 1, 1)
        expanded_T = torch.cat([I, expanded_T], axis=1)
        
        s2_probs = expanded_T[actions, :, states, :]
        B, D_plus_1, S = s2_probs.shape
        
        discounts = torch.Tensor([self.gamma] * D_plus_1) ** torch.arange(D_plus_1)
        discounts = discounts.to(self.device).unsqueeze(0).unsqueeze(-1)
        discounts = discounts.repeat(B, 1, S).to(self.device)
        
        reward_vecs = self.reward_lookup[actions]  # B, S
        if self.terminal_state is not None:
            reward_vecs[:, self.terminal_state] = self.terminal_reward

        reward_vecs = reward_vecs.unsqueeze(1).repeat(1, D_plus_1, 1)
        
        expected_rewards = s2_probs * discounts * reward_vecs
        expected_rewards = torch.cumsum(expected_rewards, dim=1)
        delta_idxs = deltas
        idxs = list(range(len(delta_idxs)))
        expected_rewards = expected_rewards[idxs, delta_idxs, :]        
        expected_rewards = expected_rewards.sum(dim=-1)
        
        costs = torch.Tensor([self.action_cost if s != self.terminal_state else 0 for s in states]).to(self.device)
        expected_rewards = expected_rewards - costs
        return expected_rewards


class RewardModel(nn.Module):  # tabular reward model
    def __init__(self, states, actions, deltas, delta_names,
                 terminal_state, gamma, action_cost, device,
                 input_size, output_size, hidden_size=None,
                 is_windygrid=False):
        super(RewardModel, self).__init__()

        self.states = states
        self.actions = actions
        self.deltas = deltas
        self.delta_names = delta_names
        self.terminal_state = terminal_state
        self.gamma = gamma
        self.action_cost = action_cost
        self.is_windygrid = is_windygrid
        self.device = device

        if hidden_size is not None:
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, output_size),
                torch.nn.ReLU()
            ).to(self.device)
        else:
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.ReLU()
            ).to(self.device)

    def forward(self, x):
        out = self.layers(x)
        return out

    def get_prediction(self, s, a, d, transition_model=None):  # d can be 0-indexed
        x = get_feature_vec(s, a, d, self.states, self.actions, self.deltas)
        r = self.forward(x)
        return r

    def get_prediction_batch(self, states, actions, deltas, transition_model):  # d can be 0-indexed
        predictions = []
        for (s, a, d) in zip(states, actions, deltas):
            predictions.append(self.get_prediction(s, a, d, transition_model))
        return predictions


class OracleRewardModel:
    def __init__(self, env, 
                 states, actions, deltas, delta_names,
                 terminal_state, gamma, action_cost,
                 device,
                 is_windygrid=False):
        super(OracleRewardModel, self).__init__()
        self.states = states
        self.actions = actions
        self.deltas = deltas
        self.delta_names = delta_names

        self.terminal_state = terminal_state
        self.gamma = gamma
        self.action_cost = action_cost
        self.is_windygrid = is_windygrid
        self.device = device
        
        self.delta_names = delta_names
        if 'windygrid' not in env.name.lower():
            self.oracle_transition = OracleTransitionModel(env, actions, states, delta_names, device, dumb=False).to(device)

        self.states = states
        self.actions = actions
        self.delta_names = self.delta_names
        self.S, self.A, self.D = len(states), len(actions), len(delta_names)
        self.env = env

    def get_prediction(self, s, a, d):
        if self.is_windygrid:
            assert 'windygrid' in self.env.name.lower()

        if self.is_windygrid:
            assert self.env.env.goal_loc == self.env.idx_to_state[self.terminal_state]
            assert self.env.terminal_state == self.terminal_state
        
            coords = self.env.idx_to_state[s]
            if coords == self.env.idx_to_state[self.terminal_state]:
                assert(self.env.env.goal_reward == 10000)
                assert(self.env.terminal_reward == 10000)
                return 10000
            elif self.env.env.hazards and coords in self.env.env.hazard_locs:
                assert(self.env.env.hazard_cost == 5)
                return -5
            else:
                return -1

        if s == self.env.terminal_state:
            return self.env.terminal_reward

        expected_reward = 0
        cur_t = 0
        cur_s = torch.zeros(len(self.states)).to(self.device)
        cur_s[s] = 1
        reward_vec = torch.Tensor([self.env.reward_function(st, a) for st in self.states]).to(self.device)
        expected_reward -= self.env.cost
        for d_idx in range(d + 1):
            discount = self.env.gamma ** cur_t
            expected_reward += (discount * reward_vec * cur_s).sum()
            cur_s = self.oracle_transition.get_next_s_dist(a, d_idx, s)
            cur_t += 1
        return expected_reward

"""Transition models."""

import torch
import torch.nn as nn
from utils import get_expanded_T


class TransitionModel(nn.Module):
    def __init__(self, actions, states, delta_names, device, terminal_state, init_T_unconstrained=None):
        super(TransitionModel, self).__init__()

        S = len(states)
        A = len(actions)

        self.states = states
        self.actions = actions
        self.delta_names = delta_names
        self.terminal_state = terminal_state

        self.device = device

        if init_T_unconstrained is None:
            if self.terminal_state is None:
                init_T_unconstrained = torch.ones(A, S, S)
            else:
                init_T_unconstrained = torch.ones(A, S - 1, S)
        self.T_unconstrained = nn.Parameter(init_T_unconstrained.to(device), requires_grad=True)

    def forward(self):
        T = torch.softmax(self.T_unconstrained, dim=2)
        if self.terminal_state is not None:
            terminal = torch.zeros(len(self.actions), 1, len(self.states)).to(self.device)
            terminal[:, :, self.terminal_state] = 1
            T = torch.cat([
                T[:, :self.terminal_state, :], 
                terminal, 
                T[:, self.terminal_state:, :]], axis=1)
        return T

    def get_expanded_T(self):
        T = self.forward()
        A, S1, S2 = T.shape
        assert S1 == S2
        expanded_T = torch.ones(A, S1, S2, len(self.delta_names)).to(self.device)
        for a in range(A):
            for j, k in enumerate(self.delta_names):
                T_delta = torch.matrix_power(T[a, :, :], k)
                expanded_T[a, :, :, j] = T_delta
        expanded_T = expanded_T.permute(0, 3, 1, 2)
        return expanded_T

    def get_next_s_dist(self, action, delta, state):
        delta = self.delta_names[delta]
        T = self.forward()
        T_delta = torch.matrix_power(T[action, :, :], int(delta))
        return T_delta[state]  # distribution over next states

    def get_NLL(self, action, delta, s_start, s_delta):
        s2_dist = self.get_next_s_dist(action, delta, s_start)
        NLL = -torch.log(s2_dist[s_delta])
        return NLL  # can add these up along entire trajectory to get NLL of trajectory

    def get_batched_NLL(self, deltas, actions, state1s, state2s):
        expanded_T = self.get_expanded_T()  # a, d, s1, s2
        probs = expanded_T[actions, deltas, state1s, state2s]
        batched_NLL = -torch.log(probs)
        return batched_NLL
        

    
class DumbTransitionModel(nn.Module):
    def __init__(self, actions, states, delta_names, device, terminal_state, init_T_unconstrained=None):
        super(DumbTransitionModel, self).__init__()

        S = len(states)
        A = len(actions)
        D = len(delta_names)

        self.states = states
        self.actions = actions
        self.delta_names = delta_names
        self.terminal_state = terminal_state

        self.device = device

        if init_T_unconstrained is None:
            init_T_unconstrained = torch.ones(A, D, S - 1, S)
        self.T_unconstrained = nn.Parameter(init_T_unconstrained.to(device), requires_grad=True)

    def forward(self):
        T = torch.softmax(self.T_unconstrained, dim=3)
        if self.terminal_state is not None:
            terminal = torch.zeros(len(self.actions), len(self.delta_names), 1, len(self.states)).to(self.device)
            terminal[:, :, :, self.terminal_state] = 1
            T = torch.cat([
                T[:, :, :self.terminal_state, :], 
                terminal, 
                T[:, :, self.terminal_state:, :]], axis=2)
            # T = torch.cat([T, terminal], axis=2)
        return T

    def get_expanded_T(self):
        return self.forward()

    def get_next_s_dist(self, action, delta, state):
        T = self.forward()
        T_delta = T[action, delta, state, :]
        return T_delta  # distribution over next states

    def get_NLL(self, action, delta, s_start, s_delta):
        s2_dist = self.get_next_s_dist(action, delta, s_start)
        NLL = -torch.log(s2_dist[s_delta])
        return NLL  # can add these up along entire trajectory to get NLL of trajectory

    def get_batched_NLL(self, deltas, actions, state1s, state2s):
        expanded_T = self.get_expanded_T()  # a, d, s1, s2
        probs = expanded_T[actions, deltas, state1s, state2s]
        batched_NLL = -torch.log(probs)
        return batched_NLL


class OracleTransitionModel(nn.Module):
    def __init__(self, env, actions, states, delta_names, device, dumb=False):
        super(OracleTransitionModel, self).__init__()
        self.env = env
        self.actual_T = torch.Tensor(self.env.T)
        T_pred_shape = (len(actions), len(delta_names), len(states), len(states))
        self.expanded_T = get_expanded_T(T_pred_shape, self.actual_T, delta_names)

        if dumb:
            self.T = self.expanded_T
        else:
            self.T = self.actual_T

        self.states = states
        self.actions = actions
        self.delta_names = delta_names

        self.dumb = dumb

    def forward(self):
        return self.T

    def get_next_s_dist(self, action, delta, state):
        T_delta = self.expanded_T[action, delta, state, :]
        return T_delta
        
    def get_NLL(self, action, delta, s_start, s_delta):
        s2_dist = self.get_next_s_dist(action, delta, s_start)
        NLL = -torch.log(s2_dist[s_delta])
        return NLL  # can add these up along entire trajectory to get NLL of trajectory

    def get_expanded_T(self):
        return self.expanded_T
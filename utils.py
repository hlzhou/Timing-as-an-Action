"""Utility functions."""

import numpy.random as npr
import torch


def sample_simplex(size):
    vals = npr.uniform(low=0., high=1., size=size - 1)
    vals = [0] + list(sorted(list(vals))) + [1]
    samples = list([vals[i + 1] - vals[i] for i in range(len(vals) - 1)])
    assert(len(samples) == size)
    return samples


def get_expanded_T(T_pred_shape, T_actual, delta_names, device=None):  # T_pred_shape: (A, D, S, S)
    assert(len(T_pred_shape) == 4)
    if device is None:
        device = torch.device('cpu')
    
    T_actual_expanded = torch.zeros(T_pred_shape).to(device)
    for action in range(T_pred_shape[0]):
        for delta_idx in range(T_pred_shape[1]):
            delta = delta_names[delta_idx]
            T_delta = torch.matrix_power(T_actual[action, :, :], int(delta))
            T_actual_expanded[action, delta_idx, :, :] = T_delta
    return T_actual_expanded


def get_T_L1_error(T_pred, delta_names, env, maximum=False, device=None):
    if device is None:
        device = torch.device('cpu')
    with torch.no_grad():
        T_actual = torch.Tensor(env.T).to(device)
        if len(T_pred.shape) == 4:
            T_actual = get_expanded_T(T_pred.shape, T_actual, delta_names, device=device)
        
        assert(T_pred.shape == T_actual.shape)
        if maximum:
            return abs(T_pred - T_actual).sum(dim=2).max()
        else:
            return abs(T_pred - T_actual).sum(dim=2).mean()

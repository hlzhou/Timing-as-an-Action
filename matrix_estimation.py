"""Experiments for estimating transition matrices."""

from algorithms import TransitionModel, DumbTransitionModel, update_transition_matrix
import numpy as np
import torch
import numpy.random as npr
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.utils import shuffle
import itertools
import time


def get_expanded_T(T, num_deltas):
    A, S1, S2 = T.shape
    assert S1 == S2
    expanded_T = torch.ones(A, S1, S2, num_deltas)
    for a in range(A):
        for k in range(num_deltas):
            T_delta = torch.matrix_power(T[a, :, :], k + 1)
            expanded_T[a, :, :, k] = T_delta
    return expanded_T


def get_empirical_est(true_T, history, delta_names):
    expanded_true_T = get_expanded_T(torch.from_numpy(true_T), len(delta_names))
    empirical_T_cts = torch.from_numpy(np.zeros_like(expanded_true_T))
    for (a, d, s, s2, r) in history:
        empirical_T_cts[a, s, s2, d] += 1
        denom = empirical_T_cts.sum(axis=2)[:, :, np.newaxis, :].repeat(1, 1, empirical_T_cts.shape[1], 1)
        empirical_T = empirical_T_cts / denom
        empirical_T = np.nan_to_num(empirical_T.numpy(), 100)
        empirical_T = torch.from_numpy(empirical_T)
    denom = empirical_T_cts.sum(axis=2)[:, :, np.newaxis, :].repeat(1, 1, empirical_T_cts.shape[1], 1)
    empirical_est_T = empirical_T_cts / np.maximum(denom, 1)
    return empirical_est_T


def get_perfs(expanded_est_T, expanded_true_T, per_action=False):
    if per_action:
        d = {}
        for k in range(expanded_est_T.shape[-1]):
            for a in range(expanded_est_T.shape[0]):
                err = (expanded_true_T[a, :, :, k] - expanded_est_T[a, :, :, k]).abs().sum(dim=-1).max()
                err = err.detach().item()
                d[f'T_L1_maxs (a = {a}, k = {k})'] = err
    else:
        d = {
            f'T_L1_maxs (k = {k})': (expanded_true_T[:, :, :, k] - expanded_est_T[:, :, :, k]).abs().sum(dim=-1).max().detach().item()
            for k in range(expanded_true_T.shape[-1])
        }
    d['expanded_T_L1_means'] = ((expanded_true_T - expanded_est_T).abs().sum(dim=-2).mean()).detach().item() 
    d['expanded_T_L1_maxs'] = ((expanded_true_T - expanded_est_T).abs().sum(dim=-2).max()).detach().item() 
    return d


def run_estimation(
    true_T, delta_names, explore_deltas, Ns, reps, 
    states, actions, device, terminal_state, T_lr, convergence, patience, 
    delta_schedule=None, batchsize=20, exhaustive=True, action_ratio=None, 
    per_action=False, weight_by_actions=False, weight_by_state_action=False):
    start = time.time()
    expanded_true_T = get_expanded_T(torch.from_numpy(true_T), len(delta_names))
    summary = {}
    for i, N in enumerate(Ns):
        if delta_schedule is not None:
            delts = delta_schedule[i]
        else:
            delts = explore_deltas
        print('N: ', N, 'delts: ', delts)
        smart_perfs = {}
        dumb_perfs = {}
        empirical_perfs = {}
        for rep in range(reps):
            print(f'----- N: {N}, rep {rep} ({time.time() - start:.2f} sec) ------')
            dataset = []
            if exhaustive:
                history = itertools.product(states, actions, delts)
                for (s, a, k) in history:
                    for _ in range(N):
                        p = expanded_true_T[a, s, :, k].numpy()
                        assert(abs(sum(p) - 1) <= 0.0001)
        
                        s2 = npr.choice(states, p=p)
                        dataset.append((a, k, s, s2, None))
            else:
                num = len(states) * len(actions) * len(delts) * N
                sampled_s = list(npr.choice(states, size=num, replace=True, p=None))
                sampled_a = list(npr.choice(actions, size=num, replace=True, p=action_ratio))
                sampled_k = list(npr.choice(delts, size=num, replace=True, p=None))
                history = zip(sampled_s, sampled_a, sampled_k)
                for (s, a, k) in history:
                    p = expanded_true_T[a, s, :, k].numpy()
                    assert(abs(sum(p) - 1) <= 0.0001)
    
                    s2 = npr.choice(states, p=p)
                    dataset.append((a, k, s, s2, None))
            dataset = shuffle(dataset)
            
            print('training smart...')
            # smart
            transition = TransitionModel(actions, states, delta_names, device, terminal_state)
            optimizer = torch.optim.SGD(transition.parameters(), lr=T_lr)
            transition, nll_avg = update_transition_matrix(
                dataset, 
                transition, 
                optimizer,
                torch.device('cpu'),
                convergence=convergence, 
                patience=patience, 
                batchsize=batchsize,
                weight_by_actions=weight_by_actions,
                weight_by_state_action=weight_by_state_action) 
            est_T = transition()
            expanded_est_T = get_expanded_T(est_T, len(delta_names))
            for k, v in get_perfs(expanded_est_T, expanded_true_T, per_action=per_action).items():
                smart_perfs[k] = smart_perfs.get(k, []) + [v]
            smart_perfs['expanded_est_T'] = smart_perfs.get('expanded_est_T', []) + [expanded_est_T]

            print('training dumb...')
            # dumb
            dumb_transition = DumbTransitionModel(actions, states, delta_names, device, terminal_state)
            optimizer = torch.optim.SGD(dumb_transition.parameters(), lr=T_lr)
            dumb_transition, nll_avg = update_transition_matrix(
                dataset, 
                dumb_transition, 
                optimizer,
                torch.device('cpu'),
                convergence=convergence, 
                patience=patience, 
                batchsize=batchsize,
                weight_by_actions=weight_by_actions,
                weight_by_state_action=weight_by_state_action) 
            expanded_est_T = dumb_transition()
            expanded_est_T = expanded_est_T.permute(0, 2, 3, 1)
            for k, v in get_perfs(expanded_est_T, expanded_true_T, per_action=per_action).items():
                dumb_perfs[k] = dumb_perfs.get(k, []) + [v]
            dumb_perfs['expanded_est_T'] = dumb_perfs.get('expanded_est_T', []) + [expanded_est_T]
            
            print('training empirical...')
            # empirical
            empirical_T = get_empirical_est(true_T, dataset, delta_names)
            for k, v in get_perfs(empirical_T, expanded_true_T, per_action=per_action).items():
                empirical_perfs[k] = empirical_perfs.get(k, []) + [v]
            empirical_perfs['expanded_est_T'] = empirical_perfs.get('expanded_est_T', []) + [expanded_est_T]

        for mname, perfs in smart_perfs.items():
            if mname == 'expanded_est_T':
                summary['expanded_est_T'] = summary.get('expanded_est_T', []) + [perfs]
            else:
                mean, std = np.mean(perfs), np.std(perfs)
                summary[mname] = summary.get(mname, []) + [{'N': N, 'mean': mean, 'std': std, 'estimator': 'smart'}]
            
        for mname, perfs in dumb_perfs.items():
            if mname == 'expanded_est_T':
                # summary['expanded_est_T'] = perfs
                summary['expanded_est_T'] = summary.get('expanded_est_T', []) + [perfs]
            else:
                mean, std = np.mean(perfs), np.std(perfs)
                summary[mname] = summary.get(mname, []) + [{'N': N, 'mean': mean, 'std': std, 'estimator': 'dumb'}]
            
        for mname, perfs in empirical_perfs.items():
            if mname == 'expanded_est_T':
                # summary['expanded_est_T'] = perfs
                summary['expanded_est_T'] = summary.get('expanded_est_T', []) + [perfs]
            else:
                mean, std = np.mean(perfs), np.std(perfs)
                summary[mname] = summary.get(mname, []) + [{'N': N, 'mean': mean, 'std': std, 'estimator': 'empirical'}]    
    print('done')

    final_Ts = {
        'smart': transition,
        'dumb': dumb_transition,
        'empirical': empirical_T, 
    }

    return summary

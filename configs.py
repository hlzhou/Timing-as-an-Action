"""Configurations for the experiments."""

import torch


stronger_tradeoff = {
    'prev_dir': None,
    'tag': 'stronger_tradeoff',
    'epochs': 200,
    'greedy_delay': 'uniform',          # 0
    'oracle_reward': False,
    'oracle_transition': False,
    'action_cost': 1,
    'epsilon': 0.1,
    'gamma': 0.9,
    'Q_iters': 50,
    'condensed_reward': False,
    'horizon': 50,
    'action_names': ['a0', 'a1'],
    'state_names': ['s0', 's1', 's2'],
    'delta_names': list(range(1, 11)),
    'sgd_update': False,
    'sgd_lr': 0.3,
    'env': 'stronger_tradeoff',
    'seed': 0,
    'state_p0': [0.8, 0.2, 0],
    'learning_rate': 1e-2,
    'transition_convergence': 1e-5,
    'transition_patience': 3,
    'device': torch.device('cpu'),  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    'exploration_phase': 0,
}


custom_windygrid = {
    'prev_dir': None,
    'tag': 'custom_windygrid',
    'epochs': 2000,
    'greedy_delay': 'uniform',          # 0
    'oracle_reward': False,
    'oracle_transition': False,
    'action_cost': 0,
    'epsilon': 0.1,
    'gamma': 0.99,
    'Q_iters': 100,
    'condensed_reward': False,
    'horizon': None,
    'action_names': [],
    'state_names': [],
    'delta_names': list(range(1, 11)),
    'sgd_update': False,
    'sgd_lr': 0.3,
    'env': 'custom_windygrid',
    'seed': 0,
    'state_p0': None,
    'learning_rate': 1e-3,
    'device': torch.device('cpu'),
    'exploration_phase': 0,
}

glucose = {
    'prev_dir': None,
    'tag': 'glucose',
    'epochs': 2000,                     # 10000
    'greedy_delay': 'uniform',          # 0
    'oracle_reward': False,
    'oracle_transition': False,
    'action_cost': 0.5,
    'epsilon': 0.1,
    'gamma': 0.99,
    'Q_iters': 100,
    'condensed_reward': False,
    'horizon': 30,
    'action_names': [],
    'state_names': [],
    'delta_names': list(range(1, 5)),
    'sgd_update': False,
    'sgd_lr': 0.3,
    'env': 'glucose',
    'seed': 0,
    'state_p0': None,
    'learning_rate': 1e-3,
    'device': torch.device('cpu'),
    'exploration_phase': 0,
}



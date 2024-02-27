"""Visualize various components of the model-based RL experiments."""

import matplotlib.pyplot as plt
import pandas as pd
from algorithms import *
import pandas as pd
import torch
from matplotlib.ticker import ScalarFormatter


def get_actions_taken_dist(actions_taken, actions, deltas, ad_combos):
    a_dist = {}
    d_dist = {}
    for ((a, d), ct) in actions_taken.items():
        a_dist[a] = a_dist.get(a, 0) + ct
        d_dist[d] = d_dist.get(d, 0) + ct

    a_dist = [a_dist.get(a, 0) for a in actions]
    d_dist = [d_dist.get(d, 0) for d in deltas]
    ad_dist = [actions_taken.get((a, d), 0) for (a, d) in ad_combos]

    a_dist = pd.DataFrame({'actions': actions, 'cts': a_dist})
    d_dist = pd.DataFrame({'delays': deltas, 'cts': d_dist})
    ad_dist = pd.DataFrame({'(a, d)': ad_combos, 'cts': ad_dist})

    return a_dist, d_dist, ad_dist

def plot_action_dists(d, actions, deltas, mode, env):  
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    num_epochs = len(d['action_dists_taken'])
    action_cts = np.zeros((len(actions), num_epochs))
    delta_cts = np.zeros((len(deltas), num_epochs))
    for epoch, action_d in enumerate(d['action_dists_taken']):
        for ((a, delt), ct) in action_d.items():
            action_cts[a, epoch] = action_cts[a, epoch] + ct
            delta_cts[delt, epoch] = delta_cts[delt, epoch] + ct

    if 'action_names' in env.__dict__.keys():
        action_names = env.action_names
    else:
        action_names = actions

    ax[0].stackplot(range(num_epochs), action_cts / action_cts.sum(axis=0), labels=action_names)
    ax[0].set_title(f'{mode}, action dist')
    
    ax[1].stackplot(range(num_epochs), delta_cts / delta_cts.sum(axis=0), labels=deltas)
    ax[1].set_title(f'{mode}, delta dist')
              
    ax[0].legend(bbox_to_anchor=(1.04,1), loc="upper left")
    ax[1].legend(bbox_to_anchor=(1.04,1), loc="upper left")
    
    plt.tight_layout()
    return fig

def plot_Q(Q, states, actions, deltas, mode, env):
    if 'windygrid' in env.name.lower() and len(states) == 70:
        fig, ax = plt.subplots(7, 10, figsize=(6 * 10, 3 * 7))
    else:
        fig, ax = plt.subplots(1, len(states), figsize=(6 * len(states), 3))
    for s in states:
        s_idx = s
        for a in actions:
            values = []
            for d in deltas:
                Q_val = Q[a, s, d]
                values.append(Q_val)
            if len(states) == 70:
                x0, x1 = env.idx_to_state[s]
                assert(x0 == int(s / 10))
                assert(x1 == int(s % 10))
                s_idx = (x0, x1)
            ax[s_idx].plot(deltas, values, 'o-', label=f'Action{a}')
        ax[s_idx].set_title(f'{mode}, State {s_idx}')
        ax[s_idx].set_xlabel('Delta')
        ax[s_idx].set_ylabel('Q Value')
    ax[s_idx].legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.tight_layout()
    return fig

def plot_R(R, states, actions, deltas, mode, transition, env):
    if len(states) == 70:
        fig, ax = plt.subplots(7, 10, figsize=(5 * 10, 3 * 7))
    else:
        fig, ax = plt.subplots(1, len(states), figsize=(6 * len(states), 3))
    for s in states:
        s_idx = s
        for a in actions:
            rewards = []
            for d in deltas:
                if 'condensed' in str(R).lower():
                    r_pred = R.get_prediction(s, a, d, transition)
                else:
                    r_pred = R.get_prediction(s, a, d)
                rewards.append(r_pred)
            if len(states) == 70:
                x0, x1 = env.idx_to_state[s]
                assert(x0 == int(s / 10))
                assert(x1 == int(s % 10))
                s_idx = (x0, x1)
            ax[s_idx].plot(deltas, rewards, 'o-', label=f'Action{a}')
        ax[s_idx].set_title(f'{mode}, State {s_idx}')
        ax[s_idx].set_xlabel('Delta')
        ax[s_idx].set_ylabel('Reward')
    ax[s_idx].legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.tight_layout()
    return fig


def plot_windygrid_transition(env, T):
    if len(T.shape) == 3:
        A, S, _ = T.shape
        D = 1
    elif len(T.shape) == 4:
        A, D, S, _ = T.shape
    else:
        import pdb; pdb.set_trace()

    fig, axes = plt.subplots(D, A, figsize=(A*10, D*7))

    colors = [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
        'gold', 'darkviolet', 'fuchsia', 'red', 'blue', 'chartreuse'
    ] * 10
    vector_list = []
    for a in range(A):
        for d in range(D):
            if D == 1:
                ax = axes[a]
                T_a = T[a]
            else:
                ax = axes[d, a]
                T_a = T[a, d]
        
            for from_idx in range(S):
                for to_idx in range(S):
                    try:
                        if T_a[from_idx, to_idx] > 0:
                            scale = T_a[from_idx, to_idx] * 0.8
                            coords_from = env.idx_to_state[from_idx]
                            coords_to = env.idx_to_state[to_idx]
                            
                            x, y = coords_from[1], env.env.height - 1 - coords_from[0]

                            vector = (
                                max(min(coords_to[1] - coords_from[1], 3), -3), 
                                max(min(-(coords_to[0] - coords_from[0]), 3), -3)
                            )
                            # assert (np.linalg.norm(vector) <= 3)
                            
                            if vector not in vector_list:
                                vector_list.append(vector)
                            color = colors[vector_list.index(vector)]
                            
                            ax.arrow(x, y, scale * vector[0], scale * vector[1], head_width=0.1, color=color, label=str(vector))
                            ax.plot(x, y, 'o', label=str(vector), color=color)
                    except Exception as e:
                        import pdb; pdb.set_trace()
                        print(e)
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.05),
                ncol=4, fancybox=True, shadow=True)

        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 7)
    return fig


def plot_windygrid_Q(deltas, Q, env):
    fig, ax = plt.subplots()
    heatmap = np.zeros((7, 10))
    action_map = {}
    for s in env.states:
        coords = env.idx_to_state[s]
        Q_vals = Q[:, s, :]
        best = (Q_vals>=torch.max(Q_vals)).nonzero()
        for d1 in range(best.shape[0]):
            max_a, max_d = best[d1].detach().flatten()
            move = env.idx_to_action[max_a.item()]
            heatmap[coords[0]][coords[1]] = Q_vals.max()
            action_map[(coords[0], coords[1])] = action_map.get((coords[0], coords[1]), []) + [[env.move_to_dir[move], max_d]]
    vmin = heatmap.min()
    vmax = heatmap.max()
    for i in range(7):
        for j in range(10):
            if (i, j) == (3, 7):
                ax.scatter(7, 3, s=320, marker='*', color='black')
                continue
            if (i, j) in [(2, 5), (2,4)]:
                ax.scatter(j, i, s=80, marker='x', color='tab:red')
            for (direction, delay) in action_map[(i,j)]:
                direction = direction[1], direction[0]
                ax.arrow(j, i, direction[0] /4, direction[1]/4, head_width=0.1, head_length=0.1)
                if list(direction) == [0, 1]:  # DOWN
                    ax.annotate(f'{int(delay) + 1}',
                                xy=(j + direction[0] /8, i + direction[1]/8),
                                xytext=(3, -6),
                                textcoords='offset points')
                elif list(direction) == [1, 0]:  # RIGHT
                    ax.annotate(f'{int(delay) + 1}',
                                xy=(j + direction[0] /8, i + direction[1]/8),
                                xytext=(-2, 5),
                                textcoords='offset points')
                elif list(direction) == [-1, 0]:  # LEFT
                    ax.annotate(f'{int(delay) + 1}',
                                xy=(j + direction[0] /8, i + direction[1]/8),
                                xytext=(-5, -10),
                                textcoords='offset points')
                elif list(direction) == [0, -1]:  # UP
                    ax.annotate(f'{int(delay) + 1}',
                                xy=(j + direction[0] /8, i + direction[1]/8),
                                xytext=(-10, 0),
                                textcoords='offset points')
                else:
                    print('MISSING: ', direction)
    ax.set_title(f'Q-values', fontsize=16)
    im = ax.imshow(heatmap, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    fig.subplots_adjust(right=0.8)
    plt.subplots_adjust(hspace=0.9)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0,0))
    cbar = fig.colorbar(im, cax=cbar_ax, format=formatter)
    cbar.ax.set_ylabel('Q values', fontsize=14)
    return fig

def plot_windygrid_R(deltas, R, env, transition, gamma, action_cost, terminal_state):
    fig, ax = plt.subplots(1, len(deltas), figsize=(40, 3))
    heatmap = np.zeros((len(deltas), 7, 10))
    dirmap = np.zeros((len(deltas), 7, 10, 2))
    for s in env.states:
        coords = env.idx_to_state[s]
        for d in deltas:
            if 'condensedreward' in str(R).lower():
                r_pred = list([R.get_prediction(s, a, d, transition) for a in env.actions])
            else:
                r_pred = list([R.get_prediction(s, a, d) for a in env.actions])
            max_a = np.argmax(r_pred)
            move = env.action_names[max_a]
            heatmap[d][coords[0]][coords[1]] = max(r_pred)
            dirmap[d][coords[0]][coords[1]] = env.move_to_dir[move]
    for delt in deltas:
        for i in range(7):
            for j in range(10):
                ax[delt].arrow(j, i, dirmap[delt][i][j][0] /4, dirmap[delt][i][j][1]/4, head_width=0.1, head_length=0.1)
        ax[delt].set_title(f'R, d = {delt}')
        ax[delt].imshow(heatmap[delt])
    plt.tight_layout()
    return fig
    

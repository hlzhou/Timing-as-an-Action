"""Timing-as-an-action algorithms."""

import torch
import numpy as np
import numpy.random as npr
from sklearn.utils import shuffle
import itertools
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import pickle
import wandb
from viz import plot_action_dists, plot_Q, plot_R, plot_windygrid_transition, plot_windygrid_Q, plot_windygrid_R
from reward_model import RewardModel, OracleRewardModel, CondensedRewardModel
from transition_model import TransitionModel, DumbTransitionModel, OracleTransitionModel
import pandas as pd
from utils import get_expanded_T, get_T_L1_error
import time
from sklearn.utils import class_weight


def epsilon_greedy(s, Q, epsilon=0.5, default_delta=0, deltas=None):
    if npr.uniform(low=0.0, high=1.0) > epsilon:    
        Q_s = Q[:, s, :]
        idxs = list(torch.where(Q_s == Q_s.max()))
        try:
            if idxs[0].shape[0] > 1:
                i = int(npr.choice(range(len(idxs))))  # choose uniformly among best Q's
                return (idxs[0][i].item(), idxs[1][i].item())
            else:
                return (idxs[0].item(), idxs[1].item())
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()
    else:
        eps_action = int(npr.choice(range(Q.shape[0])))
        
        if default_delta == 'uniform':
            eps_delta = npr.choice(deltas)
        else:
            eps_delta = default_delta  # smallest delta
        return [eps_action, eps_delta]


def update_model_free_Q(Q, history, delta_names, gamma, 
                        sgd_update=False, alpha=0.5):
    """Update based on observed reward after delta."""
    for (a, d, s, s2, r) in history:
        with torch.no_grad():
            delta = delta_names[d]
            new_Q = r + (gamma ** d) * Q[:, s2, :].max()
            if sgd_update:
                Q[a, s, d] = Q[a, s, d] + alpha * (new_Q - Q[a, s, d])
            else:
                Q[a, s, d] = new_Q
    return Q


def update_smart_timing_based_Q(Q, R, T, actions, deltas, delta_names, states, gamma, device,
                                cost=5, Q_iters=50, sgd_update=False, alpha=0.5, 
                                condensed_reward=False, terminal_state=2, t=None, 
                                transition=None, verbose=False, is_windygrid=False):
    """Update Q based on predictions from R.

    Q_iters: number of iterations of Q updates. If None, iterates to convergence.
    cost: cost of performing an action. if unknown, cost = 0.

    run just this w/ oracle until it's converged, then take & see that it does at least as well as model free
    """
    prev_Q = Q.clone().detach()
    converged = 0
    with torch.no_grad():
        for i in range(Q_iters):
            for (s, a, d) in itertools.product(states, actions, list(reversed(deltas))):  # updates along the way; could freeze instead
                if condensed_reward:
                    r = R.get_prediction(s, a, 0, transition)
                else:
                    r = R.get_prediction(s, a, 0)
                if delta_names[d] <= 1:
                    expected_Q = T[a, s, :].dot(Q.max(axis=0).values.max(axis=-1).values)
                    new_Q = r + (gamma * expected_Q)
                    if sgd_update:
                        Q[a, s, d] = Q[a, s, d] + alpha * (new_Q - Q[a, s, d])
                    else:
                        Q[a, s, d] = new_Q
                else:
                    dminus1_idx = delta_names.index(delta_names[d] - 1)
                    expected_Q = T[a, s, :].dot(Q[a, :, dminus1_idx])
                    if s == terminal_state:
                        cost_adjustment = torch.Tensor([0]).to(device)
                    else:
                        if terminal_state is not None:
                            adj = T[a, s, terminal_state]
                        else:
                            adj = 0
                        cost_adjustment = torch.Tensor([cost * gamma * (1 - adj)]).to(device)
                    new_Q = r + cost_adjustment + (gamma * expected_Q)
                    if sgd_update:
                        Q[a, s, d] = Q[a, s, d] + alpha * (new_Q - Q[a, s, d])
                    else:
                        Q[a, s, d] = new_Q
            if Q.equal(prev_Q):
                converged += 1
            else:
                converged = 0

            if converged > 3:
                if verbose:
                    print(f'Q converged after {i} iterations')
                break
            
            prev_Q = Q.clone().detach()
    return Q

def update_dumb_model_based_Q(Q, R, T, actions, deltas, delta_names, states, gamma,
                              cost=5, Q_iters=15,
                              condensed_reward=False, 
                              terminal_state=2, t=None, 
                              transition=None, 
                              is_windygrid=False):
    prev_Q = Q.clone().detach()
    converged = 0
    with torch.no_grad():
        for i in range(Q_iters):
            for (s, a, d) in itertools.product(states, actions, deltas):
                if condensed_reward:
                    r = R.get_prediction(s, a, d, transition)
                else:
                    r = R.get_prediction(s, a, d)
                
                expected_Q = T[a, d, s, :].dot(Q.max(axis=0).values.max(axis=-1).values)
                Q[a, s, d] = r + ((gamma**(delta_names[d])) * expected_Q)

            if Q.equal(prev_Q):
                converged += 1
            else:
                converged = 0

            if converged > 3:
                print(f'Q converged after {i} iterations')
                break
            
            prev_Q = Q.clone().detach()

    return Q


def train_to_convergence(history, model, loss_batch_func, optimizer, device,
                         convergence=None, 
                         patience=2, 
                         batchsize=20,
                         max_iters=1000,
                         weight_by_actions=False,
                         weight_by_state_action=False,
                         eval_only=False):
    
    if weight_by_actions:
        actions = [h[0] for h in history]
        unique_actions = list(np.unique(actions))
        weight_lookup = class_weight.compute_class_weight('balanced', unique_actions, actions)

        action_weights = [weight_lookup[unique_actions.index(a)] for a in actions]
    elif weight_by_state_action:
        state_actions = list([(s, a) for (a, d, s, s2, r) in history])
        unique_state_actions = list(set(state_actions))

        state_actions = list([unique_state_actions.index(sa) for sa in state_actions])
        unique_state_actions = range(len(unique_state_actions))
        
        sa_weight_lookup = class_weight.compute_class_weight('balanced', unique_state_actions, state_actions)
        action_weights = [sa_weight_lookup[sa] for sa in state_actions]
    else:
        action_weights = [1 for _ in range(len(history))]
    if batchsize is None or batchsize > len(history):
        batchsize = len(history)
    prev_loss_avg = 1e9
    patience_ct = 0
    itr_ct = 0
    while True:
        if itr_ct > max_iters:
            break
        # print(f'iter: {itr_ct}, {prev_loss_avg}')
        
        if not eval_only:
            optimizer.zero_grad()
        loss_total = 0
        shuffled_history, shuffled_weights = shuffle(history, action_weights)
        for i in range(0, len(shuffled_history), batchsize):
            batch = shuffled_history[i:i+batchsize]
            batch_weights = torch.Tensor(shuffled_weights[i:i+batchsize]).to(device)
            deltas = [] 
            actions = [] 
            state1s = [] 
            state2s = []
            rewards = []
            for (a, d, s, s2, r) in batch:
                deltas.append(d)
                actions.append(a)
                state1s.append(s)
                state2s.append(s2)
                rewards.append(r)

            loss_batch = loss_batch_func(model, deltas, actions, state1s, state2s, rewards)
            try:
                loss_batch = loss_batch * batch_weights
            except Exception as e:
                import pdb; pdb.set_trace()
                print(e)

            if not eval_only:
                loss_batch.sum().backward()
                optimizer.step()
                optimizer.zero_grad()

            loss_total += loss_batch.sum().cpu().detach().item()
        
        loss_avg = loss_total / len(history)
        
        if (not eval_only) and (convergence is not None):
            if prev_loss_avg - loss_avg < convergence:
                patience_ct += 1
            else:
                patience_ct = 0
        else:
            break
        if patience_ct == patience:
            break
        prev_loss_avg = loss_avg
        itr_ct += 1
#         print(transition())
    print(f'dataset of size {len(history)} converged after {itr_ct} iters ({str(model)})')
    return model, loss_avg


def update_transition_matrix(history, transition, optimizer, device,
                             convergence=None, 
                             patience=2, 
                             batchsize=20,
                             max_iters=1000,
                             weight_by_actions=False,
                             weight_by_state_action=False):
    
    def get_nll_batch(model, deltas, actions, state1s, state2s, rewards):
        return model.get_batched_NLL(deltas, actions, state1s, state2s)
    
    transition, nll_avg = train_to_convergence(
        history, transition, get_nll_batch, optimizer, device,
        convergence=convergence, 
        patience=patience, 
        batchsize=batchsize,
        max_iters=max_iters,
        weight_by_actions=weight_by_actions,
        weight_by_state_action=weight_by_state_action)
    return transition, nll_avg
    

def update_reward_model(history, reward_model, r_loss_func, optimizer, device,
                        convergence=None, 
                        patience=2, 
                        batchsize=20,
                        max_iters=1000,
                        transition_model=None,
                        weight_by_actions=False,
                        weight_by_state_action=False,
                        eval_only=False):
    def reward_loss_batch_func(model, deltas, actions, state1s, state2s, rewards):
        preds = model.get_prediction_batch(state1s, actions, deltas, transition_model=transition_model)
        loss = r_loss_func(preds, torch.Tensor(rewards).to(device))
        return loss
    
    reward_model, loss_avg = train_to_convergence(
        history, reward_model, reward_loss_batch_func, optimizer, device,
        convergence=convergence, 
        patience=patience, 
        batchsize=batchsize,
        max_iters=max_iters,
        eval_only=eval_only)
    return reward_model, loss_avg
    

def run_experiment(env, actions, states, delta_names, device,
                   seed=0, horizon=None, epsilon=0.5, greedy_delay=0,
                   epochs=500, T_lr=1e-3, R_lr=0.1, gamma=1, Q_iters=15,
                   prev_transition=None, prev_R=None, prev_Q=None, folder=None, 
                   Q_update_model='timing_smart', 
                   oracle_reward=False, oracle_transition=False, 
                   action_cost=5, sgd_update=False, sgd_lr=0.1,
                   reward_hidden_size=20, condensed_reward=False,
                   exploration_phase=0, Q_update_frequency=1,
                   convergence=1e-5, patience=3, max_iters=1000,
                   model_batchsize=20, oracle_reward_lookup=True,
                   pure_reward_update=True,
                   weight_by_actions=False,
                   exp_dir=None):
    """
    state_p0: initial probabilities of starting in each state
    device: device to perform computations on
    seed: random seed
    horizon: maximum episode length
    epsilon: epsilon-greedy probability of exploration
    epochs: number of epochs
    T_lr: learning rate for transition model
    R_lr: learning rate for reward model
    gamma: decay rate
    Q_iters: number of Q iterations 
    pure_reward_update: whether just to update the reward model or also update transitions when computing reward loss
    """
    start_time = time.time()
    def tprint(*msg):
        elapsed = time.time() - start_time
        msg = '\t'.join([str(m) for m in msg])
        print(f"[{elapsed:.2f} sec] {msg}")

    deltas = list(range(len(delta_names)))
    if isinstance(epsilon, float):
        eps_range = [epsilon, epsilon]
    else:
        eps_range = epsilon

    terminal_state = env.terminal_state
    is_windygrid = ('windygrid' in env.name)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    dumb = (Q_update_model == 'timing_dumb')
    if prev_transition is None:
        if oracle_transition:
            transition = OracleTransitionModel(env, actions, states, delta_names, device, dumb=dumb)
        elif Q_update_model == 'timing_dumb':
            transition = DumbTransitionModel(actions, states, delta_names, device, terminal_state).to(device)
        else:
            transition = TransitionModel(actions, states, delta_names, device, terminal_state).to(device)
    else:
        transition = prev_transition

    if prev_R is None:
        if oracle_reward:
            R = OracleRewardModel(
                env, 
                states, actions, deltas, delta_names,
                terminal_state, gamma, action_cost,
                device,
                is_windygrid=is_windygrid)
        elif condensed_reward:
            R = CondensedRewardModel(
                states, actions, deltas, delta_names,
                terminal_state, gamma, action_cost, device,
                is_windygrid=is_windygrid, env=env, oracle_reward_lookup=oracle_reward_lookup).to(device)
        else:
            input_size = len(actions) + len(states) + len(deltas)
            output_size = 1
            R = RewardModel(
                states, actions, deltas, delta_names,
                terminal_state, gamma, action_cost, device,
                input_size, output_size, hidden_size=reward_hidden_size,
                is_windygrid=is_windygrid).to(device)
    else:
        R = prev_R.to(device)

    if prev_Q is None:
        Q = torch.zeros((len(actions), len(states), len(deltas))).to(device)
    else:
        Q = prev_Q.to(device)

    if not oracle_transition:
        optimizer = torch.optim.Adam(transition.parameters(), lr=T_lr)
    
    if (not oracle_reward) and (not oracle_reward_lookup):
        r_optimizer = torch.optim.Adam(R.parameters(), lr=R_lr)
    
    r_loss_func = torch.nn.MSELoss()

    avg_R_mses = []
    avg_T_nlls = []
    max_T_l1s = []
    mean_T_l1s = []
    ep_lens = []
    total_rewards = []
    action_dists_taken = []
    history = []
    for epoch in tqdm(range(epochs)):
        if epoch % 20 == 0:
            tprint('T: ', transition())
        
        epsilon = eps_range[0] + ((eps_range[1] - eps_range[0]) * (epoch / epochs))
        print('epsilon: ', epsilon)

        s = env.reset()
        terminated = False
        t = 0
        R_mses = []
        T_nlls = []
        total_reward = 0
        actions_taken = {}
        a_ct = 0  # number of actions taken
        while not terminated:
            if epoch < exploration_phase:
                a = npr.choice(actions)
                d = 0
            else:
                a, d = epsilon_greedy(s, Q, epsilon=epsilon, default_delta=greedy_delay, deltas=deltas)

            s2, r, steps_taken, terminated = env.step(a, delta_names[d], t)
            total_reward += (gamma ** t) * r 

            ### Update transition matrix
            if not d == steps_taken - 1:
                if env.name != 'glucose':
                    assert (s2 == env.terminal_state) or (t + delta_names[d] >= env.horizon)
            d = max(steps_taken - 1, 0)
            actions_taken[(a, d)] = actions_taken.get((a, d), 0) + 1
            history.append((a, d, s, s2, r))
            if Q_update_model in ['timing_smart', 'timing_dumb', 'timing_smart_and_model_free']:
                nll = transition.get_NLL(a, d, s, s2)
                if not oracle_transition:
                    transition, nll_avg = update_transition_matrix(
                        history, transition, optimizer, device,
                        convergence=convergence, 
                        patience=patience, 
                        batchsize=model_batchsize,
                        max_iters=max_iters,
                        weight_by_actions=weight_by_actions)
                T_nlls.append(nll.item())
                        
            ### Update reward model
            # freeze transition model
            if pure_reward_update and (Q_update_model in ['timing_smart', 'timing_dumb']):
                for param in transition.parameters():
                    param.requires_grad = False

            if oracle_reward:
                with torch.no_grad():
                    mse_total = 0
                    for (h_a, h_d, h_s, h_s2, h_r) in shuffle(history):
                        r_pred = R.get_prediction(h_s, h_a, h_d)
                        r_loss = r_loss_func(torch.Tensor([r_pred]).to(device), torch.Tensor([h_r]).to(device))
                        mse_total += r_loss.item()
                R_mses.append(mse_total / float(len(history)))
            elif oracle_reward_lookup:
                with torch.no_grad():
                    mse_total = 0
                    delta_hist = [] 
                    action_hist = [] 
                    state1_hist = [] 
                    state2_hist = []
                    reward_hist = []
                    for (a, d, s, s2, r) in history:
                        delta_hist.append(d)
                        action_hist.append(a)
                        state1_hist.append(s)
                        state2_hist.append(s2)
                        reward_hist.append(r)
                    preds = R.get_prediction_batch(state1_hist, action_hist, delta_hist, transition)
                    loss = r_loss_func(preds, torch.Tensor(reward_hist).to(device))
                    R_mses.append(loss.mean().item())
            else:
                tprint('updating reward model...')
                R, avg_r_loss = update_reward_model(
                    history, R, r_loss_func, r_optimizer, device,
                    convergence=convergence, 
                    patience=patience, 
                    batchsize=model_batchsize,
                    max_iters=max_iters,
                    transition_model=transition,
                    weight_by_actions=weight_by_actions)
                R_mses.append(avg_r_loss)

            # unfreeze transition model
            if pure_reward_update and (Q_update_model in ['timing_smart', 'timing_dumb']):
                for param in transition.parameters():
                    param.requires_grad = True

            # update Q
            with torch.no_grad():
                T = transition()
                if (epoch % Q_update_frequency == 0) and (a_ct == 0):  # update Q values every epoch
                    tprint(f'updating Q... ({Q_update_model})')
                    if Q_update_model == 'timing_smart':
                        Q = update_smart_timing_based_Q(Q, R, T, 
                                                        actions, deltas, delta_names, states, gamma, device,
                                                        cost=env.cost, Q_iters=Q_iters, sgd_update=sgd_update, alpha=sgd_lr,
                                                        condensed_reward=condensed_reward, terminal_state=terminal_state, t=t, transition=transition,
                                                        is_windygrid=is_windygrid,
                                                        verbose=True)
                    elif Q_update_model == 'timing_dumb':
                        Q = update_dumb_model_based_Q(Q, R, T, 
                                                    actions, deltas, delta_names, states, gamma, 
                                                    cost=env.cost, Q_iters=Q_iters,
                                                    condensed_reward=condensed_reward, terminal_state=terminal_state, t=t, transition=transition,
                                                    is_windygrid=is_windygrid)
                    elif Q_update_model == 'model_free': 
                        # feed in history for experience replay
                        Q = update_model_free_Q(Q, history, delta_names, gamma, 
                                                sgd_update=sgd_update, alpha=sgd_lr)
                    elif Q_update_model == 'timing_smart_and_model_free':
                        Q = update_model_free_Q(Q, s, s2, a, d, r, delta_names, gamma, 
                                                sgd_update=sgd_update, alpha=sgd_lr)
                        Q = update_smart_timing_based_Q(Q, R, T, 
                                actions, deltas, delta_names, states, gamma, device,
                                cost=env.cost, Q_iters=Q_iters, sgd_update=sgd_update, alpha=sgd_lr,
                                condensed_reward=condensed_reward, terminal_state=terminal_state, t=t, transition=transition,
                                is_windygrid=is_windygrid)
                    tprint(f'done updating Q.')

            t += steps_taken
            a_ct += 1
            s = s2
            if terminated or t >= horizon:
                terminated = True

        if 'grid' in env.name:
            action_cts = {}
            heatmap_s = np.zeros((7, 10))
            heatmap_s2 = np.zeros((7, 10))
            for (a, d, s, s2, r) in history:
                a_dict = action_cts.get('a', {})
                a_dict[a] = a_dict.get(a, 0) + 1

                d_dict = action_cts.get('d', {})
                d_dict[d] = a_dict.get(d, 0) + 1
                
                coords = env.idx_to_state[s]
                heatmap_s[coords[0], coords[1]] += 1
                coords = env.idx_to_state[s2]
                heatmap_s2[coords[0], coords[1]] += 1
                
                action_cts['a'] = a_dict
                action_cts['d'] = d_dict
                
        # update error metrics each epoch
        metrics = {}
        if hasattr(env, 'T') and env.T is not None:
            with torch.no_grad():
                T = transition()
                mean_l1_error = get_T_L1_error(T, delta_names, env, maximum=False, device=device)
                mean_T_l1s.append(mean_l1_error)

                max_l1_error = get_T_L1_error(T, delta_names, env, maximum=True, device=device)
                max_T_l1s.append(max_l1_error)
                
                metrics['Transition Max L1 Error'] = max_l1_error
                metrics['Transition Mean L1 Error'] = mean_l1_error
        
        avg_R_mses.append(np.mean(R_mses))
        avg_T_nlls.append(np.mean(T_nlls))  
        ep_lens.append(t)
        total_rewards.append(total_reward)
        action_dists_taken.append(actions_taken)
        
        metrics['Reward Avg MSE'] = np.mean(R_mses)
        if condensed_reward:
            tprint('Reward lookup: ', R.reward_lookup.cpu().detach().numpy())
            metrics['Reward Lookups'] = R.reward_lookup.cpu().detach().numpy()
        metrics['Transition Avg NLL'] = np.mean(T_nlls)
        metrics['Episode Length'] = t
        metrics['Cumulative Episode Lengths'] = sum(ep_lens)
        metrics['Per-Episode Reward'] = total_reward
        metrics['Cumulative Reward'] = sum(total_rewards)
        wandb.log(metrics)

        d = {
            'avg_R_mses': avg_R_mses,
            'avg_T_nlls': avg_T_nlls,
            'max_T_l1s': max_T_l1s,
            'mean_T_l1s': mean_T_l1s,
            'ep_lens': ep_lens,
            'total_rewards': total_rewards,
            'action_dists_taken': action_dists_taken,
        }

        if exp_dir:
            with open(f'{exp_dir}/transition_e{epoch}.pkl', 'wb') as fout:
                pickle.dump(transition, fout)
            
            with open(f'{exp_dir}/R_e{epoch}.pkl', 'wb') as fout:
                pickle.dump(R.reward_lookup, fout)

            with open(f'{exp_dir}/Q_e{epoch}.pkl', 'wb') as fout:
                pickle.dump(Q, fout)

            with open(f'{exp_dir}/d_e{epoch}.pkl', 'wb') as fout:
                pickle.dump(d, fout)
        print('saved checkpoints')

        if (epoch % 20 == 0) or (epoch == epochs - 1):
            tprint('plotting...')
            with torch.no_grad():
                action_dist_fig = plot_action_dists(d, actions, deltas, Q_update_model, env)
                Q_fig = plot_Q(Q, states, actions, deltas, Q_update_model, env)
                R_fig = plot_R(R, states, actions, deltas, Q_update_model, transition, env)
                figlog = {
                    'Distribution of Actions': wandb.Image(action_dist_fig),
                    'Q': wandb.Image(Q_fig),
                    'R': wandb.Image(R_fig),
                    'T': pd.DataFrame({'Transition Matrix': [str(T)]}),
                }
                if 'windygrid' in env.name.lower():
                    windyT = plot_windygrid_transition(env, T.cpu().detach().numpy())
                    windyQ = plot_windygrid_Q(deltas, Q, env)
                    windyR = plot_windygrid_R(deltas, R, env, transition, gamma, action_cost, terminal_state)
                    figlog['windyT'] = wandb.Image(windyT)
                    figlog['windyQ'] = wandb.Image(windyQ)
                    figlog['windyR'] = wandb.Image(windyR)

                wandb.log(figlog)

                plt.close('all')
            tprint('done plotting.')
    
    assert(len(avg_R_mses) == len(avg_T_nlls))
    assert(len(ep_lens) == len(avg_T_nlls))

    tprint('-------- Learned T matrix --------')
    tprint(transition())

    return transition, R, Q, d

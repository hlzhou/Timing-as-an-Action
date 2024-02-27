"""Main file for running experiments."""

import os
import pickle
import wandb 
from datetime import datetime

from gyms.sim_library import get_stronger_tradeoff_simulator, get_custom_windy_gridworld_simulator, get_glucose_simulator
from algorithms import run_experiment
from configs import stronger_tradeoff, custom_windygrid, glucose

import argparse


def run(prev_dir=None, tag='simple',
        epochs=10000, greedy_delay=0,
        oracle_reward=False, oracle_transition=False, action_cost=5, 
        epsilon=0.3, gamma=0.8, Q_iters=100, condensed_reward=False, horizon=None,
        action_names=None, state_names=None, delta_names=None, 
        sgd_update=False, sgd_lr=0.3,
        env='simple_simulator', 
        seed=0, state_p0=None,
        learning_rate=1e-3,
        device=None,
        exploration_phase=0,
        reward_hidden_size=20,
        Q_update_frequency=1,
        transition_convergence=1e-5,
        transition_patience=3,
        model_batchsize=20,
        T_lr=1e-3, R_lr=0.1,
        Q_update_models=None,
        oracle_reward_lookup=True,
        pure_reward_update=True,
        weight_by_actions=False):

    cfg = {
        'prev_dir': prev_dir,
        'tag': tag,
        'epochs': epochs,
        'greedy_delay': greedy_delay,
        'oracle_reward': oracle_reward,
        'oracle_transition': oracle_transition,
        'action_cost': action_cost,
        'epsilon': epsilon,
        'gamma': gamma,
        'Q_iters': Q_iters,
        'condensed_reward': condensed_reward,
        'horizon': horizon,
        'action_names': action_names,
        'state_names': state_names,
        'delta_names': delta_names,
        'sgd_update': sgd_update,
        'sgd_lr': sgd_lr,
        'env': env,
        'seed': seed,
        'state_p0': state_p0,
        'learning_rate': learning_rate,
        'device': str(device),
        'exploration_phase': exploration_phase,
        'reward_hidden_size': reward_hidden_size,
        'Q_update_frequency': str(Q_update_frequency),
        'convergence': transition_convergence,
        'patience': transition_patience,
        'model_batchsize': model_batchsize,
        'T_lr': T_lr,
        'R_lr': R_lr,
        'oracle_reward_lookup': oracle_reward_lookup,
        'pure_reward_update': pure_reward_update,
        'weight_by_actions': weight_by_actions,
    }

    ## Load in previous objects
    if prev_dir is not None:
        prev_transition = pickle.load(open(f'{prev_dir}/transition.pkl', 'rb'))
        prev_R = pickle.load(open(f'{prev_dir}/R.pkl', 'rb'))
        prev_Q = pickle.load(open(f'{prev_dir}/Q.pkl', 'rb'))
    else:
        prev_transition = None
        prev_R = None
        prev_Q = None

    ## Set up directory to save experiment results
    timestamp = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    save_dir = f'experiments/{timestamp}_{tag}TAG_{epochs}Epochs_{greedy_delay}GreedyDelay_{oracle_reward}OracleReward_{oracle_transition}OracleTransition_{sgd_update}SGD_{condensed_reward}CondensedReward_{epsilon}Epsilon_{exploration_phase}Explore'
    print('results will save here: ', save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    actions = list(range(len(action_names)))
    states = list(range(len(state_names)))

    ## Get simulator
    if env == 'stronger_tradeoff':
        env = get_stronger_tradeoff_simulator(actions, states, state_p0, 
            gamma=gamma, action_cost=action_cost, horizon=horizon, terminal_reward=0)
    elif env == 'glucose':
        env = get_glucose_simulator(gamma, action_cost, horizon=horizon)
        actions = env.actions
        states = env.states
    elif env == 'custom_windygrid':
        env = get_custom_windy_gridworld_simulator(
            gamma, action_cost, horizon=None, 
            hazards=True, start_state='default', stochastic_wind=True, goal_reward=10000)
        actions = env.actions
        states = env.states
        print('A: ', actions, len(actions), 'S: ', states, len(states))
    else:
        raise Exception(f'env not recognized: {env}')

    ## Run experiment
    transitions = {}
    Rs = {}
    Qs = {}
    infos = {}
    if Q_update_models is None:
        Q_update_models = ['timing_smart', 'timing_dumb', 'model_free']
    for Q_update_model in Q_update_models: # , 'timing_smart_and_model_free']:        
        cfg['Q_update_model'] = Q_update_model
        wandb.init(
            project=tag, 
            name=f"{timestamp}_{Q_update_model}", 
            config=cfg,
            reinit=True,
        )

        checkpt_dir = f'{save_dir}/{Q_update_model}/'
        if not os.path.exists(checkpt_dir):
            os.makedirs(checkpt_dir)

        Qfreq = Q_update_frequency if isinstance(Q_update_frequency, int) else Q_update_frequency[Q_update_model]
        transition, R, Q, info = run_experiment(
            env, actions, states, delta_names, device,
            seed=seed, horizon=horizon, epsilon=epsilon, greedy_delay=greedy_delay,
            epochs=epochs, T_lr=T_lr, R_lr=R_lr, gamma=gamma, Q_iters=Q_iters,
            prev_transition=prev_transition, prev_R=prev_R, prev_Q=prev_Q, folder=save_dir, Q_update_model=Q_update_model,
            oracle_reward=oracle_reward, oracle_transition=oracle_transition, 
            sgd_update=sgd_update, sgd_lr=sgd_lr, 
            action_cost=action_cost, condensed_reward=condensed_reward,
            exploration_phase=exploration_phase, reward_hidden_size=reward_hidden_size,
            Q_update_frequency=Qfreq,
            convergence=transition_convergence,
            patience=transition_patience,
            model_batchsize=model_batchsize,
            oracle_reward_lookup=oracle_reward_lookup,
            weight_by_actions=weight_by_actions,
            exp_dir=checkpt_dir)

        transitions[Q_update_model] = transition
        Rs[Q_update_model] = R
        Qs[Q_update_model] = Q
        infos[Q_update_model] = info

        try:
            pickle.dump(transitions, open(f'{save_dir}/transitions.pkl', 'wb'))
            pickle.dump(Rs, open(f'{save_dir}/Rs.pkl', 'wb'))
        except Exception as e:
            print(e)
        pickle.dump(Qs, open(f'{save_dir}/Qs.pkl', 'wb'))
        pickle.dump(infos, open(f'{save_dir}/infos.pkl', 'wb'))
        wandb.alert(
            title="WandB Timing as an Action Experiment Update",
            text=f"Completed training for \nTAG: {tag} \nQ_update_model {Q_update_model} \ntimestamp {timestamp}")

    return transitions, Rs, Qs, infos, save_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--oracle_reward', action='store_true')
    parser.add_argument('--oracle_reward_lookup', action='store_true')
    parser.add_argument('--oracle_transition', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--explore', type=int, default=0)
    parser.add_argument('--action_cost', type=float, default=1)
    parser.add_argument('--timing_smart', action='store_true')
    parser.add_argument('--timing_dumb', action='store_true')
    parser.add_argument('--model_free', action='store_true')
    parser.add_argument('--epsilon_max', type=float, default=0.1)
    parser.add_argument('--epsilon_min', type=float, default=0.1)
    parser.add_argument('--env', type=str, choices=['custom_windygrid', 'stronger_tradeoff', 'glucose'])
    parser.add_argument('--T_lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--transition_convergence', type=float, default=1e-5)
    parser.add_argument('--weight_by_actions', action='store_true')
    parser.add_argument('--greedy_delay', type=str, default='0')


    args = parser.parse_args()

    wandb.login()
    
    if args.env == 'custom_windygrid':
        cfg = custom_windygrid
    elif args.env == 'stronger_tradeoff':
        cfg = stronger_tradeoff
    elif args.env == 'glucose':
        cfg = glucose
    else:
        raise NotImplementedError('Unrecognized environment: ', args.env)

    cfg['seed'] = args.seed
    
    cfg['epochs'] = args.epochs
    cfg['gamma'] = args.gamma
    cfg['action_cost'] = args.action_cost
    cfg['weight_by_actions'] = args.weight_by_actions
    if args.greedy_delay == '0':
        cfg['greedy_delay'] = 0
    else:
        cfg['greedy_delay'] = args.greedy_delay

    cfg['Q_iters'] = 50
    
    cfg['oracle_reward'] = args.oracle_reward
    cfg['oracle_reward_lookup'] = args.oracle_reward_lookup
    cfg['condensed_reward'] = (not args.oracle_reward)
    cfg['oracle_transition'] = args.oracle_transition
    cfg['device'] = 'cpu' if args.cpu else 'cuda'

    cfg['exploration_phase'] = args.explore
    
    cfg['Q_update_frequency'] = {
        'timing_smart': 1,
        'timing_dumb': 1,
        'model_free': 1,
    }

    cfg['horizon'] = 50
    cfg['learning_rate'] = 1e-2
    cfg['epsilon'] = [args.epsilon_max, args.epsilon_min]
    print(args.epsilon_max, args.epsilon_min)
    cfg['transition_convergence'] = args.transition_convergence
    cfg['transition_patience'] = 3
    cfg['model_batchsize'] = 500

    cfg['R_lr'] = 0.1
    cfg['T_lr'] = args.T_lr
    
    cfg['Q_update_models'] = []
    if args.timing_smart:
        cfg['Q_update_models'].append('timing_smart')
    if args.timing_dumb:
        cfg['Q_update_models'].append('timing_dumb')
    if args.model_free:
        cfg['Q_update_models'].append('model_free')

    timestamp = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    transitions, Rs, Qs, infos, save_dir = run(**cfg)
    wandb.finish()

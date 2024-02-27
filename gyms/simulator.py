"""Timing-as-an-action disease progression simulator."""


import numpy.random as npr


class Simulator:
    def __init__(self, name, actions, states, transition_matrix, reward_function, action_cost, terminal_state, state_p0, 
                 horizon=None, gamma=1, terminal_reward=0):
        self.name = name
        self.actions = actions
        self.states = states
        self.T = transition_matrix
        self.reward_function = reward_function
        self.cost = action_cost
        self.gamma = gamma
        self.terminal_state = terminal_state
        self.terminal_reward = terminal_reward
        self.state_p0 = state_p0
        self.cur_s = int(npr.choice(states, p=state_p0))
        self.horizon = horizon

    def step(self, a, delay, t):
        assert(a in self.actions)
        assert(self.cur_s in self.states)

        T_a = self.T[a]
        reward = 0
        cur_s = self.cur_s
        cur_t = 0  # steps taken
        terminated = (cur_s == self.terminal_state)
        
        if terminated:
            reward += self.terminal_reward
            return cur_s, reward, cur_t, terminated

        reward -= self.cost
        for _ in range(delay):
            discount = self.gamma ** cur_t
            reward += discount * self.reward_function(cur_s, a)
            cur_s = npr.multinomial(1, T_a[cur_s], size=None).tolist().index(1)
            cur_t += 1
            if cur_s == self.terminal_state:
                terminated = True
                break
            if (self.horizon is not None) and (t + cur_t >= self.horizon):
                terminated = True
                break
        
        self.cur_s = cur_s
        return cur_s, reward, cur_t, terminated

    def reset(self):
        self.cur_s = int(npr.choice(self.states, p=self.state_p0))
        return self.cur_s


class OpenAISimulatorWrapper:
    def __init__(self, name, env, action_names, state_names, action_cost, terminal_state, horizon=None, gamma=1, terminal_reward=0):
        self.name = name
        self.cost = action_cost
        self.gamma = gamma
        self.env = env
        self.reward_function = None
        if hasattr(env, 'reward_func'):
            self.reward_function = env.reward_func
        self.horizon = horizon

        self.action_names = action_names
        self.state_names = state_names
        self.actions = list(range(len(action_names)))
        self.states = list(range(len(state_names)))
        self.idx_to_action = {i: a for (i, a) in enumerate(action_names)}
        self.idx_to_state = {i: s for (i, s) in enumerate(state_names)}
        self.state_to_idx = {s: i for (i, s) in enumerate(state_names)}

        self.terminal_state_name = terminal_state
        if terminal_state is None:
            self.terminal_state = None
            self.terminal_reward = None
        else:
            self.terminal_state = self.state_to_idx[terminal_state]
            self.terminal_reward = terminal_reward

        self.T = None
        if hasattr(self.env, '_construct_T'):
            self.T = env._construct_T(action_names, state_names)

    def _get_state_idx(self, state):
        return self.state_to_idx[state]

    def step(self, a, delay, t, verbose=False):
        cur_t = 0
        reward = -self.cost
        assert(delay > 0)
        for _ in range(delay):
            discount = self.gamma ** cur_t
            cur_s, rew, terminated, _ = self.env.step(a)	
            if verbose:
                print('a: ' , a, 'raw s: ', cur_s, 'terminated: ', terminated)
            reward += discount * rew
            cur_t += 1

            if (self.horizon is not None) and (t + cur_t >= self.horizon):
                terminated = True
            
            if terminated:
                break
        s = self._get_state_idx(cur_s)
        return s, reward, cur_t, terminated

    def reset(self):
        s = self.env.reset()
        s = self._get_state_idx(s)
        return s

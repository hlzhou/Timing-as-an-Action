"""Timing-as-an-action version of SimGlucose simulator."""


from gyms.simulator import OpenAISimulatorWrapper
from simglucose.simulation.scenario import CustomScenario
from simglucose.envs.simglucose_gym_env import T1DSimEnv
from datetime import datetime


class GlucoseSimulator(OpenAISimulatorWrapper):
    def __init__(self, action_cost, horizon, gamma, terminal_reward):
        now = datetime.now()
        start_time = datetime.combine(now.date(), datetime.min.time())
        scen = []
        scenario = CustomScenario(start_time=start_time, scenario=scen)

        def custom_reward(BG_last_hour):
            if BG_last_hour[-1] > 180:
                return -1
            elif BG_last_hour[-1] < 70:
                return -2
            else:
                return 1
        
        env = T1DSimEnv(
            patient_name='adolescent#002',
            custom_scenario=scenario,
            reward_fun=custom_reward,
        )
        
        name = 'glucose'
        action_names = [0, 10, 20, 30]
        # state_names = ['low', 'normal', 'high']  # states: low (<70 mg=dL), medium, high (>180 mg=dL)
        state_names = list(range(70, 350, 10)) + ['terminal']
        terminal_state = 'terminal'
        super().__init__(
            name, env, action_names, state_names,  action_cost, terminal_state, 
            horizon=horizon, gamma=gamma, terminal_reward=terminal_reward)
        self.idx_to_state = None
        self.state_to_idx = None

    def _get_state_idx(self, raw_s):
        raw_s = int(raw_s.CGM)
        if raw_s < 70 or raw_s > 350:  # note: CGM is noisy measurement of BG
            return len(self.state_names) - 1
        else:
            return int((min(raw_s, 349) - 70) / 10.)  # group 350 with 349

    def step(self, a, delay, t):
        a = self.action_names[a]
        s, reward, cur_t, terminated = super().step(a, delay, t, verbose=False)
        if s == self.terminal_state:
            terminated = True
        reward = reward
        return s, reward, cur_t, terminated
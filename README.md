# Timing as an Action

Code repository to reproduce timing-as-an-action experiments.

## Dependencies

```
conda create --name taaa python=3.9
conda activate taaa
pip install wandb
git clone https://github.com/jxx123/simglucose.git
cd simglucose
pip install -e .
```

## Experiments

To reproduce the estimation experiments, run the following notebooks:
* `notebooks/estimate_transitions.ipynb`
* `notebooks/viz_estimated_transitions.ipynb`
* `notebooks/estimate_reward.ipynb`

To reproduce the experiments in the disease progression, glucose, and windy grid simulators, run the following shell scripts:
* `./run_disease.sh`
* `./run_glucose.sh`
* `./run_windygrid.sh`

To visualize the results of the RL experiments, use the following notebooks:
* `viz_final_returns.ipynb`
* `viz_windygrid.ipynb`

## Timing-as-an-action Simulators

All timing-as-an-action simulators can be found in the `gyms/` folder, and retrieved in their default configurations using functions in the `sim_library.py` file. The API for all timing-as-an-action environments `env` is a `env.step(a, delay, t)` function that executes the action with the delay at time `t` and returns the next state, next `t`, and whether the episode is terminated; as well as a `env.reset()` function which re-initializes the state of the environment to a starting state.

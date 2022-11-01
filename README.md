# Score Matching for RL
Code to reproduce experiments for "Exponential Family Model-Based 
Reinforcement Learning via Score Matching" at NeurIPS 2022 (Oral).

## Installation

Tested on Python 3.9.

```bash
pip install -r requirements.txt
# If functorch not installed
pip install "git+https://github.com/pytorch/functorch.git@release/torch_1.10_preview"
```

## Experiments

### Learning handcrafted transition model

We consider transitions evolving as $s' \sim \mathbb{P}( \cdot | s, a)$ where
```math
\mathbb{P}(s' | s, a) = q(s') \cdot \exp (\langle \psi(s'), W_0 \phi(s, a) - Z_{s, a}(W)).
```
Here we consider 1d densities, thus $d_s = d_a = 1$.

We call the density "CustomSinDensity" when the functions $q, \psi, \phi$ are:
```math
q(s') = -(s')^{\alpha} / \alpha, \quad \psi(s') = \sin (P \cdot s'), \quad \text{and } \phi(s, a) = [s, a].
```

Given a specific setting for $\alpha, P$, and a real-valued reward function 
$r(s, a)$, the goal is to play actions that maximize rewards when states 
evolve due to the transition model. Refer to the paper for more details on 
the specific setting for $\alpha, P$ and reward function.

`scripts/simulate.py` uses a simple planner: it chooses the action at 
each step that maximizes rewards in random rollouts based on an estimated 
transition model. The length of each rollout (called lookahead) is `tau` 
and planner plays `num_trials` rollouts.

```bash
# To use Gaussian as the estimated transition model
SAMPLING_DENSITY="normal" SAMPLING_NOISE_SCALE="1e-1" \
  bash scripts/simulate_custom_sin.sh

# To use CustomSin as the estimated transition model (functions q, psi, phi 
# known; parameter W is learned)
SAMPLING_DENSITY="custom_sin" SAMPLING_NOISE_SCALE="1e-1" \
  bash scripts/simulate_custom_sin.sh

# To use actual transition model as the estimate (baseline CustomSin), i.e. 
# parameter W is known and NOT learned
bash scripts/simulate_custom_sin_baseline_self.sh
```

#### Plots

Finally, generate plots for the three experiments with the following two 
scripts. The first plots the action the planner chooses for the transition 
model, and the second plots the cumulative rewards over the `T` episodes.

```bash
# Save plots in directory
mkdir -p notebooks/plots

python scripts/plot_planner_actions.py

python scripts/plot_cum_regret_baselines.py
```

### Adding noise to cartpole (inverted pendulum) and using a guess transition model to fit noisy transitions

OpenAI gym environment in `gym_env/stochastic_cartpole.py`, noise model under 
development.

## Under development/known issues

- `".*hmc.*"` sampling methods are not maintained. Initialization influences 
  convergence for these sampling methods.
- `"inv_cdf"` sampling method requires computing the log-partition function 
  $Z_{s, a}(W) = \log \int q(x) \exp (\langle \psi(x), W \phi(s, a) \rangle) dx$.
  The integral is computed using either `scipy.integrate.quad` or as a 
  Riemann integral. If the integral overflows `np.float32` maximum
  (`np.finfo(np.float32).max`), the computed density is degenerate and the 
  sampling process fails. A simple fix would be to use `np.float64`.

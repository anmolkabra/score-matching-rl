# Score Matching for RL
Code for "Exponential Family Model-Based 
Reinforcement Learning via Score Matching" at NeurIPS 2022 (**Oral
presentation**).

## Installation

Tested on Python 3.9. Create a virtual or a conda environment, and install 
the requirements.

```bash
pip install -r requirements.txt
# If functorch not installed
pip install torch==1.11.0 functorch==0.1.1
```

## Experiments

### Learning handcrafted transition model

We test a synthetic MDP with transitions evolving as $s' \sim 
\mathbb{P}( \cdot | s, a)$ where
```math
\mathbb{P}(s' | s, a) = q(s') \cdot \exp (\langle \psi(s'), W_0 \phi(s, a) \rangle - Z_{s, a}(W)).
```
We work with 1d real-valued state and binary actions, i.e. $d_s \in \mathbb{R}$ 
and $d_a \in \{-1, 1\}$.

We call the density "CustomSinDensity" when the functions $q, \psi, \phi$ are:
```math
q(s') = -(s')^{\alpha} / \alpha, \quad \psi(s') = \sin (P \cdot s'), \quad \text{and} \quad \phi(s, a) = [s, a].
```

Given a specific setting for $\alpha, P$, and a real-valued reward function 
$r(s, a)$, the goal is to play actions that maximize rewards when states 
evolve according to the transition model. Refer to the paper for more details on 
the specific setting for $\alpha, P$ and the reward function.

#### Running experiments with CustomSinDensity

`scripts/simulate.py` uses a simple random shooting planner: it (i) simulates 
lookaheads of playing `[+1, . . . , +1]` and `[−1, . . . , −1]`, and (ii) 
chooses action depending on which yields higher reward. The length of each 
lookahead is `tau` and the planner simulates `num_trials` lookaheads.

```bash
# To use Gaussian as the estimated transition model
SAMPLING_DENSITY="normal" bash scripts/simulate_custom_sin.sh

# To use CustomSin as the estimated transition model (functions q, psi, phi 
# known; parameter W is learned)
SAMPLING_DENSITY="custom_sin" bash scripts/simulate_custom_sin.sh

# To use actual transition model as the estimate (baseline CustomSin), i.e. 
# parameter W is known and NOT learned
bash scripts/simulate_custom_sin_baseline_self.sh
```

#### Plots

Finally, generate plots for the three experiments with the following two 
scripts. The first plots the action the planner chooses for the transition 
model, and the second plots the cumulative rewards over the training episodes.

```bash
# Save plots in this directory
mkdir -p notebooks/plots

python scripts/plot_planner_actions.py

python scripts/plot_cum_regret_baselines.py
```

#### Example transitions and Score Matching fit on 1d densities

We also provide code that checks the fit of Score Matching estimate on 
handcrafted densities and MDPs. We used these for testing purposes.
- `notebooks/fit_1d_densities_w_sm.ipynb` has examples of different
  densities that we could fit with Score Matching. Several densities do not
  follow the required theoretical assumptions for Score Matching to work,
  but Score Matching empirically finds a good fit anyway. This could enable
  research on potentially relaxing assumptions for Score Matching.
- `notebooks/synthetic_mdp.ipynb` has examples of how the CustomSinDensity 
  behaves on a selection of states and actions.

### Learning noisy control tasks

Learning noisy cartpole (inverted pendulum) environment remains a challenge. 
We have tried several noise models and guesses for an exponential family 
density to fit the noisy cartpole transitions.

Cartpole environment uses OpenAI gym in `gym_env/stochastic_cartpole.py` and 
can be simulated using the relevant flags in `scripts/simulate.py`.

## Under development/known issues

- `".*hmc.*"` sampling methods are not maintained. Initialization influences 
  convergence for these sampling methods.
- `"inv_cdf"` sampling method requires computing the log-partition function 
  $Z_{s, a}(W) = \log \int q(x) \exp (\langle \psi(x), W \phi(s, a) \rangle) dx$.
  The integral is computed using either `scipy.integrate.quad` or as a 
  Riemann integral. If the integral overflows `np.float32` maximum
  (`np.finfo(np.float32).max`), the computed density is degenerate and the 
  sampling process fails. A simple fix would be to use `np.float64`.

## Citation

```
@article{li2021exponential,
  title={Exponential Family Model-Based Reinforcement Learning via Score Matching},
  author={Li, Gene and Li, Junbo and Kabra, Anmol and Srebro, Nathan and Wang, Zhaoran and Yang, Zhuoran},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```

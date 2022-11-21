import argparse
import collections
import copy
import json
import logging
import multiprocessing
import os
from dataclasses import dataclass

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from data_gen.density import ExpGLMDensity, NonLDSDensity, CustomSinDensity
from data_gen.phi_embedding import LDSPhiEmbedding, PhiEmbedding
from estimation.score_matching import ScoreMatching
import gym
from gym_env.stochastic_cartpole import StochasticCartPoleEnv
from gym_env.mdp_env import CustomSinMDPEnv
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import utils
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger("simulate_cartpole")
logger.setLevel(logging.DEBUG)

T = Union[torch.Tensor, np.ndarray]


class CustomForExpReLUDensity(ExpGLMDensity):
    ALPHA = 2
    SUPPORT_LO = -np.inf
    SUPPORT_HI = np.inf
    IS_PRODUCT_DENSITY = True

    def __init__(
        self,
        d_s: int,
        sigma,
        use_scipy_integrate: bool = False,
        integrate_lo: float = -5.0,
        integrate_hi: float = 5.0,
        integrate_linspace_L: int = 1000,
    ):
        super(CustomForExpReLUDensity, self).__init__(
            d_s,
            use_scipy_integrate=use_scipy_integrate,
            integrate_lo=integrate_lo,
            integrate_hi=integrate_hi,
            integrate_linspace_L=integrate_linspace_L,
        )
        self.sigma = utils.to_torch(sigma)
        self.d_psi = self.d_s * 2  # adding extra coordinates

    def __repr__(self) -> str:
        ALPHA = self.ALPHA
        sigma = self.sigma
        return f"$\\exp ( - \\frac{{ ||s'||_\\alpha^\\alpha }}{{ \\alpha \\cdot \\sigma }} ) \\cdot \\exp ( \\langle [s', relu(s')], W \\phi(s, a) \\rangle - Z_{{s,a}}(W) )$\n$\\alpha={ALPHA:.1f},\\sigma={sigma}$"

    def in_support(self, x: T) -> torch.Tensor:
        return torch.ones((x.shape[0]), dtype=bool)

    def psi(self, x: T) -> torch.Tensor:
        x = utils.to_torch(x)  # shape (n, d_s)
        psi = torch.concat([x, F.relu(x)], dim=-1)
        return psi

    def logq(self, x: T) -> torch.Tensor:
        x = utils.to_torch(x)  # shape (n, d_s)
        ALPHA = self.ALPHA
        # Weird issue of dimension collapse when sigma is 1d
        sigma = self.sigma.item() if self.sigma.shape[0] == 1 else self.sigma
        logq = -x.norm(p=ALPHA, dim=-1).pow(ALPHA) / (
            sigma * ALPHA
        )  # shape (n)
        return logq

    def _inv_cdf__get_vji_tilde(
        self,
        v: torch.Tensor,
        j: int,
        i: int,
        inv_cdf_round_decimals: int,
    ) -> str:
        # For custom density, different sigma result in different
        # densities. So to differentiate densities with same v[j, i] but
        # different sigma, add sigma info to vji_tilde str
        sigma_str = f"sigma{self.sigma[i].item():.2e}"
        vji_tilde = ",".join(
            str(round(v[j, k].item(), inv_cdf_round_decimals))
            for k in range(v.shape[1])
        )
        vji_tilde = f"{sigma_str}_{vji_tilde}"
        return vji_tilde


class CustomForMoGReLUDensity(ExpGLMDensity):
    ALPHA = 2
    SUPPORT_LO = -np.inf
    SUPPORT_HI = np.inf
    IS_PRODUCT_DENSITY = True

    def __init__(
        self,
        d_s: int,
        sigma,
        use_scipy_integrate: bool = False,
        integrate_lo: float = -5.0,
        integrate_hi: float = 5.0,
        integrate_linspace_L: int = 1000,
    ):
        super(CustomForMoGReLUDensity, self).__init__(
            d_s,
            use_scipy_integrate=use_scipy_integrate,
            integrate_lo=integrate_lo,
            integrate_hi=integrate_hi,
            integrate_linspace_L=integrate_linspace_L,
        )
        self.sigma = utils.to_torch(sigma)
        self.d_psi = self.d_s * 3  # adding extra coordinates

    def __repr__(self) -> str:
        ALPHA = self.ALPHA
        sigma = self.sigma
        return f"$\\exp ( - \\frac{{ ||s'||_\\alpha^\\alpha }}{{ \\alpha \\cdot \\sigma }} ) \\cdot \\exp ( \\langle [s', relu(s')], W \\phi(s, a) \\rangle - Z_{{s,a}}(W) )$\n$\\alpha={ALPHA:.1f},\\sigma={sigma}$"

    def in_support(self, x: T) -> torch.Tensor:
        return torch.ones((x.shape[0]), dtype=bool)

    def psi(self, x: T) -> torch.Tensor:
        x = utils.to_torch(x)  # shape (n, d_s)
        psi = torch.concat(
            [x, F.relu(x), F.relu(-x)],
            dim=-1,
        )
        return psi

    def logq(self, x: T) -> torch.Tensor:
        x = utils.to_torch(x)  # shape (n, d_s)
        ALPHA = self.ALPHA
        # Weird issue of dimension collapse when sigma is 1d
        sigma = self.sigma.item() if self.sigma.shape[0] == 1 else self.sigma
        logq = -x.norm(p=ALPHA, dim=-1).pow(ALPHA) / (
            sigma * ALPHA
        )  # shape (n)
        return logq

    def _inv_cdf__get_vji_tilde(
        self,
        v: torch.Tensor,
        j: int,
        i: int,
        inv_cdf_round_decimals: int,
    ) -> str:
        # For custom density, different sigma result in different
        # densities. So to differentiate densities with same v[j, i] but
        # different sigma, add sigma info to vji_tilde str
        sigma_str = f"sigma{self.sigma[i].item():.2e}"
        vji_tilde = ",".join(
            str(round(v[j, k].item(), inv_cdf_round_decimals))
            for k in range(v.shape[1])
        )
        vji_tilde = f"{sigma_str}_{vji_tilde}"
        return vji_tilde


class ExpGLM_SimEnvFactory:
    """
    We just need a closure function from the factory, but closures can't be used
    with multiprocessing as they can't be pickled by python (closures are local
    to global functions). Instead, capture the variables using an object, and
    use a method.
    """

    def __init__(
        self,
        exp_glm_density: Union[ExpGLMDensity, List[ExpGLMDensity]],
        W: Union[torch.Tensor, List[torch.Tensor]],
        phi_embedding: PhiEmbedding,
        sampling_method: str,
        sampling_kwargs: Dict[str, Any],
    ):
        """
        Args:
            exp_glm_density: ExpGLMDensity instance (or list).
            W: Parameter matrix for the density (or list). shape (d_psi, d_phi)
            phi_embedding: phi instance to get phi(s, a).
            sampling_method: Sampling method ('deterministic', 'exact', 'hmc'
                etc.)
            sampling_kwargs: Keyword args for sampling.
        """
        self.exp_glm_density = exp_glm_density
        self.W = W
        self.phi_embedding = phi_embedding
        self.sampling_method = sampling_method
        self.sampling_kwargs = sampling_kwargs

    def sim_env(self, s: np.ndarray, a: np.ndarray, tau: int) -> np.ndarray:
        """
        Returns simulated sequential samples s_{t+1} = env(s_t, a) according to
        the specified exp_glm_density.

        But if sampling_method is 'deterministic', s' = W phi(s, a) are returned.

        Args:
            s: Starting state. shape (d_s)
            a: Actions to take at each step t <= tau. shape (tau, d_a)
            tau: Number of simulations to run.

        Returns:
            samples: shape (tau, d_s) where
            ```
            for i = 1..tau
                samples[i] = sample( W @ phi(s, a[i]) )
                s = samples[i]
            ```
        """
        if len(a.shape) == 1:
            # d_a must have been 1 for action_space.sample() to return ints as
            # actions
            a = a[:, np.newaxis]  # shape (num_samples, 1)

        s = utils.to_torch(s[np.newaxis, :])  # shape (1, d_s)
        a = utils.to_torch(a[np.newaxis, :])  # shape (1, tau, d_a)

        if self.sampling_method == "deterministic":
            assert s.shape[1] == self.W.shape[0]
            # s' = W phi(s, a)
            samples, phis = [], []
            for t in range(tau):
                # shape (1, d_phi)
                phi = self.phi_embedding.get_phi(s, a[:, t, :])
                if args.W_fit_states_sep:
                    s = torch.hstack(
                        [phi @ Wi.T for Wi in self.W]
                    )  # shape (1, d_s)
                else:
                    s = phi @ self.W.T  # shape (1, d_s)
                samples.append(s)
                phis.append(phi)
            # shape (1, tau, d_s)
            samples = torch.stack(samples, dim=0).permute(1, 0, 2)
            # shape (1, tau, d_phi)
            phis = torch.stack(phis, dim=0).permute(1, 0, 2)
        else:
            if args.W_fit_states_sep:
                # sample sequentially from exp_glm_density
                sj = s
                samples = []
                for j in range(tau):
                    # each entry has shape (1, d_s=1)
                    samples_j = [
                        d.sample_iid(
                            Wi,
                            sj,
                            a[:, j, :],
                            self.phi_embedding,
                            1,
                            sampling_method=self.sampling_method,
                            sampling_kwargs=self.sampling_kwargs,
                        )[0][:, 0, :]
                        for (d, Wi) in zip(self.exp_glm_density, self.W)
                    ]
                    samples_j = torch.hstack(samples_j)  # shape (1, d_s)
                    samples_j[
                        torch.isnan(samples_j)
                    ] = 0.0  # density could be really bad
                    samples.append(samples_j)
                    sj = samples_j
                samples = torch.stack(samples).permute(
                    1, 0, 2
                )  # shape (1, tau, d_s)
            else:
                # sample sequentially from exp_glm_density
                # shape (1, tau, d_s), (1, tau, d_phi)
                samples, phis = self.exp_glm_density.sample_seq(
                    self.W,
                    s,
                    a,
                    self.phi_embedding,
                    tau,
                    sampling_method=self.sampling_method,
                    sampling_kwargs=self.sampling_kwargs,
                )
        # shape (tau, d_s)
        return utils.to_numpy(samples.squeeze(dim=0).detach())


def do_random_shoot(
    real_env: gym.Env,
    sim_env: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    start_s: np.ndarray,
    actions: np.ndarray,
) -> int:
    """
    Performs one random shoot of tau lookahead starting at state start_s using
    the actions (shape (tau, d_a)), and returns the reward of the shoot.
    """
    tau = actions.shape[0]

    # Simulate the actions starting at s_h for tau lookahead, and
    # accumulate the rewards
    total_reward = 0

    samples = sim_env(start_s, actions, tau)  # shape (tau, d_s)
    # shape (tau, d_s)
    states = np.concatenate((start_s[np.newaxis, :], samples[:-1, :]), axis=0)
    for s, a in zip(states, actions):
        # Can query the real_env with a given state-action to calculate
        # reward without altering the env's state
        _, reward, done, _ = real_env.step(a, state_query=s)
        total_reward += reward
        if done:
            break

    return total_reward


def random_shooting_planner(
    args,
    real_env: gym.Env,
    sim_env: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
) -> List[Tuple[Any, ...]]:
    """
    Returns a trajectory of length *upto* H (horizon) planned using random shooting.

    sim_env(s, a, tau)
    Args:
        real_env: Real environment instance.
        sim_env: Simulator function, should return a simulated trajectory
            starting at state s of length tau using the given actions a, which
            is of shape (tau, d_a).

    Returns:
        trajectory, list of tuples (s, a, s', r)
    """
    H = args.random_shooting_H
    tau = args.random_shooting_tau
    num_trials = args.random_shooting_num_trials

    trajectory = []
    s_h = real_env.reset()

    for h in range(H):
        if not args.env_no_render:
            real_env.render()

        # Find the best action in num_trials shoots of tau lookahead
        if args.random_shooting_all_ones_actions:
            # list, each shape (tau, d_a)
            actions_trials_pos = [
                np.stack([1 for _ in range(tau)], axis=0)
                for _ in range(num_trials)
            ]
            actions_trials_neg = [
                np.stack([-1 for _ in range(tau)], axis=0)
                for _ in range(num_trials)
            ]

            rewards_trials_pos = [
                do_random_shoot(real_env, sim_env, s_h, actions)
                for actions in actions_trials_pos
            ]
            rewards_trials_neg = [
                do_random_shoot(real_env, sim_env, s_h, actions)
                for actions in actions_trials_neg
            ]
            # Pick action that consistently delivered more reward
            opt_a_h = (
                1
                if np.mean(rewards_trials_pos) > np.mean(rewards_trials_neg)
                else -1
            )
        else:
            # list, each shape (tau, d_a)
            actions_trials = [
                np.stack(
                    [real_env.action_space.sample() for _ in range(tau)], axis=0
                )
                for _ in range(num_trials)
            ]

            if args.num_workers > 1:
                # TODO hamiltorch weird behavior: even if store_on_GPU is false,
                # why does hamiltorch create use gpu contexts in each child
                # proc? To avoid CUDA errors, only non hmc-high num_workers
                # combination triggers multiprocessing

                pool = multiprocessing.Pool(args.num_workers)
                pool_args = [
                    (real_env, sim_env, s_h, actions)
                    for actions in actions_trials
                ]
                rewards_trials = pool.starmap(do_random_shoot, pool_args)
                pool.close()
                pool.join()
            else:
                rewards_trials = [
                    do_random_shoot(real_env, sim_env, s_h, actions)
                    for actions in actions_trials
                ]

            # Pick first action from the trial with the highest reward
            best_trial_idx = np.argmax(rewards_trials)
            opt_a_h = actions_trials[best_trial_idx][0]

        # Play the action in the real env
        ss_h, reward, done, info = real_env.step(opt_a_h)
        trajectory.append([s_h, opt_a_h, ss_h, reward])
        s_h = ss_h
        if done:
            break

    return trajectory


def W_estimator(
    args,
    dataset: Dict,
    phi_embedding: PhiEmbedding,
    exp_glm_density_for_estimate: Union[ExpGLMDensity, List[ExpGLMDensity]],
) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
    """
    Returns an estimate of W (or list) from the dataset modeled using
    exp_glm_density_for_estimate and the avg residual of the estimate as if W is
    estimated using least squares.
    """
    ss = np.concatenate(dataset["ss"], axis=0)  # shape (n, d_s)
    s = np.concatenate(dataset["s"], axis=0)  # shape (n, d_s)
    a = np.concatenate(dataset["a"], axis=0)  # shape (n) or (n, d_a) if d_a!=1
    if len(a.shape) == 1:
        a = a[:, np.newaxis]
    phis = phi_embedding.get_phi(s, a)  # shape (n, d_phi)
    d_s = ss.shape[1]

    if args.W_use_MLE_estimator:
        # Already fits each state separately (MLE assumes noise is gaussian)
        # shape (d_s, d_phi)
        W_MLE = torch.linalg.lstsq(phis, utils.to_torch(ss)).solution.T
        avg_loss = (
            torch.norm(phis @ W_MLE.T - utils.to_torch(ss)) / phis.shape[0]
        )
        return W_MLE, avg_loss
    else:
        if args.W_fit_states_sep:
            avg_loss = 0.0
            lams = [
                args.W_sm_lam
                if type(args.W_sm_lam) == float
                else args.W_sm_lam[i]
                for i in range(d_s)
            ]
            W_SM = [
                ScoreMatching(d)
                .W_estimate(ss[:, i : i + 1], phis, lam=lam)
                .detach()
                for i, (d, lam) in enumerate(
                    zip(exp_glm_density_for_estimate, lams)
                )
            ]  # compute the training loss on the last 3 coordinates, since these are fit with Gaussian.
        else:
            sm_estimator = ScoreMatching(exp_glm_density_for_estimate)
            W_SM = sm_estimator.W_estimate(ss, phis, lam=args.W_sm_lam).detach()
            avg_loss = (
                torch.norm(phis @ W_SM.T - utils.to_torch(ss)) / phis.shape[0]
            )
        return W_SM, avg_loss


def benchmark_random_actions(args, env: gym.Env):
    """
    Runs random actions benchmark (horizon specified in args).
    """
    H = args.random_shooting_H
    num_runs = args.benchmark_random_actions_num_runs
    rewards_runs = np.zeros(num_runs)
    for i in range(num_runs):
        env.reset()
        total_reward = 0

        for h in range(H):
            a_h = env.action_space.sample()
            _, reward, done, _ = env.step(a_h)
            total_reward += reward
            if done:
                break

        rewards_runs[i] = total_reward

    logger.info("Random Actions benchmark results:")
    logger.info(stats.describe(rewards_runs))


sim_tuple_T = Tuple[
    argparse.Namespace,  # args
    Union[ExpGLMDensity, List[ExpGLMDensity]],  # exp_glm_density
    Union[torch.Tensor, List[torch.Tensor]],  # W
    PhiEmbedding,  # phi_embedding
]


def wrap_gym_as_sim_env(
    env: gym.Env,
) -> Callable[[np.ndarray, np.ndarray, int], np.ndarray]:
    def gym_as_sim_env(
        start_s: np.ndarray, actions: np.ndarray, tau: int
    ) -> np.ndarray:
        samples = np.empty((tau, start_s.shape[0]))  # shape (tau, d_s)
        s = start_s  # shape (d_s)
        for i in range(tau):
            ai = actions[i]  # shape (d_a)
            s, reward, done, _ = env.step(ai, state_query=s)
            samples[i] = s
        return samples

    return gym_as_sim_env


def train(
    env: gym.Env,
    sim: sim_tuple_T,
    base_sim: Optional[sim_tuple_T] = None,
):
    """
    Returns the trained W (or list). If base_sim is None, returns a tuple
    (W_trained, base_W_trained) where the second W is for the base_sim.
    """
    args, exp_glm_density, W_init, phi_embedding = sim
    W_k = W_init
    dataset = collections.defaultdict(list)
    if base_sim is not None:
        b_args, b_exp_glm_density, b_W_init, b_phi_embedding = base_sim
        b_W_k = b_W_init
        b_dataset = collections.defaultdict(list)

    # For plotting density at fixed s and a
    num_samples = 10000
    if args.env == "cartpole":
        d_s = 4
        state_query_max = np.array(
            [
                env.x_threshold,
                0.1,
                env.theta_threshold_radians,
                0.1,
            ]
        )
    elif args.env == "custom_sin":
        d_s = 1
        state_query_max = np.array([1.0])

    state_query = (
        np.random.rand(1, d_s) * 2 * state_query_max
    ) - state_query_max  # shape (1, d_s)
    action_query = (
        2 * torch.randint(0, 2, (1, 1)) - 1
    ).numpy()  # shape (1, d_a)

    # Get samples from the real environment for plotting
    env_samples = []
    env_samples_r = []
    for i in range(num_samples):
        result = env.step(action=action_query[0, 0], state_query=state_query[0])
        next_state, reward, done, info = result
        env_samples.append(next_state)
        env_samples_r.append(reward)
    env_samples = np.array(env_samples)  # shape (num_samples, d_s)
    env_samples_r = np.array(env_samples_r)  # shape (num_samples)

    def update_W(_args, _exp_glm_density, _W, _phi_embedding, _dataset):
        sim_env_factory = ExpGLM_SimEnvFactory(
            _exp_glm_density,
            _W,
            _phi_embedding,
            _args.sampling_method,
            _args.sampling_kwargs,
        )
        sim_env_k = sim_env_factory.sim_env
        trajectory = random_shooting_planner(_args, env, sim_env_k)

        s, a, ss, r = zip(*trajectory)
        _dataset["s"].append(s)
        _dataset["a"].append(a)
        _dataset["ss"].append(ss)
        _dataset["r"].append(r)

        W_k, avg_loss = W_estimator(
            _args, _dataset, _phi_embedding, _exp_glm_density
        )
        return W_k, avg_loss

    # Fitting density
    H = args.random_shooting_H
    avg_regret = 0
    if args.baseline_self:
        logger.info(">>> Using real env for simulation...")
    for k in tqdm(range(args.train_num_episodes), desc="Episode"):
        if args.baseline_self:
            # Use the real env as sim, no need to estimate W_k
            W_k = torch.tensor(env.W, dtype=torch.float32)
            sim_env = wrap_gym_as_sim_env(env)
            trajectory = random_shooting_planner(args, env, sim_env)

            s, a, ss, r = zip(*trajectory)
            dataset["s"].append(s)
            dataset["a"].append(a)
            dataset["ss"].append(ss)
            dataset["r"].append(r)
            avg_loss = 0.0

        else:
            logger.info(f"learned W_k at step {k}, {W_k}")
            # Update the sim W
            W_k, avg_loss = update_W(
                args, exp_glm_density, W_k, phi_embedding, dataset
            )
        r = dataset["r"][-1]
        opt_a = dataset["a"][-1]
        total_r = sum(r)
        avg_regret += (H - total_r) / H
        r_info = f"Reward: {total_r}"

        if base_sim is not None:
            b_W_k, b_avg_loss = update_W(
                b_args, b_exp_glm_density, b_W_k, b_phi_embedding, b_dataset
            )
            b_r = b_dataset["r"][-1]
            r_info += f" (baseline NonLDS-exact: {sum(b_r)})"
            base_sim = (b_args, b_exp_glm_density, b_W_k, b_phi_embedding)

        # Log results from the trajectory
        logger.info(r_info)
        args.tb_writer.add_scalar("reward", total_r, k)
        args.tb_writer.add_scalar("avg_regret", avg_regret, k)
        args.tb_writer.add_scalar("avg_loss", avg_loss, k)
        with open(os.path.join(args.tb_log_dir, "results.csv"), "a") as f:
            f.write(f"{k},{total_r:.10e},{avg_regret:.10e},{avg_loss:.10e}\n")
        with open(
            os.path.join(args.tb_log_dir, "planner_action_counts.csv"), "a"
        ) as f:
            count_1s = sum(np.array(opt_a) == 1)
            count_m1s = sum(np.array(opt_a) == -1)
            f.write(f"{k},{count_1s},{count_m1s}\n")
        if args.W_fit_states_sep:
            for i, Wi in enumerate(W_k):
                np.save(
                    os.path.join(args.tb_log_dir, f"last_W{i}.npy"), Wi.numpy()
                )
        else:
            np.save(os.path.join(args.tb_log_dir, "last_W.npy"), W_k.numpy())

        plot_density(
            env_samples,
            env_samples_r,
            state_query,
            action_query,
            sim=(args, exp_glm_density, W_k, phi_embedding),
            base_sim=base_sim,
            num_samples=num_samples,
        )

    if base_sim is None:
        return W_k
    else:
        return W_k, b_W_k


@dataclass
class SamplesPlotting:
    name: str
    samples: np.ndarray
    plot_name: str


def plot_density(
    env_samples: np.ndarray,
    env_samples_r: np.ndarray,
    state_query: np.ndarray,
    action_query: np.ndarray,
    sim: sim_tuple_T,
    base_sim: Optional[sim_tuple_T] = None,
    num_samples: int = 1000,
):
    """
    Plots the env_samples (next_states from real env) and simulator next_states
    when polled at state_query and action_query. Saves the plot in the logging
    dir.

    Args:
        env_samples: Samples obtained at s=state_query, a=action_query from
            real env). shape (num_samples, d_s).
        env_samples_r: Reward values of samples. shape (num_samples).
        state_query: State at which real env was sampled. shape (1, d_s)
        action_query: Action at which real env was sampled. shape (1, d_a)
        sim: Simulator density tuple
        base_sim: (default None) Baseline simulator density tuple
        num_samples: Number of samples.
    """
    samples_plotting = [
        SamplesPlotting(
            name="env", samples=env_samples, plot_name="s' $\\sim$ True Env"
        )
    ]

    # Get samples from the simulator for plotting
    args, exp_glm_density, W, phi_embedding = sim
    if args.W_fit_states_sep:
        # exp_glm_density is a list of densities for each state, sample
        # each state independently
        sim_samples = [
            d.sample_iid(
                Wi,
                state_query,
                action_query,
                phi_embedding,
                num_samples,
                sampling_method=args.sampling_method,
                sampling_kwargs=args.sampling_kwargs,
            )[0][0, :, 0].numpy()
            # shape (num_samples)
            for i, (d, Wi) in enumerate(zip(exp_glm_density, W))
        ]
        sim_samples = np.stack(sim_samples, axis=1)  # shape (num_samples, d_s)
    else:
        sim_samples = exp_glm_density.sample_iid(
            W,
            state_query,
            action_query,
            phi_embedding,
            num_samples,
            sampling_method=args.sampling_method,
            sampling_kwargs=args.sampling_kwargs,
        )[0][
            0, :, :
        ].numpy()  # shape (num_samples, d_s)

    samples_plotting.append(
        SamplesPlotting(
            name=f"sim",
            samples=sim_samples,
            plot_name=f"s' $\\sim$ Sim Env (check saved args on info)",
        )
    )

    # Get samples from base simulator for plotting
    if base_sim is not None:
        b_args, b_exp_glm_density, b_W, b_phi_embedding = base_sim
        if b_args.W_fit_states_sep:
            # exp_glm_density is a list of densities for each state, sample
            # each state independently
            sim_samples = [
                d.sample_iid(
                    Wi,
                    state_query,
                    action_query,
                    b_phi_embedding,
                    num_samples,
                    sampling_method=b_args.sampling_method,
                    sampling_kwargs=b_args.sampling_kwargs,
                )[0][
                    0, :, 0
                ].numpy()  # shape (num_samples)
                for i, (d, Wi) in enumerate(zip(b_exp_glm_density, b_W))
            ]
            sim_samples = np.stack(
                sim_samples, axis=1
            )  # shape (num_samples, d_s)
        else:
            sim_samples = b_exp_glm_density.sample_iid(
                b_W,
                state_query,
                action_query,
                b_phi_embedding,
                num_samples,
                sampling_method=b_args.sampling_method,
                sampling_kwargs=b_args.sampling_kwargs,
            )[0][
                0, :, :
            ].numpy()  # shape (num_samples, d_s)

        samples_plotting.append(
            SamplesPlotting(
                name=f"base_sim",
                samples=sim_samples,
                plot_name=f"s' $\\sim$ Base Sim Env (NonLDS-exact)",
            )
        )

    # Plot densities of each state separately
    d_s = state_query.shape[1]
    nr = int(np.sqrt(d_s))
    nc = int(np.ceil(d_s / nr))
    fig, axs = plt.subplots(nr, nc, figsize=(nc * 4, nr * 4))

    for i in range(nr):
        for j in range(nc):
            # Plot 1 state of the samples
            state_idx = nr * i + j
            x_min = min(min(d.samples[:, state_idx]) for d in samples_plotting)
            x_max = max(max(d.samples[:, state_idx]) for d in samples_plotting)
            ax_obj = axs[i, j] if d_s > 1 else axs
            for d in samples_plotting:
                ax_obj.hist(
                    d.samples[:, state_idx],
                    bins=100,
                    range=(x_min, x_max),
                    density=True,
                    alpha=0.3,
                    label=d.plot_name,
                )
            ax_obj.scatter(
                env_samples[:, state_idx],
                env_samples_r,
                label="rewards on true samples",
                marker=".",
            )
            ax_obj.set_title(f"$s_{state_idx}$")
    plt.legend(bbox_to_anchor=(0, -0.1), loc="upper center")

    if args.env == "cartpole":
        env_str = f"cartpole_{args.env_noise_density_name}"
        if args.env_noise_density_name == "mog":
            env_str += f"_width{args.mogwidth}"
        sim_str = f"sim_{args.sampling_density}"
        if args.sampling_density in [
            "custom_for_exp_relu",
            "custom_for_mog_relu",
        ]:
            sim_str += f"_sigma_mult{args.sampling_noise_scale_mult}"
        plot_title = f"{env_str}__{sim_str}"
    elif args.env == "custom_sin":
        plot_title = (
            f"custom_sin__action{action_query[0, 0]:.0f}__"
            f"P{args.env_custom_sin_P}_ALPHA"
            f"{args.env_custom_sin_ALPHA}"
        )
    plt.suptitle(
        f"{plot_title}\nFitted with multiple densities: {args.sampling_method} smpl"
    )

    plot_save_path = os.path.join(args.tb_log_dir, plot_title)
    plt.savefig(f"{plot_save_path}.png", bbox_inches="tight")
    plt.close()


def evaluate(
    real_env: gym.Env,
    sim: sim_tuple_T,
    base_sim: Optional[sim_tuple_T] = None,
):
    # Simulate a trajectory and get total rewards
    args, exp_glm_density, W, phi_embedding = sim
    sim_env_factory = ExpGLM_SimEnvFactory(
        exp_glm_density,
        W,
        phi_embedding,
        args.sampling_method,
        args.sampling_kwargs,
    )
    trajectory = random_shooting_planner(
        args, real_env, sim_env_factory.sim_env
    )
    s, a, ss, r = zip(*trajectory)
    total_r = sum(r)
    logger.info(f"Reward achieved/Horizon: {total_r}/{args.random_shooting_H}")


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set up tensorboard and log results
    args.tb_log_dir = os.path.join(args.log_dir, args.exp_name)
    os.makedirs(args.tb_log_dir, exist_ok=True)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(args.tb_log_dir, "exp.log"), "w")
    fh.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    ch.setFormatter(log_formatter)
    fh.setFormatter(log_formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    with open(os.path.join(args.tb_log_dir, f"config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    logger.info(json.dumps(vars(args), indent=2, sort_keys=True))

    args.tb_writer = SummaryWriter(log_dir=args.tb_log_dir)
    with open(os.path.join(args.tb_log_dir, "results.csv"), "w") as f:
        f.write("episode,reward,avg_regret,avg_loss\n")
    with open(
        os.path.join(args.tb_log_dir, "planner_action_counts.csv"), "w"
    ) as f:
        f.write("episode,count_plus_ones,count_minus_ones\n")

    # Set num_workers
    num_cpus = multiprocessing.cpu_count()
    if args.num_workers == -1:
        args.num_workers = num_cpus
    elif args.num_workers > num_cpus:
        args.num_workers = num_cpus

    # Get sampling kwargs
    if os.path.isfile(args.sampling_kwargs_filepath):
        with open(args.sampling_kwargs_filepath, "r") as f:
            args.sampling_kwargs = json.load(f)
        args.sampling_kwargs["store_on_GPU"] = False
    else:
        raise ValueError(
            f"sampling file {args.sampling_kwargs_filepath} does not exist"
        )

    if args.W_fit_states_sep:
        assert type(args.W_sm_lam) == str
        args.W_sm_lam = [float(s) for s in args.W_sm_lam.split(",")]
    else:
        args.W_sm_lam = float(args.W_sm_lam)

    # Set up environment
    if args.env == "cartpole":
        d_s = 4
        d_a = 1
        env = StochasticCartPoleEnv(
            args.env_use_cont_action,
            noise_type=args.env_noise_type,
            noise_density_name=args.env_noise_density_name,
            noise_density_sampling_method=args.env_noise_density_sampling_method,
            noise_stdev=args.env_state_noise_scale,
            mog_width=args.mogwidth,
        )
        sigma = (
            args.sampling_noise_scale
            * np.array(
                [
                    env.x_threshold,
                    (env.tau * env.force_mag / env.total_mass),
                    env.theta_threshold_radians,
                    (env.tau * env.gravity / env.length),
                ]
            )
        ) ** 2
    elif args.env == "custom_sin":
        d_s = 1
        d_a = 1
        env = CustomSinMDPEnv(
            P=args.env_custom_sin_P,
            ALPHA=args.env_custom_sin_ALPHA,
            noise_density_sampling_method=args.env_noise_density_sampling_method,
        )
        sigma = args.sampling_noise_scale * np.array([1.0])
    else:
        raise ValueError(f"{args.env} not supported")

    env.reset(seed=args.seed)  # seed the environment once initially.
    env.action_space.seed(args.seed)  # separately seed the action space.
    logger.info(f"Noise var: {sigma}")

    # baseline sim density
    if args.baseline_nonlds:
        b_args = copy.copy(args)
        b_args.sampling_density = "normal"
        b_args.sampling_method = "exact"
        b_phi_embedding = LDSPhiEmbedding(d_s, d_a)
        if b_args.W_fit_states_sep:
            b_exp_glm_density = [
                NonLDSDensity(1, sigma[i : i + 1]) for i in range(d_s)
            ]
            b_W_init = [
                torch.zeros(d.d_psi, b_phi_embedding.d_phi)
                for d in b_exp_glm_density
            ]
        else:
            b_exp_glm_density = NonLDSDensity(d_s, sigma)
            b_W_init = torch.zeros(
                b_exp_glm_density.d_psi, b_phi_embedding.d_phi
            )

    # sim density
    phi_embedding = LDSPhiEmbedding(d_s, d_a)
    if args.env == "cartpole":
        if args.W_fit_states_sep:
            # Fit each state separately
            if args.sampling_density == "normal":
                exp_glm_density = [
                    NonLDSDensity(1, sigma[i : i + 1]) for i in range(d_s)
                ]
            elif args.sampling_density == "custom_for_exp_relu":
                exp_glm_density = [
                    CustomForExpReLUDensity(
                        1, sigma[0:1] * args.sampling_noise_scale_mult
                    ),
                    NonLDSDensity(1, sigma[1:2]),
                    NonLDSDensity(1, sigma[2:3]),
                    NonLDSDensity(1, sigma[3:4]),
                ]
            elif args.sampling_density == "custom_for_mog_relu":
                # only use the weird density for the first coordinate.
                exp_glm_density = [
                    CustomForMoGReLUDensity(
                        1, sigma[0:1] * args.sampling_noise_scale_mult
                    ),
                    NonLDSDensity(1, sigma[1:2]),
                    NonLDSDensity(1, sigma[2:3]),
                    NonLDSDensity(1, sigma[3:4]),
                ]
            # shape (d_psi, d_phi)
            W_init = [
                torch.zeros(d.d_psi, phi_embedding.d_phi)
                for d in exp_glm_density
            ]
        else:
            # Fit full system simultaneously
            if args.sampling_density == "normal":
                exp_glm_density = NonLDSDensity(d_s, sigma)
            elif args.sampling_density == "custom_for_exp_relu":
                exp_glm_density = CustomForExpReLUDensity(d_s, sigma)
            W_init = torch.zeros(exp_glm_density.d_psi, phi_embedding.d_phi)

    elif args.env == "custom_sin":
        if args.W_fit_states_sep:
            # Fit each state separately
            if args.sampling_density == "normal":
                exp_glm_density = [
                    NonLDSDensity(1, sigma[i : i + 1]) for i in range(d_s)
                ]
            elif args.sampling_density == "custom_sin":
                exp_glm_density = [
                    CustomSinDensity(
                        1,
                        P=args.env_custom_sin_P,
                        ALPHA=args.env_custom_sin_ALPHA,
                    )
                    for i in range(d_s)
                ]
            # shape (d_psi, d_phi)
            W_init = [
                torch.zeros(d.d_psi, phi_embedding.d_phi)
                for d in exp_glm_density
            ]
        else:
            # Fit full system simultaneously
            if args.sampling_density == "normal":
                exp_glm_density = NonLDSDensity(d_s, sigma)
            elif args.sampling_density == "custom_sin":
                exp_glm_density = CustomSinDensity(
                    1, P=args.env_custom_sin_P, ALPHA=args.env_custom_sin_ALPHA
                )
            W_init = torch.zeros(exp_glm_density.d_psi, phi_embedding.d_phi)

    # Run random actions benchmark
    benchmark_random_actions(args, env)

    # Train simulator

    # Get estimate of W
    sim = (args, exp_glm_density, W_init, phi_embedding)
    base_sim = (
        (b_args, b_exp_glm_density, b_W_init, b_phi_embedding)
        if args.baseline_nonlds
        else None
    )

    W_trained = train(env, sim=sim, base_sim=base_sim)
    if args.baseline_nonlds:
        W_trained, b_W_trained = W_trained

    logger.info(f"W_trained {W_trained}")

    # Evaluate W on fresh simulation
    sim = (args, exp_glm_density, W_trained, phi_embedding)
    base_sim = (
        (b_args, b_exp_glm_density, b_W_trained, b_phi_embedding)
        if args.baseline_nonlds
        else None
    )

    evaluate(env, sim=sim, base_sim=base_sim)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Simulator")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--exp_name", type=str, default="simulate_cartpole")
    parser.add_argument("--num_workers", type=int, default=-1)

    parser.add_argument(
        "--benchmark_random_actions_num_runs",
        type=int,
        default=10,
        help="Num runs to run benchmark",
    )
    parser.add_argument("--baseline_self", action="store_true")
    parser.add_argument("--baseline_nonlds", action="store_true")

    env_args = parser.add_argument_group("cartpole env")
    env_args.add_argument(
        "--env",
        type=str,
        default="cartpole",
        choices=["cartpole", "custom_sin"],
    )
    env_args.add_argument(
        "--env_noise_type",
        type=str,
        default="no",
        choices=StochasticCartPoleEnv.noise_types,
    )
    env_args.add_argument(
        "--env_noise_density_name",
        type=str,
        default="normal",
        choices=StochasticCartPoleEnv.noise_density_names,
    )
    env_args.add_argument(
        "--env_noise_density_sampling_method",
        type=str,
        default="inv_cdf",
        choices=ExpGLMDensity.SAMPLING_METHODS,
    )
    env_args.add_argument(
        "--env_state_noise_scale",
        type=float,
        default=1e-1,
        help="for state noise in real env, scales the noise stdev by this fraction",
    )
    env_args.add_argument("--env_no_render", action="store_true")
    env_args.add_argument(
        "--env_use_cont_action",
        action="store_true",
        help="Flag to use continuous action",
    )
    env_args.add_argument("--env_custom_sin_P", type=float, default=4.0)
    env_args.add_argument("--env_custom_sin_ALPHA", type=float, default=1.7)

    sampling_args = parser.add_argument_group("simulator sampling")
    sampling_args.add_argument(
        "--sampling_density",
        type=str,
        default="normal",
        choices=[
            "normal",
            "custom_sin",
            "custom_for_exp_relu",
            "custom_for_mog_relu",
        ],
    )
    sampling_args.add_argument(
        "--sampling_method", type=str, default="deterministic"
    )
    sampling_args.add_argument(
        "--sampling_kwargs_filepath",
        type=str,
        default="config/cartpole_inv_cdf_sampling_kwargs.json",
    )
    sampling_args.add_argument(
        "--sampling_noise_scale",
        type=float,
        default=1e-1,
        help="for exact sampling, scales the noise stdev by this fraction",
    )
    sampling_args.add_argument(
        "--sampling_noise_scale_mult",
        type=float,
        default=1.0,
        help="extra factor multiplied to the noise stdev",
    )

    sampling_args.add_argument("--mogwidth", default=1, type=float)

    random_shooting_args = parser.add_argument_group("random_shooting")
    random_shooting_args.add_argument(
        "--random_shooting_H",
        type=int,
        default=200,
        help="Length of horizon to plan for",
    )
    random_shooting_args.add_argument(
        "--random_shooting_tau",
        type=int,
        default=10,
        help="Lookahead for random shooting",
    )
    random_shooting_args.add_argument(
        "--random_shooting_num_trials",
        type=int,
        default=100,
        help="Num random shoots to simulate",
    )
    random_shooting_args.add_argument(
        "--random_shooting_all_ones_actions",
        action="store_true",
        help="play [1,1,...] and [-1,-1,...]",
    )

    W_estimator_args = parser.add_argument_group("W_estimator")
    W_estimator_args.add_argument(
        "--W_fit_states_sep",
        action="store_true",
        help="Fits each state separately",
    )
    W_estimator_args.add_argument("--W_use_MLE_estimator", action="store_true")
    W_estimator_args.add_argument("--W_sm_lam", default=1e-2)

    train_args = parser.add_argument_group("training")
    train_args.add_argument("--train_num_episodes", type=int, default=50)

    args = parser.parse_args()
    main(args)

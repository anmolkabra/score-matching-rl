"""
s' ~ M(s, a) where M is the CustomSinDensity
"""
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np

from data_gen.density import CustomSinDensity
from data_gen.phi_embedding import LDSPhiEmbedding
from gym_env.stochastic_cartpole import RademacherDiscreteAction


class CustomSinMDPEnv(gym.Env[float, float]):
    def __init__(
        self,
        P: float,
        ALPHA: float,
        noise_density_sampling_method: str = "inv_cdf",
        noise_density_sampling_kwargs: Dict[str, Any] = {
            "inv_cdf_linspace_L": 1000,
            "inv_cdf_dicts": ({}, {}, {}),
            # Init integ_vals_dict, cdf_supp_dict, cdf_dict
            "inv_cdf_round_decimals": 2,
        },
    ):
        self.P = P
        self.ALPHA = ALPHA
        self.noise_density_sampling_method = noise_density_sampling_method
        self.noise_density_sampling_kwargs = noise_density_sampling_kwargs

        self.action_space = RademacherDiscreteAction()

        # Bounds at which fail the episode
        self.high = np.array([2.0], dtype=np.float32)
        self.d_s = 1

        # Define MDP
        self.density = CustomSinDensity(
            d_s=self.d_s, P=self.P, ALPHA=self.ALPHA
        )
        self.phi_embedding = LDSPhiEmbedding(d_s=self.d_s, d_a=1)
        self.W = np.array([[1.0, 1.0]])  # shape (d_psi = d_s = 1, d_phi = 2)

        self.state = None
        self.steps_beyond_done = None

    def step(
        self, action: float, state_query: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        # modified code: the step function allows for an optional query
        # of the model from any (s,a) pair.
        # another way to do this is to deep copy the whole environment
        # but this might be slow: https://stackoverflow.com/questions/57839665/how-to-set-a-openai-gym-environment-start-with-a-specific-state-not-the-env-res

        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert len(self.state) == self.d_s
        if state_query is None:
            assert (
                self.state is not None
            ), "Call reset before using step method."
            s = self.state[None, :]  # shape (1, d_s)
        else:
            s = state_query[None, :]  # shape (1, d_s)
        a = np.array(action).reshape((1, 1))  # shape (1, d_a)
        next_state, phi = self.density.sample_iid(
            self.W,
            s,
            a,
            self.phi_embedding,
            1,
            sampling_method=self.noise_density_sampling_method,
            sampling_kwargs=self.noise_density_sampling_kwargs,
        )  # shape (1, 1, d_s), (1, d_phi)
        next_state = next_state.numpy()[0, 0]
        actual_next_state = next_state.copy()

        if state_query is None:
            # actually step forward the system.
            self.state = next_state

        done = False
        # sharp peaky rewards at the peaks of the density
        reward = np.exp(-10 * (s[0, 0] - np.pi / (2 * self.P)) ** 2) + np.exp(
            -10 * (s[0, 0] + 3 * np.pi / (2 * self.P)) ** 2
        )

        return (
            next_state,
            reward,
            done,
            {"actual_next_state": actual_next_state},
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = 2 * np.random.rand(self.d_s) - 1
        self.steps_beyond_done = None
        if not return_info:
            return self.state
        else:
            return self.state, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass

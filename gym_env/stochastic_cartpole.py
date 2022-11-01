"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Any, Dict, Optional, Union

import gym
import numpy as np
from gym import logger, spaces

import const
from data_gen.density import ExpDensity, NonLDSDensity, CustomSinDensity
from data_gen.phi_embedding import ConstPhiEmbedding, LDSPhiEmbedding


class RademacherDiscreteAction(spaces.Discrete):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(2, seed=seed, start=0)

    def sample(self) -> int:
        return 2 * super().sample() - 1

    def contains(self, x) -> bool:
        return super().contains((x + 1) // 2)

    def __repr__(self) -> str:
        return "Discrete({-1, 1})"

    def __eq__(self, other) -> bool:
        return isinstance(other, RademacherDiscreteAction)


class StochasticCartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Description

    This environment corresponds to the version of the cart-pole problem
    described by Barto, Sutton, and Anderson in ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a
    frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.

    ### Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction of the fixed force the cart is pushed with.

    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ### Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                  | Max                |
    |-----|-----------------------|----------------------|--------------------|
    | 0   | Cart Position         | -4.8                 | 4.8                |
    | 1   | Cart Velocity         | -Inf                 | Inf                |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°)  | ~ 0.418 rad (24°)  |
    | 3   | Pole Angular Velocity | -Inf                 | Inf                |

    **Note:** While the ranges above denote the possible values for observation space of each element, it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ### Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken, including the termination step, is allotted. The threshold for rewards is 475 for v1.

    ### Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ### Episode Termination

    The episode terminates if any one of the following occurs:
    1. Pole Angle is greater than ±12°
    2. Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Episode length is greater than 500 (200 for v0)

    ### Arguments

    ```
    gym.make('CartPole-v1')
    ```

    No additional arguments are currently supported.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    noise_types = ["no", "in_action", "in_next_state"]
    noise_density_names = ["normal", "exp", "custom_sin", "mog"]

    def __init__(
        self,
        use_cont_action: bool,
        noise_type: str = "in_next_state",
        noise_density_name: str = "normal",
        noise_density_sampling_method: str = "inv_cdf",
        noise_density_sampling_kwargs: Dict[str, Any] = {
            "inv_cdf_linspace_L": 1000,
            "inv_cdf_dicts": ({}, {}, {}),
            # Init integ_vals_dict, cdf_supp_dict, cdf_dict
            "inv_cdf_round_decimals": 2,
        },
        noise_stdev: float = 1e-1,
        noise_action_flip_prob: float = 0.4,
        mog_width: float = 1,
    ):
        assert noise_type in self.noise_types, f"{noise_type} invalid."
        assert (
            noise_density_name in self.noise_density_names
        ), f"{noise_density_name} invalid."

        self.noise_type = noise_type
        self.noise_density_name = noise_density_name
        self.noise_density_sampling_method = noise_density_sampling_method
        self.noise_density_sampling_kwargs = noise_density_sampling_kwargs
        self.noise_stdev = noise_stdev
        self.noise_action_flip_prob = noise_action_flip_prob
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        self.use_cont_action = use_cont_action
        if self.use_cont_action:
            self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        else:
            self.action_space = RademacherDiscreteAction()

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.d_s = 4

        eps_sigma = (
            self.noise_stdev
            * np.array(
                [
                    self.x_threshold,
                    (self.tau * self.force_mag / self.total_mass),
                    self.theta_threshold_radians,
                    (self.tau * self.gravity / self.length),
                ]
            )
        ) ** 2

        # Init noise creators
        if self.noise_density_name == "normal":
            self.noise_density = NonLDSDensity(self.d_s, sigma=eps_sigma)
            self.noise_density_W = np.zeros(
                (self.d_s, self.d_s + 1)
            )  # phi = [s, a]
            self.noise_density_phi_embedding = LDSPhiEmbedding(self.d_s, 1)
        elif self.noise_density_name == "exp":
            # Noise only for the first state
            self.noise_density = ExpDensity(1)
            # shape (d_psi, d_phi) where d_psi == d_phi == d_s
            # exp(lmb) density has variance 1 / lmb^2
            self.noise_density_W = (
                -1.0 / np.sqrt(eps_sigma[0:1] + const.DIV_EPS)[:, None]
            )
            self.noise_density_phi_embedding = ConstPhiEmbedding(1, 1, 1.0)
        elif self.noise_density_name == "custom_sin":
            self.noise_density = []
            self.noise_density_W = []
            for i in range(self.d_s):
                P = 1.0 / np.sqrt(eps_sigma[i])
                d = CustomSinDensity(1, P=P)
                Wi = np.ones((d.d_psi, self.d_s + 1))  # phi = [s, a]
                self.noise_density.append(d)
                self.noise_density_W.append(Wi)
            self.noise_density_phi_embedding = LDSPhiEmbedding(self.d_s, 1)
        elif self.noise_density_name == "mog":
            self.eps_sigma = eps_sigma
            self.eps_sigma_onlyfirst = np.array([eps_sigma[0], 0, 0, 0])
            logger.info("USED EPS SIGMA: ", self.eps_sigma_onlyfirst)
            self.mog_threshold = 0.5  # I have hard coded this for now.
            self.mog_width = mog_width

        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_done = None

    def step(self, action, state_query: Optional[np.ndarray] = None):
        # modified code: the step function allows for an optional query
        # of the model from any (s,a) pair.
        # another way to do this is to deep copy the whole environment
        # but this might be slow: https://stackoverflow.com/questions/57839665/how-to-set-a-openai-gym-environment-start-with-a-specific-state-not-the-env-res

        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        if state_query is None:
            assert (
                self.state is not None
            ), "Call reset before using step method."
            x, x_dot, theta, theta_dot = self.state.tolist()
        else:
            # initialize the system at the state query.
            x, x_dot, theta, theta_dot = state_query.tolist()
        force = float(action * self.force_mag)
        if (self.noise_type == "in_action") and (
            np.random.rand(1) < self.noise_action_flip_prob
        ):
            force = -force

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = (
            temp - self.polemass_length * thetaacc * costheta / self.total_mass
        )

        prev_state = (x, x_dot, theta, theta_dot)

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        next_state = (x, x_dot, theta, theta_dot)

        if self.noise_type in ["in_action", "in_next_state"]:
            s = np.array(prev_state)[None, :]  # shape (1, d_s)
            a = np.array(action).reshape(
                (1, 1)
            )  # shape (1, d_a) where d_a == 1
            if self.noise_density_name == "mog":
                # only add noise to the first coordinate.
                if np.random.uniform() < self.mog_threshold:
                    next_state += np.random.randn(self.d_s) * np.sqrt(
                        self.eps_sigma_onlyfirst
                    )
                else:
                    next_state += (
                        np.random.randn(self.d_s)
                        * np.sqrt(self.eps_sigma_onlyfirst)
                        + np.sqrt(self.eps_sigma_onlyfirst) * self.mog_width
                    )
            elif self.noise_density_name == "exp":
                # only add noise to the first coordinate.
                # only pass in s[:, :1] as exp noise is independent of state
                eps, phi = self.noise_density.sample_iid(
                    self.noise_density_W,
                    s[:, :1],
                    a,
                    self.noise_density_phi_embedding,
                    1,
                    sampling_method=self.noise_density_sampling_method,
                    sampling_kwargs=self.noise_density_sampling_kwargs,
                )  # shape (1, 1, 1), (1, d_phi) where d_phi = d_s+d_a
                eps = np.array([eps[0, 0].item(), 0.0, 0.0, 0.0])
                next_state += eps
            else:
                if isinstance(self.noise_density, list):
                    # Different noise model for each state
                    eps = []
                    for i, (d, Wi) in enumerate(
                        zip(self.noise_density, self.noise_density_W)
                    ):
                        eps_i, phi_i = d.sample_iid(
                            Wi,
                            s,
                            a,
                            self.noise_density_phi_embedding,
                            1,
                            sampling_method=self.noise_density_sampling_method,
                            sampling_kwargs=self.noise_density_sampling_kwargs,
                        )  # shape (1, 1, 1), (1, d_phi) where d_phi = d_s+d_a
                        eps.append(eps_i[0, 0].numpy())
                    next_state += np.hstack(eps)  # shape (d_s)
                else:
                    eps, phi = self.noise_density.sample_iid(
                        self.noise_density_W,
                        s,
                        a,
                        self.noise_density_phi_embedding,
                        1,
                        sampling_method=self.noise_density_sampling_method,
                        sampling_kwargs=self.noise_density_sampling_kwargs,
                    )  # shape (1, 1, d_s), (1, d_phi) where d_phi = d_s+d_a
                    next_state += eps.numpy()[0, 0]

        if state_query is None:
            # actually step forward the system.
            self.state = next_state

        done = bool(
            next_state[0] < -self.x_threshold
            or next_state[0] > self.x_threshold
            or next_state[2] < -self.theta_threshold_radians
            or next_state[2] > self.theta_threshold_radians
        )

        if state_query is None:
            if not done:
                reward = 1.0
            elif self.steps_beyond_done is None:
                # Pole just fell!
                self.steps_beyond_done = 0
                reward = 1.0
            else:
                if self.steps_beyond_done == 0:
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned done = True. You "
                        "should always call 'reset()' once you receive 'done = "
                        "True' -- any further steps are undefined behavior."
                    )
                self.steps_beyond_done += 1
                reward = 0.0
        else:
            # simplified reward structure for the state query.
            # there might be some issues here with the system simulation
            # if the state query is already "invalid", i.e. has done=1.
            # then the next state might not be valid.
            # but maybe the physics equations will still work out
            if not done:
                reward = 1.0
            else:
                reward = 0.0

        return np.array(next_state, dtype=np.float32), reward, done, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(
            low=-0.05, high=0.05, size=(self.d_s,)
        )
        self.steps_beyond_done = None
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}

    def render(self, mode="human"):
        import pygame
        from pygame import gfxdraw

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = (
            -cartwidth / 2,
            cartwidth / 2,
            cartheight / 2,
            -cartheight / 2,
        )
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

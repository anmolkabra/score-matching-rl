import abc
import collections
import copy
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import hamiltorch
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import torch
from scipy.optimize import minimize

import const
import utils
from data_gen.phi_embedding import PhiEmbedding

T = Union[np.ndarray, torch.Tensor]
INTEG_EPS = 1e-2  # eps error for scipy.integrate.quad


class ExpGLMDensity(abc.ABC):
    """
    Estimate density of P_W (x | s, a) modeled as an Exponential GLM.
    """

    SAMPLING_METHODS = (
        "exact",
        "hmc",
        "hmc_nuts",
        "implicit_rmhmc",
        "explicit_rmhmc",
        "inv_cdf",
    )

    def __init__(
        self,
        d_s: int,
        use_scipy_integrate: float = False,
        integrate_lo: float = -5.0,
        integrate_hi: float = 5.0,
        integrate_linspace_L: int = 1000,
    ):
        """
        Args:
            d_s: state space dimension.
        """
        self.d_s = d_s

        self.use_scipy_integrate = use_scipy_integrate
        self.integrate_lo = integrate_lo
        self.integrate_hi = integrate_hi
        self.integrate_linspace_L = integrate_linspace_L

    def __repr__(self):
        return "$q(s') \\cdot \\exp ( \\langle \\psi(s'), W \\phi(s, a) \\rangle - Z_{{s, a}}(W) )$"

    @property
    @abc.abstractmethod
    def SUPPORT_LO(self) -> Union[float, T]:
        """
        Defines the lower end of support for each dimension.
        """
        pass

    @property
    @abc.abstractmethod
    def SUPPORT_HI(self) -> Union[float, T]:
        """
        Defines the upper end of support for each dimension.
        """
        pass

    @property
    def IS_PRODUCT_DENSITY(self) -> bool:
        """
        Returns a bool indicating whether the density is a product density.
        If the density is a product density, inverse CDF sampling is possible.
        """
        return False

    @abc.abstractmethod
    def in_support(self, x: T) -> torch.Tensor:
        """
        Returns a boolean tensor if points xi are in support (non-zero density).

        Args:
            x: points to check for support. shape (n, d_s) where n is the number
                of points.

        Returns:
            shape (n)
        """
        pass

    def q(self, x: T) -> torch.Tensor:
        """
        Calculates q for each data point in x.

        Args:
            x: points to evaluate on. shape (n, d_s) where n is the number of
                points.

        Returns:
            shape (n)
        """
        return torch.exp(self.logq(x))

    @abc.abstractmethod
    def logq(self, x: T) -> torch.Tensor:
        """
        Calculates log q for each data point in x.

        Args:
            x: points to evaluate on. shape (n, d_s) where n is the number of
                points.

        Returns:
            shape (n)
        """
        pass

    @abc.abstractmethod
    def psi(self, x: T) -> torch.Tensor:
        """
        Calculates psi for each data point in x.

        Args:
            x: points to evaluate on. shape (n, d_s) where n is the number of
                points.

        Returns:
            shape (n, d_psi)
        """
        pass

    def _check_uni_or_prod_density(
        self,
        d_s: Optional[int] = None,
        IS_PRODUCT_DENSITY: Optional[bool] = None,
        W: Optional[T] = None,
    ) -> bool:
        """
        Returns true if density is univariate or multivariate-product. For the
        unspecified parameters, the object's attributes are used.
        """
        d_s = self.d_s if d_s is None else d_s
        IS_PRODUCT_DENSITY = (
            self.IS_PRODUCT_DENSITY
            if IS_PRODUCT_DENSITY is None
            else IS_PRODUCT_DENSITY
        )

        if W is None:
            return (d_s == 1) or IS_PRODUCT_DENSITY
        else:
            return (d_s == 1) or (IS_PRODUCT_DENSITY and (W.shape[0] == d_s))

    # Define integrand for log_part
    def unnorm_px(self, x: float, v: T) -> float:
        """Compute normalization constant for density with prior v."""
        x_tens = torch.tensor(x).reshape((1, 1))
        log_prob = self.log_density_v(x_tens, v)  # shape (1, 1)
        return torch.exp(log_prob).item()

    # Define integrand for log_part
    def unnorm_px_tensor(self, x: T, v: T) -> torch.Tensor:
        """Compute normalization constant for density with prior v."""
        log_prob = self.log_density_v(x, v)  # shape (n, m)
        return torch.exp(log_prob)

    def logpart_v(self, v: T) -> torch.Tensor:
        """
        Calculates the log partition function for the parameters v.

        If the density is a product distribution, then the log partition
        function for each dimension is returned. If not, and if d_s > 1, then
        the log-partition is intractable.

        Args:
            v: parameter for evaluating the log partition function. shape
                (m, d_psi)

        Raises:
            AssertionError: If density is not tractable, when not (univariate or
                multivariate-product).

        Returns:
            shape (m, d_s)
        """
        # Passing v.T for W is alright as the func checks for W.shape[0] == d_s
        assert self._check_uni_or_prod_density(
            W=v.T
        ), "Log-partition only tractable for univariate or multivariate-product densities."

        m = v.shape[0]

        if self.d_s == 1:
            if self.use_scipy_integrate:
                # Determine integration bounds
                support_lo = self.SUPPORT_LO
                support_hi = self.SUPPORT_HI

                log_part_v = torch.log(
                    torch.tensor(
                        [
                            scipy.integrate.quad(
                                self.unnorm_px,
                                support_lo,
                                support_hi,
                                args=(v[j : j + 1]),
                                epsabs=INTEG_EPS,
                                epsrel=INTEG_EPS,
                            )[0]
                            for j in range(m)
                        ]
                    )
                ).unsqueeze(
                    1
                )  # shape (m, d_s) where d_s == 1
            else:
                # Determine integration bounds
                support_lo = self.integrate_lo
                support_hi = self.integrate_hi

                # Compute log_part_v
                x = torch.linspace(
                    support_lo, support_hi, self.integrate_linspace_L
                )
                px = self.unnorm_px_tensor(x[:, None], v)  # shape (n, m)
                dx = x[1:] - x[:-1]
                px_mid = (px[1:] + px[:-1]) / 2
                log_part_v = torch.log(torch.sum(px_mid.T * dx, dim=1))[
                    :, None
                ]  # shape (m, d_s)
        else:
            # product density, d_psi == d_s
            log_part_v = torch.zeros(m, self.d_s)
            if self.use_scipy_integrate:
                for i in range(self.d_s):
                    # Determine integration bounds for dimension i
                    support_lo = (
                        self.SUPPORT_LO
                        if isinstance(self.SUPPORT_LO, float)
                        else utils.to_torch(self.SUPPORT_LO)[i].item()
                    )
                    support_hi = (
                        self.SUPPORT_HI
                        if isinstance(self.SUPPORT_HI, float)
                        else utils.to_torch(self.SUPPORT_HI)[i].item()
                    )

                    # Compute log_part_v
                    log_part_v[:, i] = torch.log(
                        torch.tensor(
                            [
                                scipy.integrate.quad(
                                    self.unnorm_px,
                                    support_lo,
                                    support_hi,
                                    args=(v[j : j + 1, i : i + 1]),
                                    epsabs=INTEG_EPS,
                                    epsrel=INTEG_EPS,
                                )[0]
                                for j in range(m)
                            ]
                        )
                    )  # shape (m)
            else:
                for i in range(self.d_s):
                    # Determine integration bounds for dimension i
                    support_lo = (
                        self.integrate_lo
                        if isinstance(self.integrate_lo, float)
                        else utils.to_torch(self.integrate_lo)[i].item()
                    )
                    support_hi = (
                        self.integrate_hi
                        if isinstance(self.integrate_hi, float)
                        else utils.to_torch(self.integrate_hi)[i].item()
                    )

                    # Compute log_part_v
                    x = torch.linspace(
                        support_lo, support_hi, self.integrate_linspace_L
                    )
                    px = self.unnorm_px_tensor(
                        x[:, None], v[:, i : i + 1]
                    )  # shape (n, m)
                    dx = x[1:] - x[:-1]
                    px_mid = (px[1:] + px[:-1]) / 2
                    log_part_v[:, i] = torch.log(
                        torch.sum(px_mid.T * dx, dim=1)
                    )  # shape (m)
        return log_part_v

    def logpart(self, W: T, phi: T) -> torch.Tensor:
        """
        Calculates the log partition function for each embedding phi(s, a) of
        the prior tuple (s, a).

        If the density is a product distribution, then the log partition
        function for each dimension is returned. If not, and if d_s > 1, then
        the log-partition is intractable.

        Args:
            W: parameter for evaluating the log partition function. shape
                (d_psi, d_phi)
            phi: value of phi(s, a). shape (m, d_phi) where m is the number of
                prior (s, a) points.

        Raises:
            AssertionError: If density is not tractable, when not (univariate or
                multivariate-product).

        Returns:
            shape (m, d_s)
        """
        assert self._check_uni_or_prod_density(
            W=W
        ), "Log-partition only tractable for univariate or multivariate-product densities."
        W = utils.to_torch(W)
        phi = utils.to_torch(phi)

        v = phi @ W.T  # shape (m, d_psi)
        return self.logpart_v(v)  # shape (m, d_s)

    # Define integrand for mean
    def xpx(self, x: float, W: T, phi: T, log_part: T) -> float:
        """Computes x*p(x; v) where v = phi @ W.T."""
        x_tens = torch.tensor(x).reshape((1, 1))
        return x * self.density(x_tens, W, phi, log_part=log_part).item()

    # Define integrand for log_part
    def xpx_tensor(self, x: T, W: T, phi: T, log_part: T) -> torch.Tensor:
        """Computes x*p(x; v) where v = phi @ W.T."""
        return x * self.density(x, W, phi, log_part=log_part)  # shape (n, m)

    def density_mean(
        self, W: T, phi: T, log_part: Optional[T] = None
    ) -> torch.Tensor:
        """
        Returns the mean of the densities specified by parameters v. If the
        density is multivariate, then the mean is defined along individual
        dimensions (i.e. the density must be product).

        Args:
            W: parameter for evaluating the log partition function. shape
                (d_psi, d_phi)
            phi: value of phi(s, a). shape (m, d_phi) where m is the number of
                prior (s, a) points.
            log_part: (default None) log partition values for the densities
                parameterized by the priors. shape (m, d_s). If None, it is
                calculated.

        Raises:
            AssertionError: If density is not tractable, when not (univariate or
                multivariate-product).

        Returns:
            shape (m, d_s)
        """
        assert self._check_uni_or_prod_density(
            W=W
        ), "Mean only defined for univariate or multivariate-product densities."

        m = phi.shape[0]
        if log_part is None:
            log_part = self.logpart(W, phi)  # shape (m, d_s)

        if self.d_s == 1:
            if self.use_scipy_integrate:
                # Determine integration bounds
                support_lo = float(self.SUPPORT_LO)
                support_hi = float(self.SUPPORT_HI)

                # Compute mean
                mean = torch.tensor(
                    [
                        scipy.integrate.quad(
                            self.xpx,
                            support_lo,
                            support_hi,
                            args=(W, phi[j : j + 1], log_part[j : j + 1]),
                            epsabs=INTEG_EPS,
                            epsrel=INTEG_EPS,
                        )[0]
                        for j in range(m)
                    ]
                ).unsqueeze(
                    1
                )  # shape (m, d_s) where d_s = 1
            else:
                # Determine integration bounds
                support_lo = self.integrate_lo
                support_hi = self.integrate_hi

                # Compute log_part_v
                x = torch.linspace(
                    support_lo, support_hi, self.integrate_linspace_L
                )
                xpx = self.xpx_tensor(
                    x[:, None], W, phi, log_part
                )  # shape (n, m)
                dx = x[1:] - x[:-1]
                xpx_mid = (xpx[1:] + xpx[:-1]) / 2
                mean = torch.sum(xpx_mid.T * dx, dim=1)[
                    :, None
                ]  # shape (m, d_s)
        else:
            # product density, d_psi == d_s
            mean = torch.zeros(m, self.d_s)
            if self.use_scipy_integrate:
                for i in range(self.d_s):
                    # Determine integration bounds for dimension i
                    support_lo = (
                        self.SUPPORT_LO
                        if isinstance(self.SUPPORT_LO, float)
                        else utils.to_torch(self.SUPPORT_LO)[i].item()
                    )
                    support_hi = (
                        self.SUPPORT_HI
                        if isinstance(self.SUPPORT_HI, float)
                        else utils.to_torch(self.SUPPORT_HI)[i].item()
                    )

                    # Compute mean
                    mean[:, i] = torch.tensor(
                        [
                            scipy.integrate.quad(
                                self.xpx,
                                support_lo,
                                support_hi,
                                args=(
                                    W[i : i + 1],
                                    phi[j : j + 1],
                                    log_part[j : j + 1, i : i + 1],
                                ),
                                epsabs=INTEG_EPS,
                                epsrel=INTEG_EPS,
                            )[0]
                            for j in range(m)
                        ]
                    )  # shape (m)
            else:
                for i in range(self.d_s):
                    # Determine integration bounds for dimension i
                    support_lo = (
                        self.integrate_lo
                        if isinstance(self.integrate_lo, float)
                        else utils.to_torch(self.integrate_lo)[i].item()
                    )
                    support_hi = (
                        self.integrate_hi
                        if isinstance(self.integrate_hi, float)
                        else utils.to_torch(self.integrate_hi)[i].item()
                    )

                    # Compute log_part_v
                    x = torch.linspace(
                        support_lo, support_hi, self.integrate_linspace_L
                    )
                    xpx = self.xpx_tensor(
                        x[:, None], W[i : i + 1], phi, log_part[:, i : i + 1]
                    )  # shape (n, m)
                    dx = x[1:] - x[:-1]
                    xpx_mid = (xpx[1:] + xpx[:-1]) / 2
                    mean[:, i] = torch.sum(xpx_mid.T * dx, dim=1)  # shape (m)
        return mean

    # Define integrand for variance
    def xxpx(self, x: float, W: T, phi: T, log_part: T) -> float:
        """Computes x^2 * p(x; v) where v = phi @ W.T."""
        x_tens = torch.tensor(x).reshape((1, 1))
        return x * x * self.density(x_tens, W, phi, log_part=log_part).item()

    # Define integrand for variance
    def xxpx_tensor(self, x: T, W: T, phi: T, log_part: T) -> torch.Tensor:
        """Computes x^2 * p(x; v) where v = phi @ W.T."""
        return (
            x * x * self.density(x, W, phi, log_part=log_part)
        )  # shape (n, m)

    def density_var(
        self,
        W: T,
        phi: T,
        mean: Optional[T] = None,
        log_part: Optional[T] = None,
    ) -> torch.Tensor:
        """
        Returns the variance of the densities specified by parameters v. If the
        density is multivariate, then the variance is defined along individual
        dimensions (i.e. the density must be product).

        Args:
            W: parameter for evaluating the log partition function. shape
                (d_psi, d_phi)
            phi: value of phi(s, a). shape (m, d_phi) where m is the number of
                prior (s, a) points.
            mean: (default None) computed mean of the density for each of m
                priors. shape (m, d_s)
            log_part: (default None) log partition values for the densities
                parameterized by the priors. shape (m, d_s). If None, it is
                calculated.

        Raises:
            AssertionError: If density is not tractable, when not (univariate or
                multivariate-product).

        Returns:
            shape (m, d_s)
        """
        assert self._check_uni_or_prod_density(
            W=W
        ), "Variance only defined for univariate or multivariate-product densities."

        m = phi.shape[0]
        if log_part is None:
            log_part = self.logpart(W, phi)  # shape (m, d_s)
        if mean is None:
            mean = self.density_mean(
                W, phi, log_part=log_part
            )  # shape (m, d_s)

        if self.d_s == 1:
            if self.use_scipy_integrate:
                # Determine integration bounds
                support_lo = float(self.SUPPORT_LO)
                support_hi = float(self.SUPPORT_HI)

                # Compute mean
                var = torch.tensor(
                    [
                        scipy.integrate.quad(
                            self.xxpx,
                            support_lo,
                            support_hi,
                            args=(W, phi[j : j + 1], log_part[j : j + 1]),
                            epsabs=INTEG_EPS,
                            epsrel=INTEG_EPS,
                        )[0]
                        for j in range(m)
                    ]
                ).unsqueeze(1) - (
                    mean**2
                )  # shape (m, d_s) where d_s = 1
            else:
                # Determine integration bounds
                support_lo = self.integrate_lo
                support_hi = self.integrate_hi

                # Compute log_part_v
                x = torch.linspace(
                    support_lo, support_hi, self.integrate_linspace_L
                )
                xxpx = self.xxpx_tensor(
                    x[:, None], W, phi, log_part
                )  # shape (n, m)
                dx = x[1:] - x[:-1]
                xxpx_mid = (xxpx[1:] + xxpx[:-1]) / 2
                var = torch.sum(xxpx_mid.T * dx, dim=1)[:, None] - (
                    mean**2
                )  # shape (m, d_s)
        else:
            # product density, d_psi == d_s
            var = torch.zeros(m, self.d_s)
            if self.use_scipy_integrate:
                for i in range(self.d_s):
                    # Determine integration bounds for dimension i
                    support_lo = (
                        self.SUPPORT_LO
                        if isinstance(self.SUPPORT_LO, float)
                        else utils.to_torch(self.SUPPORT_LO)[i].item()
                    )
                    support_hi = (
                        self.SUPPORT_HI
                        if isinstance(self.SUPPORT_HI, float)
                        else utils.to_torch(self.SUPPORT_HI)[i].item()
                    )

                    # Compute mean
                    var[:, i] = torch.tensor(
                        [
                            scipy.integrate.quad(
                                self.xxpx,
                                support_lo,
                                support_hi,
                                args=(
                                    W[i : i + 1],
                                    phi[j : j + 1],
                                    log_part[j : j + 1, i : i + 1],
                                ),
                                epsabs=INTEG_EPS,
                                epsrel=INTEG_EPS,
                            )[0]
                            for j in range(m)
                        ]
                    ) - (
                        mean[:, i] ** 2
                    )  # shape (m)
            else:
                for i in range(self.d_s):
                    # Determine integration bounds for dimension i
                    support_lo = (
                        self.integrate_lo
                        if isinstance(self.integrate_lo, float)
                        else utils.to_torch(self.integrate_lo)[i].item()
                    )
                    support_hi = (
                        self.integrate_hi
                        if isinstance(self.integrate_hi, float)
                        else utils.to_torch(self.integrate_hi)[i].item()
                    )

                    # Compute log_part_v
                    x = torch.linspace(
                        support_lo, support_hi, self.integrate_linspace_L
                    )
                    xxpx = self.xxpx_tensor(
                        x[:, None], W[i : i + 1], phi, log_part[:, i : i + 1]
                    )  # shape (n, m)
                    dx = x[1:] - x[:-1]
                    xxpx_mid = (xxpx[1:] + xxpx[:-1]) / 2
                    var[:, i] = torch.sum(xxpx_mid.T * dx, dim=1) - (
                        mean[:, i] ** 2
                    )  # shape (m)
        # integration might give bad values
        var[var <= 0.0] = 1.0
        var[torch.isnan(var)] = 1.0
        return var

    def density(
        self, x: T, W: T, phi: T, log_part: Optional[T] = None
    ) -> torch.Tensor:
        """
        Given m embeddings of prior points (s, a), returns the density
        evaluated at points in x.

        Args:
            x: points to evaluate on. shape (n, d_s) where n is the number of
                points.
            W: parameter W for exponential GLM. shape (d_psi, d_phi)
            phi: value of phi(s, a). shape (m, d_phi) where m is the number of
                priors (s, a).
            log_part: (default None) log partition values for the densities
                parameterized by the priors. shape (m, d_s). If None, it is
                calculated.

        Raises:
            AssertionError: If density is not tractable, when not (univariate or
                multivariate-product).

        Returns:
            shape (n, m)
        """
        assert self._check_uni_or_prod_density(
            W=W
        ), "Density only tractable for univariate or multivariate-product densities."
        assert x.shape[1] == self.d_s
        assert W.shape[1] == phi.shape[1]

        x = utils.to_torch(x)
        W = utils.to_torch(W)
        phi = utils.to_torch(phi)
        n = x.shape[0]
        m = phi.shape[0]

        logq = self.logq(x)  # shape (n)
        psi = self.psi(x)  # shape (n, d_psi)
        if log_part is None:
            log_part = self.logpart(W, phi)  # shape (m, d_s)

        if self.d_s == 1:
            v = phi @ W.T  # shape (m, d_psi)
            psi_v = psi @ v.T  # shape (n, m)
            log_prob = ((psi_v - log_part[:, 0]).T + logq).T  # shape (n, m)
        else:
            # product density, d_psi == d_s
            log_prob = torch.zeros(n, m)
            for i in range(self.d_s):
                vi = phi @ W[i : i + 1].T  # shape (m, 1)
                psi_vi = psi[:, i : i + 1] @ vi.T  # shape (n, m)

                log_prob += psi_vi - log_part[:, i]
            log_prob = (log_prob.T + logq).T  # shape (n, m)
        prob = torch.exp(log_prob)  # shape (n, m)
        prob[torch.logical_not(self.in_support(x))] = 0.0
        return prob

    def log_density(self, x: T, W: T, phi: T) -> torch.Tensor:
        """
        Given m embeddings of prior points (s, a), returns the log of density
        density at points in x. The density is not guaranteed to be normalized
        using the partition function (in cases where the partition function is
        not tractable).

        This function is used for sampling from the represented density.

        Args:
            x: points to evaluate on. shape (n, d_s) where n is the number of
                points.
            W: parameter W for exponential GLM. shape (d_psi, d_phi)
            phi: value of phi(s, a). shape (m, d_phi) where m is the number of
                priors (s, a).

        Returns:
            shape (n, m)
        """
        v = phi @ W.T  # shape (m, d_psi)
        log_prob = self.log_density_v(x, v)  # shape (n, m)
        return log_prob

    def log_density_v(self, x: T, v: T) -> torch.Tensor:
        """
        Given m priors, returns the log of density density at points in x. The
        density is not guaranteed to be normalized using the partition function
        (in cases where the partition function is not tractable).

        This function is used for sampling from the represented density.

        Args:
            x: points to evaluate on. shape (n, d_s) where n is the number of
                points.
            v: parameters, i.e. priors defining the density. shape (m, d_psi)

        Returns:
            shape (n, m)
        """
        assert x.shape[1] == self.d_s

        x = utils.to_torch(x)
        v = utils.to_torch(v)

        logq = self.logq(x)  # shape (n)
        psi = self.psi(x)  # shape (n, d_psi)

        psi_v = psi @ v.T  # shape (n, m)
        log_prob = (psi_v.T + logq).T  # shape (n, m)
        log_prob[torch.logical_not(self.in_support(x))] = -torch.inf
        return log_prob

    def sample_iid(
        self,
        W: T,
        s: T,
        a: T,
        phi_embedding: PhiEmbedding,
        num_iid_samples: int,
        sampling_method: str = "inv_cdf",
        sampling_kwargs: Dict[str, Any] = {
            "inv_cdf_linspace_L": 1000,
            "inv_cdf_dicts": ({}, {}, {}),
            # Init integ_vals_dict, cdf_supp_dict, cdf_dict
            "inv_cdf_round_decimals": 2,
        },
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples num_iid_samples iid samples from the exponential GLM
        parameterized by W conditioned on phi(s[i], a[i]) for each prior i.

        Args:
            W: parameter W for exponential GLM. shape (d_psi, d_phi)
            s: prior states. shape (m, d_s) where m is the number of priors.
            a: prior actions. shape (m, d_a) where m is the number of priors.
            phi_embedding: Instance to calculate phi(s, a). Must output phi of
                dimension d_phi.
            num_iid_samples: Number of iid samples for each prior.
            sampling_method: (default 'inv_cdf') Must be one of
                `SAMPLING_METHODS`.

        Returns:
            samples, phi: iid samples and phi(s, a)
                shapes (m, num_iid_samples, d_s), (m, d_phi)
        """
        assert s.shape[0] == a.shape[0]
        assert W.shape[1] == phi_embedding.d_phi

        samples, phi = self._draw_n_samples(
            W,
            s,
            a,
            phi_embedding,
            num_iid_samples,
            sampling_method,
            sampling_kwargs,
        )  # shape (m, num_iid_samples, d_s), (m, d_phi)
        return samples, phi

    def sample_seq(
        self,
        W: T,
        s: T,
        a: T,
        phi_embedding: PhiEmbedding,
        num_seq_samples: int,
        sampling_method: str = "inv_cdf",
        sampling_kwargs: Dict[str, Any] = {
            "inv_cdf_linspace_L": 1000,
            "inv_cdf_dicts": ({}, {}, {}),
            # Init integ_vals_dict, cdf_supp_dict, cdf_dict
            "inv_cdf_round_decimals": 2,
        },
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples num_seq_samples sequential samples from the exponential GLM
        parameterized by W. For the ith prior, state is initialized at s[i] and
        action a[i, j] is used for the jth sample.

        Args:
            W: parameter W for exponential GLM. shape (d_psi, d_phi)
            s: prior states. shape (m, d_s) where m is the number of priors.
            a: prior actions. shape (m, num_seq_samples, d_a) where m is the
                number of priors.
            phi_embedding: Instance to calculate phi(s, a). Must output phi of
                dimension d_phi.
            num_seq_samples: Number of sequential samples for each prior.
            sampling_method: (default 'inv_cdf') Must be one of
                `SAMPLING_METHODS`.

        Returns:
            (samples, phis): samples for each prior and the computed phi
                shape (m, num_seq_samples, d_s), (m, num_seq_samples, d_phi)
        """
        assert s.shape[0] == a.shape[0]
        assert W.shape[1] == phi_embedding.d_phi
        assert a.shape[1] == num_seq_samples

        sj = s  # iterates of states, shape (m, d_s)
        samples = []
        phis = []

        for j in range(num_seq_samples):
            # Get one sample from density specified by sj and aj
            x, phi = self._draw_n_samples(
                W,
                sj,
                a[:, j, :],
                phi_embedding,
                1,
                sampling_method,
                sampling_kwargs,
            )  # shape (m, 1, d_s), (m, d_phi)
            x = x.squeeze(dim=1)  # shape (m, d_s)
            x[torch.isnan(x)] = 0.0

            # Record samples and update the state iterate
            samples.append(x)
            phis.append(phi)
            sj = x

        # shape (m, num_seq_samples, d_s or d_phi)
        samples = torch.stack(samples).permute(1, 0, 2)
        phis = torch.stack(phis).permute(1, 0, 2)
        return samples, phis

    def _draw_n_samples(
        self,
        W: T,
        s: T,
        a: T,
        phi_embedding: PhiEmbedding,
        n: int,
        sampling_method: str = "inv_cdf",
        sampling_kwargs: Dict[str, Any] = {
            "inv_cdf_linspace_L": 1000,
            "inv_cdf_dicts": ({}, {}, {}),
            # Init integ_vals_dict, cdf_supp_dict, cdf_dict
            "inv_cdf_round_decimals": 2,
        },
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draws n samples for each of the prior-action pairs. Returns the drawn
        samples for each prior and the computed phi(s, a).

        Args:
            W: parameter W for exponential GLM. shape (d_psi, d_phi)
            s: prior states. shape (m, d_s) where m is the number of priors.
            a: prior actions. shape (m, d_a) where m is the number of priors.
            phi_embedding: Instance to calculate phi(s, a). Must output phi of
                dimension d_phi.
            n: number of samples to draw from GLM specified by (W, phi(s, a)).
            sampling_method: (default 'inv_cdf') Must be one of `SAMPLING_METHODS`.

        Returns:
            samples, phi: shapes (m, n, d_s), (m, d_phi)
        """
        assert (
            sampling_method in self.SAMPLING_METHODS
        ), f"{sampling_method} must be one of {self.SAMPLING_METHODS}"

        if sampling_method == "exact":
            raise ValueError("Exact sampler not implemented.")

        if sampling_method == "inv_cdf":
            return self._draw_n_samples__inv_cdf(
                W, s, a, phi_embedding, n, sampling_kwargs
            )

        return self._draw_n_samples__hmc_variants(
            W, s, a, phi_embedding, n, sampling_method, sampling_kwargs
        )

    def _inv_cdf__get_vji_tilde(
        self,
        v: torch.Tensor,
        j: int,
        i: int,
        inv_cdf_round_decimals: int,
    ) -> str:
        """
        Returns a rounded representation for v[j, i] to uniquely determine the
        density function for calculating its CDF.

        Args:
            v: parameter matrix for density ($W phi(s, a)$). shape (m, d_psi)
                where m is the number of priors.
            j: index into the mth prior.
            i: index into the state.
            inv_cdf_round_decimals: #decimals v[j, i] should be rounded to.
        """
        if self.IS_PRODUCT_DENSITY:
            vji_tilde = str(round(v[j, i].item(), inv_cdf_round_decimals))
        else:
            # v has shape (m, d_psi)
            # d_s must be 1, d_psi > 1. So combine all vals of v_j
            vji_tilde = "_".join(
                str(round(v[j, k].item(), inv_cdf_round_decimals))
                for k in range(v.shape[1])
            )
        return vji_tilde

    def _draw_n_samples__inv_cdf(
        self,
        W: T,
        s: T,
        a: T,
        phi_embedding: PhiEmbedding,
        n: int,
        sampling_kwargs: Dict[str, Any] = {
            "inv_cdf_linspace_L": 1000,
            "inv_cdf_dicts": ({}, {}, {}),
            # Init integ_vals_dict, cdf_supp_dict, cdf_dict
            "inv_cdf_round_decimals": 2,
        },
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements inv_cdf sampling. Same spec as _draw_n_samples().
        """
        assert self._check_uni_or_prod_density(
            W=W
        ), "Inv CDF sampling not possible if density is not 1d or product."

        m = s.shape[0]
        samples = torch.zeros(m, n, self.d_s)  # shape (m, n, d_s)
        s = utils.to_torch(s)
        a = utils.to_torch(a)
        phi = phi_embedding.get_phi(s, a)  # shape (m, d_phi)

        W = utils.to_torch(W)

        if torch.any(phi.isnan()):
            print("phi", phi)
        if torch.any(W.isnan()):
            print("W", W)
        # L is the discretized len of interval [ support_lo ... support_hi ]
        # on which CDFs is computed.
        # integ_vals_dict : vji_tilde |-> (log_part, mean, var)
        # cdf_supp_dict : vji_tilde |-> [ support_lo ... support_hi ]
        # cdf_dict : vji_tilde |-> [ cdf[support_lo] ... cdf[support_hi] ]
        L = sampling_kwargs["inv_cdf_linspace_L"]
        integ_vals_dict, cdf_supp_dict, cdf_dict = sampling_kwargs[
            "inv_cdf_dicts"
        ]
        inv_cdf_round_decimals = sampling_kwargs["inv_cdf_round_decimals"]
        v = phi @ W.T  # shape (m, d_psi); d_psi == d_s if product density

        # Create a 1d instance of density
        density_1d = copy.copy(self)
        density_1d.d_s = 1
        if density_1d.IS_PRODUCT_DENSITY and hasattr(density_1d, "d_psi"):
            # Only change d_psi if product density (otherwise d_s must be 1)
            density_1d.d_psi = 1

        # Compute log_part, mean, var for unique v (upto rounding)
        log_part = torch.empty(m, self.d_s)
        mean = torch.empty(m, self.d_s)
        var = torch.empty(m, self.d_s)
        for i in range(self.d_s):
            vji_tilde2j = collections.OrderedDict()
            vji_tilde2phi_j = collections.OrderedDict()
            for j in range(m):
                # Round v[j, i] to approximate with nearby densities
                vji_tilde = self._inv_cdf__get_vji_tilde(
                    v, j, i, inv_cdf_round_decimals
                )
                if vji_tilde in integ_vals_dict:
                    log_part_j_i, mean_j_i, var_j_i = integ_vals_dict[vji_tilde]
                    log_part[j, i] = log_part_j_i
                    mean[j, i] = mean_j_i
                    var[j, i] = var_j_i
                else:
                    # Need to compute integ vals for vji_tilde
                    if vji_tilde not in vji_tilde2j:
                        vji_tilde2j[vji_tilde] = [j]
                        vji_tilde2phi_j[vji_tilde] = phi[j]
                    else:
                        vji_tilde2j[vji_tilde].append(j)

            if not vji_tilde2j:
                continue
            Wi = W if self.d_s == 1 else W[i : i + 1]
            uniq_phi = torch.stack(
                list(vji_tilde2phi_j.values()), dim=0
            )  # shape (m_uniq, d_phi)
            uniq_log_part = density_1d.logpart(
                Wi, uniq_phi
            )  # shape (m_uniq, 1)
            uniq_mean = density_1d.density_mean(
                Wi, uniq_phi, log_part=uniq_log_part
            )  # shape (m_uniq, 1)
            uniq_var = density_1d.density_var(
                Wi, uniq_phi, mean=uniq_mean, log_part=uniq_log_part
            )  # shape (m_uniq, 1)
            # remap m_uniq items to m using vji_tilde2j
            for k, (vji_tilde, js) in enumerate(vji_tilde2j.items()):
                log_part[js, i] = uniq_log_part[k]
                mean[js, i] = uniq_mean[k]
                var[js, i] = uniq_var[k]
                integ_vals_dict[vji_tilde] = (
                    uniq_log_part[k].item(),
                    uniq_mean[k].item(),
                    uniq_var[k].item(),
                )

        # Calculate CDF for density specified by v[j, i] if doesn't already
        # exist in the dict
        for i in range(self.d_s):
            for j in range(m):
                # Round v[j, i] to approximate with nearby densities
                vji_tilde = self._inv_cdf__get_vji_tilde(
                    v, j, i, inv_cdf_round_decimals
                )
                if vji_tilde in cdf_dict:
                    continue

                # Calculate the CDF support
                supp_range = 3 * np.sqrt(var[j, i].item())
                support_lo = (
                    self.SUPPORT_LO
                    if isinstance(self.SUPPORT_LO, float)
                    else utils.to_torch(self.SUPPORT_LO)[i].item()
                )
                support_lo = max(mean[j, i].item() - supp_range, support_lo)
                support_hi = (
                    self.SUPPORT_HI
                    if isinstance(self.SUPPORT_HI, float)
                    else utils.to_torch(self.SUPPORT_HI)[i].item()
                )
                support_hi = min(mean[j, i].item() + supp_range, support_hi)
                cdf_supp = torch.linspace(support_lo, support_hi, L)

                # Calculate the density
                if self.d_s == 1:
                    # use all W
                    prob = density_1d.density(
                        cdf_supp[:, None],
                        W,
                        phi[j : j + 1, :],
                        log_part=log_part[j : j + 1],
                    ).squeeze(
                        1
                    )  # shape (L)
                else:
                    # product density, d_psi == d_s
                    prob = density_1d.density(
                        cdf_supp[:, None],
                        W[i : i + 1, :],
                        phi[j : j + 1, :],
                        log_part=log_part[j : j + 1, i : i + 1],
                    ).squeeze(
                        1
                    )  # shape (L)
                cdf = torch.cumsum(prob, dim=0) / prob.sum()

                # Record the support and CDF
                cdf_supp_dict[vji_tilde] = cdf_supp
                cdf_dict[vji_tilde] = cdf
        sampling_kwargs["inv_cdf_dicts"] = (
            integ_vals_dict,
            cdf_supp_dict,
            cdf_dict,
        )

        # Do inverse CDF sampling: select rand(0, 1) float and lookup the
        # corresponding x axis through the CDF
        for i in range(self.d_s):
            cdf_supps = torch.stack(
                [
                    cdf_supp_dict[
                        self._inv_cdf__get_vji_tilde(
                            v, j, i, inv_cdf_round_decimals
                        )
                    ]
                    for j in range(m)
                ],
                dim=1,
            )  # shape (L, m)
            cdfs = torch.stack(
                [
                    cdf_dict[
                        self._inv_cdf__get_vji_tilde(
                            v, j, i, inv_cdf_round_decimals
                        )
                    ]
                    for j in range(m)
                ],
                dim=1,
            )  # shape (L, m)

            # Get inverse mapping to x axis from CDFs
            r = torch.rand(m, n)
            cdf_supp_idxs = torch.argmin(
                torch.abs(cdfs[:, :, None] - r[None, :, :]), dim=0
            )  # shape (m, n)

            # Two approaches: looping vs gather-repeat
            ## gather-repeat is expensive or large n
            samples[:, :, i] = torch.concat(
                [
                    torch.gather(
                        cdf_supps.T, dim=1, index=cdf_supp_idxs[:, j : j + 1]
                    )
                    for j in range(n)
                ],
                dim=1,
            )  # shape (m, n)
            # samples[:, :, i] = torch.gather(
            #     cdf_supps[:, :, None].repeat(1, 1, n),
            #     0,
            #     cdf_supp_idxs[None, :, :].repeat(L, 1, 1)
            # )[0]

        return samples, phi

    def _draw_n_samples__hmc_variants(
        self,
        W: T,
        s: T,
        a: T,
        phi_embedding: PhiEmbedding,
        n: int,
        sampling_method: str,
        sampling_kwargs: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements hmc (and variants) sampling. Same spec as _draw_n_samples().
        """
        m, d_s = s.shape
        samples = torch.zeros(m, n, d_s)  # shape (m, d_s)
        s = utils.to_torch(s)
        a = utils.to_torch(a)
        phi = phi_embedding.get_phi(s, a)  # shape (m, d_phi)

        # Define log_prob_func and initialize density's phi
        def log_prob_func(phi, x):
            """
            Args:
                phi: phi of prior. shape (1, d_phi)
                x: point to calculate log density at. shape (d_s)
            """
            return self.log_density(x.unsqueeze(dim=0), W, phi).squeeze()

        # Init params_init from the support: get a finite range
        support_lo = max(utils.to_torch(self.SUPPORT_LO)).item()
        support_lo = -10.0 if np.isneginf(support_lo) else support_lo
        support_hi = min(utils.to_torch(self.SUPPORT_HI)).item()
        support_hi = 10.0 if np.isposinf(support_hi) else support_hi

        hmc_params_init = torch.rand(d_s, requires_grad=True)  # shape (d_s)
        hmc_params_init = (
            hmc_params_init * (support_hi - support_lo) + support_lo
        )

        # Based on sampling_method, set sampling_kwargs
        if "sampling_seed" in sampling_kwargs:
            hamiltorch.set_random_seed(sampling_kwargs.pop("sampling_seed"))

        sampling_kwargs["num_samples"] = 10 + n  # 10 for warmup
        sampling_kwargs["num_steps_per_sample"] = 10
        if sampling_method == "hmc":
            sampling_kwargs["sampler"] = hamiltorch.Sampler.HMC

        elif sampling_method == "hmc_nuts":
            sampling_kwargs["sampler"] = hamiltorch.Sampler.HMC_NUTS
            sampling_kwargs["num_samples"] = sampling_kwargs["burn"] + 1

        elif sampling_method == "implicit_rmhmc":
            sampling_kwargs["sampler"] = hamiltorch.Sampler.RMHMC
            sampling_kwargs["integrator"] = hamiltorch.Integrator.IMPLICIT
            sampling_kwargs["fixed_point_max_iterations"] = 1000
            sampling_kwargs["fixed_point_threshold"] = 1e-5

        elif sampling_method == "explicit_rmhmc":
            sampling_kwargs["sampler"] = hamiltorch.Sampler.RMHMC
            sampling_kwargs["integrator"] = hamiltorch.Integrator.EXPLICIT
            sampling_kwargs["explicit_binding_const"] = 100.0

        # TODO m loop can be parallelized
        for i in range(m):
            # hamiltorch uses hmc_params_init as the first returned sample, so
            # sample more than n points, and use the last n ones
            with utils.HiddenPrints():
                x = hamiltorch.sample(
                    log_prob_func=partial(log_prob_func, phi[i : i + 1]),
                    params_init=s[i].clone(),
                    **sampling_kwargs,
                )  # returns a list of samples, use the last one
                samples[i] = torch.stack(x[-n:])

        return samples, phi

    def W_MLE_estimate(
        self, x: T, phi: T, W0: Optional[T] = None
    ) -> torch.Tensor:
        """
        Returns the MLE estimate of W using points x sampled with prior
        embeddings phi.

        Args:
            x: points sampled from the distribution. shape (m, n, d_s) where n
                is the number of points sampled for m prior points.
            phi: value of phi(s, a). shape (m, d_phi) where m is the number
                of priors (s, a).
            W0: Initial estimate for W to numerically compute MLE. shape (
                d_psi, d_phi).

        Returns:
            shape (d_psi, d_phi)
        """
        m, n, d_s = x.shape
        if d_s != 1:
            raise ValueError("MLE is intractable")
        else:
            if W0 is None:
                raise ValueError("Set W0 for initializating numerical MLE")

            # d_s == 1, numerically calculate MLE by minimizing NLL
            # assume m == 1

            def NLL(v: np.ndarray, x_: np.ndarray) -> float:
                """
                Args:
                    v: parameter v = W * phi(s, a). shape (d_psi)
                    x_: data shape (n, d_s)

                Returns:
                    Negative log likelihood of data given parameter v.
                """
                v = utils.to_torch(v)
                x_ = utils.to_torch(x_)

                psi = self.psi(x_)  # shape (n, d_psi)
                log_part_v = self.logpart_v(v.unsqueeze(0))  # shape (1)
                nll = n * log_part_v - torch.sum(psi @ v, dim=0)
                return nll.item()

            # Estimate v_MLE = W_MLE @ phi.T by minimizing the NLL of data
            x = utils.to_numpy(x)

            v0 = phi @ W0.T
            v_MLE = np.stack(
                [
                    minimize(
                        NLL,
                        v0[i],
                        args=(x[i]),
                        method="bfgs",
                        jac="3-point",
                        options={"eps": 1e-5},
                    ).x
                    for i in range(m)
                ]
            )  # shape (m, d_psi)
            v_MLE = utils.to_torch(v_MLE)

            # Solve for W_MLE (v_MLE, phi known W_MLE matrix unkwown)
            W_MLE = torch.linalg.lstsq(
                phi, v_MLE
            ).solution.T  # shape (d_psi, d_phi)
            return W_MLE


class NonLDSDensity(ExpGLMDensity):
    """
    x = W phi(s, a) + eps, where eps is N(0, diag(sigma))

    NonLDS density with a diagonal error covariance matrix. Here, d_psi == d_s
    """

    # NonLDS with gaussian noise defined for x in R^d
    SUPPORT_LO = -np.inf
    SUPPORT_HI = np.inf
    IS_PRODUCT_DENSITY = True

    def __init__(self, d_s: int, sigma: Optional[T] = None):
        super(NonLDSDensity, self).__init__(d_s)

        if sigma is None:
            sigma = torch.ones(self.d_s)
        assert (
            len(sigma.shape) == 1
        ), "Non-diagonal error covariance matrix not supported"
        assert sigma.shape[0] == self.d_s

        self.sigma = utils.to_torch(sigma)
        self.sigma_inv = 1.0 / (self.sigma + const.DIV_EPS)
        self.d_psi = self.d_s

    def __repr__(self):
        return f"$N(W \\phi(s, a), \\Sigma=diag({self.sigma}))$"

    def in_support(self, x: T) -> torch.Tensor:
        return torch.ones((x.shape[0]), dtype=bool)

    def q(self, x: T) -> torch.Tensor:
        x = utils.to_torch(x)
        sigma_det = torch.prod(self.sigma)

        numerator = torch.exp(-torch.sum(x * (x * self.sigma_inv), dim=-1) / 2)
        denominator = np.power(2 * np.pi, self.d_s / 2) * torch.sqrt(sigma_det)
        return numerator / (denominator + const.DIV_EPS)

    def logq(self, x: T) -> torch.Tensor:
        # Since q has an exp, simplify log q rather than doing log (exp (...))
        # Reduces floating point errors
        x = utils.to_torch(x)
        sigma_det = torch.prod(self.sigma)

        logc = -(np.log(2 * np.pi) * self.d_s + torch.log(sigma_det)) / 2
        logq = logc - torch.sum(x * (x * self.sigma_inv), dim=-1) / 2
        return logq

    def psi(self, x: T) -> torch.Tensor:
        # Since sigma_inv is diagonal, scale each point with sigma_inv using
        # broadcasting
        x = utils.to_torch(x)
        return x * self.sigma_inv

    def logpart_v(self, v: T) -> torch.Tensor:
        v = utils.to_torch(v)  # shape (m, d_s) as d_psi == d_s
        return v * (v * self.sigma_inv) / 2  # shape (m, d_s)

    def density_mean(
        self, W: T, phi: T, log_part: Optional[T] = None
    ) -> torch.Tensor:
        W = utils.to_torch(W)
        phi = utils.to_torch(phi)
        v = phi @ W.T  # shape (m, d_s)
        return v

    def density_var(
        self,
        W: T,
        phi: T,
        mean: Optional[T] = None,
        log_part: Optional[T] = None,
    ) -> torch.Tensor:
        m = phi.shape[0]
        return self.sigma.unsqueeze(0).repeat((m, 1))  # shape (m, d_s)

    def _draw_n_samples(
        self,
        W: T,
        s: T,
        a: T,
        phi_embedding: PhiEmbedding,
        n: int,
        sampling_method: str = "inv_cdf",
        sampling_kwargs: Dict[str, Any] = {
            "inv_cdf_linspace_L": 1000,
            "inv_cdf_dicts": ({}, {}, {}),
            # Init integ_vals_dict, cdf_supp_dict, cdf_dict
            "inv_cdf_round_decimals": 2,
        },
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if sampling_method == "exact":
            phi = phi_embedding.get_phi(s, a)  # shape (m, d_phi)
            mu = phi @ W.T  # shape (m, d_s)
            dist = torch.distributions.Normal(mu, torch.sqrt(self.sigma))
            samples = dist.sample([n])  # shape (n, m, d_s)
            return (
                samples.permute(1, 0, 2),
                phi,
            )  # shape (m, n, d_s), (m, d_phi)

        else:
            return super()._draw_n_samples(
                W,
                s,
                a,
                phi_embedding,
                n,
                sampling_method=sampling_method,
                sampling_kwargs=sampling_kwargs,
            )

    def _inv_cdf__get_vji_tilde(
        self,
        v: torch.Tensor,
        j: int,
        i: int,
        inv_cdf_round_decimals: int,
    ) -> str:
        # For normal density, different sigma result in different
        # densities. So to differentiate densities with same v[j, i] but
        # different sigma, add sigma info to vji_tilde str
        sigma_str = f"sigma{self.sigma[i].item():.2e}"
        vji_tilde = str(round(v[j, i].item(), inv_cdf_round_decimals))
        vji_tilde = f"{sigma_str}_{vji_tilde}"
        return vji_tilde

    def _draw_n_samples__inv_cdf(
        self,
        W: T,
        s: T,
        a: T,
        phi_embedding: PhiEmbedding,
        n: int,
        sampling_kwargs: Dict[str, Any] = {
            "inv_cdf_linspace_L": 1000,
            "inv_cdf_dicts": ({}, {}, {}),
            # Init integ_vals_dict, cdf_supp_dict, cdf_dict
            "inv_cdf_round_decimals": 2,
        },
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._check_uni_or_prod_density(
            W=W
        ), "Inv CDF sampling not possible if density is not 1d or product."

        m = s.shape[0]
        samples = torch.zeros(m, n, self.d_s)  # shape (m, n, d_s)
        s = utils.to_torch(s)
        a = utils.to_torch(a)
        phi = phi_embedding.get_phi(s, a)  # shape (m, d_phi)

        # L is the discretized len of interval [ support_lo ... support_hi ]
        # on which CDFs is computed.
        # integ_vals_dict : vji_tilde |-> (log_part, mean, var)
        # cdf_supp_dict : vji_tilde |-> [ support_lo ... support_hi ]
        # cdf_dict : vji_tilde |-> [ cdf[support_lo] ... cdf[support_hi] ]
        L = sampling_kwargs["inv_cdf_linspace_L"]
        integ_vals_dict, cdf_supp_dict, cdf_dict = sampling_kwargs[
            "inv_cdf_dicts"
        ]
        inv_cdf_round_decimals = sampling_kwargs["inv_cdf_round_decimals"]
        v = phi @ W.T  # shape (m, d_psi); d_psi == d_s if product density

        # Create a 1d instance of density
        density_1d = copy.copy(self)
        density_1d.d_s = 1
        density_1d.d_psi = 1

        # Compute log_part, mean, var for unique v (upto rounding)
        log_part = torch.empty(m, self.d_s)
        mean = torch.empty(m, self.d_s)
        var = torch.empty(m, self.d_s)
        for i in range(self.d_s):
            # Set sigma of the 1d density
            density_1d.sigma = self.sigma[i]
            density_1d.sigma_inv = self.sigma_inv[i]

            vji_tilde2j = collections.OrderedDict()
            vji_tilde2phi_j = collections.OrderedDict()
            for j in range(m):
                # Round v[j, i] to approximate with nearby densities
                vji_tilde = self._inv_cdf__get_vji_tilde(
                    v, j, i, inv_cdf_round_decimals
                )
                if vji_tilde in integ_vals_dict:
                    log_part_j_i, mean_j_i, var_j_i = integ_vals_dict[vji_tilde]
                    log_part[j, i] = log_part_j_i
                    mean[j, i] = mean_j_i
                    var[j, i] = var_j_i
                else:
                    # Need to compute integ vals for vji_tilde
                    if vji_tilde not in vji_tilde2j:
                        vji_tilde2j[vji_tilde] = [j]
                        vji_tilde2phi_j[vji_tilde] = phi[j]
                    else:
                        vji_tilde2j[vji_tilde].append(j)

            if not vji_tilde2j:
                # Empty set of vji_tilde to compute
                continue
            Wi = W if self.d_s == 1 else W[i : i + 1]
            uniq_phi = torch.stack(
                list(vji_tilde2phi_j.values()), dim=0
            )  # shape (m_uniq, d_phi)
            uniq_log_part = density_1d.logpart(
                Wi, uniq_phi
            )  # shape (m_uniq, 1)
            uniq_mean = density_1d.density_mean(
                Wi, uniq_phi, log_part=uniq_log_part
            )  # shape (m_uniq, 1)
            uniq_var = density_1d.density_var(
                Wi, uniq_phi, mean=uniq_mean, log_part=uniq_log_part
            )  # shape (m_uniq, 1)
            # remap m_uniq items to m using vji_tilde2j
            for k, (vji_tilde, js) in enumerate(vji_tilde2j.items()):
                log_part[js, i] = uniq_log_part[k]
                mean[js, i] = uniq_mean[k]
                var[js, i] = uniq_var[k]
                integ_vals_dict[vji_tilde] = (
                    uniq_log_part[k].item(),
                    uniq_mean[k].item(),
                    uniq_var[k].item(),
                )

        # Calculate CDF for density specified by v[j, i] if doesn't already
        # exist in the dict
        for i in range(self.d_s):
            # Set sigma of the 1d density
            density_1d.sigma = self.sigma[i]
            density_1d.sigma_inv = self.sigma_inv[i]

            for j in range(m):
                # Round v[j, i] to approximate with nearby densities
                vji_tilde = self._inv_cdf__get_vji_tilde(
                    v, j, i, inv_cdf_round_decimals
                )
                if vji_tilde in cdf_dict:
                    continue

                # Calculate the CDF support
                supp_range = 3 * np.sqrt(var[j, i].item())
                support_lo = (
                    self.SUPPORT_LO
                    if isinstance(self.SUPPORT_LO, float)
                    else utils.to_torch(self.SUPPORT_LO)[i].item()
                )
                support_lo = max(mean[j, i].item() - supp_range, support_lo)
                support_hi = (
                    self.SUPPORT_HI
                    if isinstance(self.SUPPORT_HI, float)
                    else utils.to_torch(self.SUPPORT_HI)[i].item()
                )
                support_hi = min(mean[j, i].item() + supp_range, support_hi)
                cdf_supp = torch.linspace(support_lo, support_hi, L)

                # Calculate the density
                prob = density_1d.density(
                    cdf_supp[:, None],
                    W[i : i + 1, :],
                    phi[j : j + 1, :],
                    log_part=log_part[j : j + 1, i : i + 1],
                ).squeeze(
                    1
                )  # shape (L)
                cdf = torch.cumsum(prob, dim=0) / prob.sum()

                # Record the support and CDF
                cdf_supp_dict[vji_tilde] = cdf_supp
                cdf_dict[vji_tilde] = cdf
        sampling_kwargs["inv_cdf_dicts"] = (
            integ_vals_dict,
            cdf_supp_dict,
            cdf_dict,
        )

        # Do inverse CDF sampling: select rand(0, 1) float and lookup the
        # corresponding x axis through the CDF
        for i in range(self.d_s):
            cdf_supps = torch.stack(
                [
                    cdf_supp_dict[
                        self._inv_cdf__get_vji_tilde(
                            v, j, i, inv_cdf_round_decimals
                        )
                    ]
                    for j in range(m)
                ],
                dim=1,
            )  # shape (L, m)
            cdfs = torch.stack(
                [
                    cdf_dict[
                        self._inv_cdf__get_vji_tilde(
                            v, j, i, inv_cdf_round_decimals
                        )
                    ]
                    for j in range(m)
                ],
                dim=1,
            )  # shape (L, m)

            # Get inverse mapping to x axis from CDFs
            r = torch.rand(m, n)
            cdf_supp_idxs = torch.argmin(
                torch.abs(cdfs[:, :, None] - r[None, :, :]), dim=0
            )  # shape (m, n)

            # Two approaches: looping vs gather-repeat
            ## gather-repeat is expensive or large n
            samples[:, :, i] = torch.concat(
                [
                    torch.gather(
                        cdf_supps.T, dim=1, index=cdf_supp_idxs[:, j : j + 1]
                    )
                    for j in range(n)
                ],
                dim=1,
            )  # shape (m, n)
            # samples[:, :, i] = torch.gather(
            #     cdf_supps[:, :, None].repeat(1, 1, n),
            #     0,
            #     cdf_supp_idxs[None, :, :].repeat(L, 1, 1)
            # )[0]

        return samples, phi

    def W_MLE_estimate(self, x: T, phi: T, W0=None) -> torch.Tensor:
        assert x.shape[0] == phi.shape[0]

        x = utils.to_torch(x)
        phi = utils.to_torch(phi)
        m, n, d_s = x.shape

        # Estimate v_MLE = W_MLE @ phi.T directly
        v_MLE = torch.sum(x, dim=1) / n  # shape (m, d_s)

        # Solve for W_MLE (v_MLE, phi known W_MLE matrix unknown)
        W_MLE = torch.linalg.lstsq(phi, v_MLE).solution.T  # shape (d_s, d_phi)
        return W_MLE

    def Sigma_MLE_estimate(self, x: T, phi: T) -> torch.Tensor:
        """
        Returns the MLE estimate of Sigma using points x sampled with prior
        embeddings phi.

        Args:
            x: points sampled from the distribution. shape (m, n, d_s) where n
                is the number of points sampled for m prior points.
            phi: value of phi(s, a). shape (m, d_phi) where m is the number
                of priors (s, a).

        Returns:
            shape (d_s)
        """
        assert x.shape[0] == phi.shape[0]

        x = utils.to_torch(x)
        phi = utils.to_torch(phi)
        m, n, d_s = x.shape

        # Estimate v_MLE = W_MLE @ phi.T directly
        v_MLE = torch.sum(x, dim=1, keepdim=True) / n  # shape (m, 1, d_s)

        # Sample covariance
        sigma_MLE = torch.sum((x - v_MLE) ** 2, dim=1) / n  # shape (m, d_s)

        sigma_MLE = torch.sum(sigma_MLE, dim=0) / m  # shape (d_s)
        return sigma_MLE


class ExpDensity(ExpGLMDensity):
    """
    Exponential density for d_s>1 defined as exponential in each dimension
    independently.
    """

    SUPPORT_LO = 1e-8
    SUPPORT_HI = np.inf
    IS_PRODUCT_DENSITY = True

    def __init__(self, d_s: int):
        super(ExpDensity, self).__init__(d_s)
        self.d_psi = d_s

    def __repr__(self):
        return f"$Exp(-W \\phi(s, a))$"

    def in_support(self, x: T) -> torch.Tensor:
        return torch.all(x > self.SUPPORT_LO, dim=1)

    def logq(self, x: T) -> torch.Tensor:
        x = utils.to_torch(x)
        # Do trivial op on x and add 0 instead of creating a zeros vector
        # allowing functorch jacobian to interpret q : R^d_s -> R
        return torch.sum(0.0 * x, dim=-1) + 0.0  # shape (n)

    def psi(self, x: T) -> torch.Tensor:
        return utils.to_torch(x)

    def logpart_v(self, v: T) -> torch.Tensor:
        v = utils.to_torch(v)  # shape (m, d_psi) where d_psi == d_s == 1
        return -torch.log(-v)

    def density_mean(
        self, W: T, phi: T, log_part: Optional[T] = None
    ) -> torch.Tensor:
        # mean = 1/lmb where lmb = -v
        W = utils.to_torch(W)
        phi = utils.to_torch(phi)
        v = phi @ W.T  # shape (m, d_psi) where d_psi == d_s
        return 1.0 / (-v + const.DIV_EPS)

    def density_var(
        self,
        W: T,
        phi: T,
        mean: Optional[T] = None,
        log_part: Optional[T] = None,
    ) -> torch.Tensor:
        # var = 1/lmb^2 where lmb = -v
        W = utils.to_torch(W)
        phi = utils.to_torch(phi)
        v = phi @ W.T  # shape (m, d_psi) where d_psi == d_s
        return 1.0 / (v**2 + const.DIV_EPS)

    def _draw_n_samples(
        self,
        W: T,
        s: T,
        a: T,
        phi_embedding: PhiEmbedding,
        n: int,
        sampling_method: str = "inv_cdf",
        sampling_kwargs: Dict[str, Any] = {
            "inv_cdf_linspace_L": 1000,
            "inv_cdf_dicts": ({}, {}, {}),
            # Init integ_vals_dict, cdf_supp_dict, cdf_dict
            "inv_cdf_round_decimals": 2,
        },
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if sampling_method == "exact":
            phi = phi_embedding.get_phi(s, a)  # shape (m, d_phi)
            minus_lmb = phi @ W.T  # shape (m, d_s)
            dist = torch.distributions.Exponential(-minus_lmb)
            samples = dist.sample([n])  # shape (n, m, d_s)
            return (
                samples.permute(1, 0, 2),
                phi,
            )  # shape (m, n, d_s), (m, d_phi)

        else:
            return super()._draw_n_samples(
                W,
                s,
                a,
                phi_embedding,
                n,
                sampling_method=sampling_method,
                sampling_kwargs=sampling_kwargs,
            )

    def W_MLE_estimate(self, x: T, phi: T, W0=None) -> torch.Tensor:
        assert x.shape[0] == phi.shape[0]

        x = utils.to_torch(x)
        phi = utils.to_torch(phi)
        m, n, d_s = x.shape

        # Estimate v_MLE = W_MLE @ phi.T directly
        v_MLE = -n / torch.sum(x, dim=1)  # shape (m, d_s)

        # Solve for W_MLE (v_MLE, phi known W_MLE matrix unknown)
        W_MLE = torch.linalg.lstsq(phi, v_MLE).solution.T  # shape (d_s, d_phi)
        return W_MLE


class BetaDensity(ExpGLMDensity):
    SUPPORT_LO = 0.0
    SUPPORT_HI = 1.0

    def __init__(self, d_s: int):
        assert d_s == 1, "Beta density only defined for univariate"
        super(BetaDensity, self).__init__(d_s)
        self.d_psi = 2

    def __repr__(self):
        return (
            f"Beta($\\alpha=(W \\phi(s, a))[0]+1, \\beta=(W \\phi(s, a))[1]+1$)"
        )

    def in_support(self, x: T) -> torch.Tensor:
        return torch.all(
            torch.logical_and(x >= self.SUPPORT_LO, x <= self.SUPPORT_HI), dim=1
        )

    def logq(self, x: T) -> torch.Tensor:
        x = utils.to_torch(x)
        # Do trivial op on x and add 0 instead of creating a zeros vector
        # allowing functorch jacobian to interpret q : R^d_s -> R
        return torch.sum(0.0 * x, dim=-1) + 0.0  # shape (n)

    def psi(self, x: T) -> torch.Tensor:
        x = utils.to_torch(x)
        # Ensure log with not give nans
        x = torch.clamp(x, const.DIV_EPS, 1.0 - const.DIV_EPS)
        psi = torch.hstack([torch.log(x), torch.log(1 - x)])  # shape (n, 2)
        return psi

    def logpart_v(self, v: T) -> torch.Tensor:
        v = utils.to_torch(v)  # shape (m, d_psi)
        log_part_v = (
            torch.lgamma(v[:, 0] + 1)
            + torch.lgamma(v[:, 1] + 1)
            - torch.lgamma(v[:, 0] + v[:, 1] + 2)
        )  # shape (m)
        return log_part_v.unsqueeze(1)  # shape (m, d_s) where d_s == 1

    def density_mean(
        self, W: T, phi: T, log_part: Optional[T] = None
    ) -> torch.Tensor:
        W = utils.to_torch(W)
        phi = utils.to_torch(phi)
        v = phi @ W.T  # shape (m, d_psi)
        alpha, beta = v[:, 0], v[:, 1]
        mean = alpha / (alpha + beta + const.DIV_EPS)
        return mean.unsqueeze(1)  # shape (m, d_s)

    def density_var(
        self,
        W: T,
        phi: T,
        mean: Optional[T] = None,
        log_part: Optional[T] = None,
    ) -> torch.Tensor:
        W = utils.to_torch(W)
        phi = utils.to_torch(phi)
        v = phi @ W.T  # shape (m, d_psi)
        alpha, beta = v[:, 0], v[:, 1]
        var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
        return var.unsqueeze(1)  # shape (m, d_s)

    def _draw_n_samples(
        self,
        W: T,
        s: T,
        a: T,
        phi_embedding: PhiEmbedding,
        n: int,
        sampling_method: str = "inv_cdf",
        sampling_kwargs: Dict[str, Any] = {
            "inv_cdf_linspace_L": 1000,
            "inv_cdf_dicts": ({}, {}, {}),
            # Init integ_vals_dict, cdf_supp_dict, cdf_dict
            "inv_cdf_round_decimals": 2,
        },
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if sampling_method == "exact":
            phi = phi_embedding.get_phi(s, a)  # shape (m, d_phi)
            concs = (phi @ W.T) + 1  # shape (m, 2)
            dist = torch.distributions.Beta(concs[:, 0], concs[:, 1])
            samples = dist.sample([n]).unsqueeze(dim=2)  # shape (n, m, 1)
            return samples.permute(1, 0, 2), phi  # shape (m, n, 1), (m, d_phi)

        else:
            return super()._draw_n_samples(
                W,
                s,
                a,
                phi_embedding,
                n,
                sampling_method=sampling_method,
                sampling_kwargs=sampling_kwargs,
            )


class CustomSinDensity(ExpGLMDensity):
    SUPPORT_LO = -10.0
    SUPPORT_HI = 10.0
    IS_PRODUCT_DENSITY = True

    def __init__(self, d_s: int, P: float, ALPHA: float = 2.0):
        super(CustomSinDensity, self).__init__(d_s)
        self.P = P
        self.ALPHA = ALPHA
        self.d_psi = self.d_s

    def __repr__(self) -> str:
        ALPHA = self.ALPHA
        P = self.P
        return f"$\\exp ( - \\frac{{ ||s'||_\\alpha^\\alpha }}{{ \\alpha }} ) \\cdot \\exp ( \\langle sin({P}s'), W \\phi(s, a) \\rangle - Z_{{s,a}}(W))$\n$\\alpha={ALPHA:.1f},P={P}$"

    def in_support(self, x: T) -> torch.Tensor:
        # HACK: Avoid assert checks as they slow down simulation with random
        #  shooting. Just give all ones, (this density is anyway defined on R)
        return torch.ones((x.shape[0]), dtype=bool)

    def psi(self, x: T) -> torch.Tensor:
        x = utils.to_torch(x)  # shape (n, d_s)
        P = self.P
        psi = torch.sin(P * x)
        return psi

    def plot_psi(self):
        x = np.linspace(-5, 5).reshape(50, 1)
        plt.plot(x, self.psi(x))
        plt.show()

    def logq(self, x: T) -> torch.Tensor:
        x = utils.to_torch(x)  # shape (n, d_s)
        ALPHA = self.ALPHA
        logq = -x.norm(p=ALPHA, dim=-1).pow(ALPHA) / ALPHA  # shape (n)
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
        sigma_str = f"ALPHA{self.ALPHA:.1f}_P{self.P:.2e}"
        vji_tilde = "_".join(
            str(round(v[j, k].item(), inv_cdf_round_decimals))
            for k in range(v.shape[1])
        )
        vji_tilde = f"{sigma_str}_{vji_tilde}"
        return vji_tilde

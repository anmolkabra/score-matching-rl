import math
import os
import sys
from typing import Callable, Tuple, Union

import numpy as np
import scipy.integrate
import torch
from functorch import vmap, jacrev


def to_torch(x: Union[float, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Converts data x to torch tensor if not already.
    """
    if isinstance(x, float):
        x = torch.tensor([x])
    elif isinstance(x, np.ndarray):
        x = torch.tensor(x.astype(np.float32))
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"Unable to convert x to torch Tensor, {type(x)} not supported."
        )
    return x


def to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Converts data x to numpy array if not already.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    if not isinstance(x, np.ndarray):
        raise TypeError(
            f"Unable to convert x to numpy array, {type(x)} not supported."
        )
    return x


jac_in_T = Union[torch.Tensor, Tuple[torch.Tensor, ...]]
jac_out_T = Union[
    torch.Tensor, Tuple[torch.Tensor, ...], Tuple[Tuple[torch.Tensor, ...]]
]


def batch_jacobian(
    func: Callable[[jac_in_T], jac_out_T], inputs: jac_in_T
) -> jac_out_T:
    """
    Calculates the jacobian of `func`'s output w.r.t. each input in `inputs`.
    Each input is a batch of points (shape (n, d_in) where n is the batch size).
    Depending on the number of inputs and the number of outputs from `func`,
    either of the 3 returned:
    1. jacobian
    2. tuple of jacobians, jacobians[i] is the jacobian of output w.r.t.
    inputs[i]
    3. tuple of tuple of jacobians, jacobians[i][j] is the jacobian of
    outputs[i] w.r.t. inputs[j].

    Args:
        func: Function to compute the jacobian of. Maps each input point of
            dimension d_in to dimension d_out. `func` can also return a tuple of
            outputs with different output dimensions.
        inputs: A single input tensor or tuple of inputs. shape (n, d_in) where
            n is the batch size.

    Returns:
        shape (n, d_out, d_in) for each jacobian matrix (multiple jacobians can
        be returned), where jacobian[i] is the jacobian matrix of function
        output w.r.t. ith input point.
    """
    return vmap(jacrev(func))(inputs)


def batch_full_hessian(
    func: Callable[[torch.Tensor], torch.Tensor], inputs: torch.Tensor
) -> jac_out_T:
    """
    Calculates the full hessian of `func`'s output w.r.t. `inputs`. Input tensor
    is a batch of points (shape (n, d_in) where n is the batch size).

    Args:
        func: Function to compute the hessian of. Maps each input point of
            dimension d_in to dimension d_out.
        inputs: input tensor to evaluate on. shape (n, d_in) where n is the
            batch size.

    Returns:
        shape (n, d_out, d_in, d_in) where hessian[i] is the hessian matrix of
        function output w.r.t. ith input point.
    """
    return vmap(jacrev(jacrev(func)))(inputs)


def batch_diag_hessian(
    func: Callable[[torch.Tensor], torch.Tensor], inputs: torch.Tensor
) -> jac_out_T:
    """
    Calculates the diagonal of hessian of `func`'s output w.r.t. `inputs`. Input
    tensor is a batch of points (shape (n, d_in) where n is the batch size).

    Args:
        func: Function to compute the hessian of. Maps each input point of
            dimension d_in to dimension d_out.
        inputs: input tensor to evaluate on. shape (n, d_in) where n is the
            batch size.

    Returns:
        shape (n, d_out, d_in) where hessian[i] is the hessian diagonal of
        function output w.r.t. ith input point. That is, hessian[i] =
        $\partial^2 f / \partial x_i^2$.
    """
    # TODO Faster implementation?
    full_hessian = vmap(jacrev(jacrev(func)))(inputs)
    diag_hessian = torch.diagonal(full_hessian, dim1=2, dim2=3)
    return diag_hessian


def tvd(x: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculates the total variational distance between two 1d densities,
    evaluated at points x. Their PDF values are given in p1 and p2.

    Args:
        x: Points where densities are evaluated at.
        p1: PDF of density P1 evaluated at all points in x.
        p2: PDF of density P2 evaluated at all points in x.
    """
    assert x.shape == p1.shape
    assert x.shape == p2.shape

    sort_idxs = np.argsort(x)
    x = x[sort_idxs]
    p1 = p1[sort_idxs]
    p2 = p2[sort_idxs]

    # Calculate the riemann sum using the mid points for area estimate
    dx = x[1:] - x[:-1]
    p1_mid = (p1[1:] + p1[:-1]) / 2
    p2_mid = (p2[1:] + p2[:-1]) / 2
    return np.sum(np.abs(p1_mid - p2_mid) * dx) / 2


def density_diff(
    d1: Callable[[torch.Tensor], torch.Tensor],
    d2: Callable[[torch.Tensor], torch.Tensor],
    lo: float,
    hi: float,
    epsabs: float = 1e-4,
    epsrel: float = 1e-4,
) -> float:
    """
    Calculates \int_{lo}^{hi} |d1(x) - d2(x)| dx for x \in R and di : R -> R.

    Args:
        d1: PDF of first density of a single point x
        d2: PDF of second density of a single point x
        lo: Low value for definite integral
        hi: High value for definite integral
        epsabs: Absolute eps tolerance for scipy integration routine
        epsrel: Relative eps tolerance for scipy integration routine
    """

    def integrand(x: float) -> float:
        x = torch.tensor(x).reshape((1, 1))
        p = torch.abs(d1(x) - d2(x))
        return torch.squeeze(p).detach().numpy()

    return scipy.integrate.quad(
        integrand, lo, hi, epsabs=epsabs, epsrel=epsrel
    )[0]


class HiddenPrints:
    """
    Suppresses outputs to stdout by redirecting to the OS's dev null.

    Usage:
        ```
        with HiddenPrints:
            ...
        ```
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def round_sigfigs(a: float, sigfigs: int) -> float:
    """
    Rounds a to some significant figures, returns 0.0 if a smaller than 1e-8
    in magnitude.
    """
    if abs(a) < 1e-8:
        return 0.0
    else:
        return round(a, sigfigs - int(math.floor(math.log10(abs(a)))) - 1)

import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.axes as axes
import matplotlib.pyplot as plt
import numpy as np
import torch

import utils
from data_gen.density import ExpGLMDensity, NonLDSDensity
from data_gen.phi_embedding import PhiEmbedding
from estimation.score_matching import ScoreMatching

SEED = 0


@dataclass
class Density1D:
    name: str
    x: np.ndarray
    prob: np.ndarray
    plot_name: str


def get_1d_densities(
    exp_glm_density: ExpGLMDensity,
    W: torch.Tensor,
    phi: torch.Tensor,
    n_points_eval: int,
    samples: torch.Tensor,
    sm_estimator: ScoreMatching,
    SM_lam: float = 0.01,
    MLE_use_normal_dist: bool = False,
    MLE_all_priors_same: bool = False,
) -> List[Density1D]:
    """
    Args:
        W: shape (d_psi, d_phi)
        phi: shape (m, d_phi)
        samples: shape (m, n, d_s)
    """
    assert W.shape[1] == phi.shape[1]
    assert phi.shape[0] == samples.shape[0]
    assert samples.shape[2] == exp_glm_density.d_s

    m, n, d_s = samples.shape
    d_phi = phi.shape[1]
    N = n_points_eval
    densities = []

    # Points to evaluate density at
    support_lo = (
        -10
        if np.isinf(exp_glm_density.SUPPORT_LO)
        and (exp_glm_density.SUPPORT_LO < 0)
        else exp_glm_density.SUPPORT_LO
    )
    support_hi = (
        10
        if np.isinf(exp_glm_density.SUPPORT_HI)
        and (exp_glm_density.SUPPORT_HI > 0)
        else exp_glm_density.SUPPORT_HI
    )
    x = np.linspace(support_lo, support_hi, N).reshape(
        (-1, 1)
    )  # shape (N, d_s)

    # True density
    # NOTE: Only need to calculate density for one prior
    print("Calculating true density...")
    try:
        prob = exp_glm_density.density(x, W, phi[:1])  # shape (N, 1)
        true_density = Density1D(
            name="true",
            x=x[:, 0],
            prob=utils.to_numpy(prob[:, 0]),
            plot_name="True density",
        )
        densities.append(true_density)
    except ValueError as e:
        warnings.warn(traceback.format_exc())

    print("Calculating density using W_SM...")
    # Get W estimate using Score Matching
    try:
        # Collapse the 3d tensor samples
        samples_exp = samples.detach().reshape((-1, d_s))  # shape (m*n, d_s)

        # Repeat each phi n times in the 2d tensor
        phi_exp = (
            phi.unsqueeze(1).expand(-1, n, -1).reshape((-1, d_phi))
        )  # shape (m*n, d_phi)

        W_SM = sm_estimator.W_estimate(
            samples_exp, phi_exp, lam=SM_lam, return_Vb_estimates=False
        )
        prob = exp_glm_density.density(x, W_SM, phi[:1])  # shape (N, 1)
        sm_density = Density1D(
            name="SM",
            x=x[:, 0],
            prob=utils.to_numpy(prob[:, 0]),
            plot_name="Density w/ $\hat{W}_{SM}$",
        )
        densities.append(sm_density)

        tvd_SM = utils.tvd(true_density.x, true_density.prob, sm_density.prob)
        print(f"tvd_SM: {tvd_SM:0.5e}")
    except ValueError as e:
        warnings.warn(
            "Score Matching assumptions are not met.\n" + traceback.format_exc()
        )

    print("Calculating density using W_MLE...")
    # Get W estimate using MLE
    try:
        if MLE_use_normal_dist:
            if MLE_all_priors_same:
                # collapse samples into one prior, m*n samples
                samples_clpsd = samples.reshape((-1, d_s)).unsqueeze(
                    dim=0
                )  # shape (1, m*n, d_s)
                phi_clpsd = phi[:1]  # shape (1, d_phi)

                W_MLE = NonLDSDensity.W_MLE_estimate(
                    samples_clpsd, phi_clpsd
                )  # shape (d_s, d_phi)
                sigma_MLE = NonLDSDensity.Sigma_MLE_estimate(
                    samples_clpsd, phi_clpsd
                )  # shape (d_s)
            else:
                W_MLE = NonLDSDensity.W_MLE_estimate(
                    samples, phi
                )  # shape (d_s, d_phi)
                sigma_MLE = NonLDSDensity.Sigma_MLE_estimate(
                    samples, phi
                )  # shape (d_s)

            nonLDS_density = NonLDSDensity(d_s, sigma_MLE)

            prob = nonLDS_density.density(x, W_MLE, phi[:1])  # shape (N, 1)
            plot_name = "Samples fit with Normal Dist (MLE)"
        else:
            W_MLE = exp_glm_density.__class__.W_MLE_estimate(
                samples, phi
            )  # shape (d_psi, d_phi)

            prob = exp_glm_density.density(x, W_MLE, phi[:1])  # shape (N, 1)
            plot_name = "Density w/ $\hat{W}_{MLE}$"
        mle_density = Density1D(
            name="MLE",
            x=x[:, 0],
            prob=utils.to_numpy(prob[:, 0]),
            plot_name=plot_name,
        )
        densities.append(mle_density)

        tvd_MLE = utils.tvd(true_density.x, true_density.prob, mle_density.prob)
        print(f"tvd_MLE: {tvd_MLE:0.5e}")
    except ValueError as e:
        warnings.warn(traceback.format_exc())

    return densities


def plot_sampled_points(
    d_s: int,
    samples: np.ndarray,
    plot_label: Optional[str] = None,
    densities: List[Density1D] = [],
    ax: Optional[axes.Axes] = None,
    plot_figsize: Tuple[int, int] = (8, 8),
    plot_fontsize: int = 16,
    plot_title: str = "",
    hist_alpha: float = 0.3,
) -> axes.Axes:
    """
    Plots samples. shape (n, d_s). If d_s == 1, also plots
    (x, prob) = densities[label] on the histogram of samples with corresponding
    labels.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=plot_figsize)

    if d_s == 1:
        # 1d data, plot the sampled points as a histogram to estimate density
        # Can calculate exact density in 1d, so plot that as well
        x_min = samples.min()
        x_max = samples.max()
        for density in densities:
            x_min = min(x_min, density.x.min())
            x_max = max(x_max, density.x.max())
        bin_vals = ax.hist(
            samples,
            bins=100,
            range=(x_min, x_max),
            density=True,
            alpha=hist_alpha,
            label=plot_label,
        )
        for density in densities:
            ax.plot(density.x, density.prob, label=density.plot_name)

    elif d_s == 2:
        # 2d data, plot the sampled points as scatter in xy plane
        ax.scatter(
            samples[:, 0], samples[:, 1], s=5, alpha=0.3, label=plot_label
        )
        plt.grid()

    plt.xlabel("$s'$", fontsize=plot_fontsize)
    plt.ylabel("$P_W (s' \\mid s, a)$", fontsize=plot_fontsize)
    plt.title(plot_title, fontsize=plot_fontsize)
    plt.legend()
    return ax


def plot_1d_density(
    exp_glm_density: ExpGLMDensity,
    W: torch.Tensor,
    s: torch.Tensor,
    a: torch.Tensor,
    phi_embedding: PhiEmbedding,
    num_samples: int,
    sampling_method: str = "hmc",
    sampling_kwargs: Dict[str, Any] = {
        "step_size": 0.05,
        "sampling_seed": SEED,
    },
    SM_lam: float = 0.0,
    MLE_use_normal_dist: bool = True,
    MLE_all_priors_same: bool = False,
    n_points_eval: int = 1000,
    compare_with_hmc_sampling: bool = False,
):
    assert exp_glm_density.d_s == 1
    assert s.shape[1] == 1
    m, d_s = s.shape
    sm_estimator = ScoreMatching(exp_glm_density)

    # shape (m, num_samples, d_s), (m, d_phi)
    samples, phi = exp_glm_density.sample_iid(
        W, s, a, phi_embedding, num_samples, sampling_method, sampling_kwargs
    )

    densities = get_1d_densities(
        exp_glm_density,
        W,
        phi,
        n_points_eval,
        samples,
        sm_estimator,
        SM_lam=SM_lam,
        MLE_use_normal_dist=MLE_use_normal_dist,
        MLE_all_priors_same=MLE_all_priors_same,
    )

    ## Also plot HMC sampling for comparison
    if (sampling_method != "hmc") and compare_with_hmc_sampling:
        hmc_sampling_kwargs = {"step_size": 0.05, "sampling_seed": SEED}
        # shape (m, num_samples, d_s), (m, d_phi)
        hmc_samples, _ = exp_glm_density.sample_iid(
            W, s, a, phi_embedding, num_samples, "hmc", hmc_sampling_kwargs
        )
        ax = plot_sampled_points(
            d_s,
            hmc_samples.detach().reshape((-1, d_s)).numpy(),
            plot_label=f"Sampled with hmc method",
            densities=[],
            plot_title="",
        )
    else:
        ax = None

    ax = plot_sampled_points(
        d_s,
        utils.to_numpy(samples[0].detach().reshape((-1, d_s))),
        plot_label=f"Sampled with {sampling_method} method",
        densities=densities,
        plot_title=repr(exp_glm_density),
        ax=ax,
    )

    plt.show()

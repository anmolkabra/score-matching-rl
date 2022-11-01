from typing import Tuple, Union

import numpy as np
import torch

import utils
from data_gen.density import ExpGLMDensity, NonLDSDensity

T = Union[np.ndarray, torch.Tensor]


class ScoreMatching:
    """
    Estimate parameter W (of density P_W (x | s, a) modeled as an Exponential
    GLM.) using Score Matching.
    """

    def __init__(self, exp_glm_density: ExpGLMDensity):
        self.exp_glm_density = exp_glm_density

    def psi_grad1(self, x: T) -> torch.Tensor:
        """
        Calculates the first gradient of psi, w.r.t. each coordinate of x (for
        each point in x). Check the implementation for returned tensor's shape.

        Args:
            x: points to evaluate on. shape (n, d_s) where n is the number of
                points.

        Returns:
            shape (n, d_s, d_psi) where psi_grad1[t, i, :] is the first gradient
            of psi w.r.t. coordinate i for point t.
        """
        x = utils.to_torch(x)
        x.requires_grad_(True)

        # shape (n, d_psi, d_s)
        psi_grad1 = utils.batch_jacobian(self.exp_glm_density.psi, x)
        return psi_grad1.permute(0, 2, 1)

    def psi_grad2(self, x: T) -> torch.Tensor:
        """
        Calculates the second gradient of psi, w.r.t. each coordinate of x (for
        each point in x). Check the implementation for returned tensor's shape.

        Args:
            x: points to evaluate on. shape (n, d_s) where n is the number of
                points.

        Returns:
            shape (n, d_s, d_psi) where psi_grad2[t, i, :] is the second
            gradient of psi w.r.t. coordinate i for point t.
        """
        x = utils.to_torch(x)
        x.requires_grad_(True)

        # TODO If already computed psi_grad1, differentiate that instead
        # shape (n, d_psi, d_s)
        psi_grad2 = utils.batch_diag_hessian(self.exp_glm_density.psi, x)
        return psi_grad2.permute(0, 2, 1)

    def logq_grad1(self, x: T) -> torch.Tensor:
        """
        Calculates the first gradient of log q, w.r.t. each coordinate of x.

        Args:
            x: points to evaluate on. shape (n, d_s) where n is the number of
                points.

        Returns:
            shape (n, d_s) where logq_grad1[t, i] is the first gradient of
            log q w.r.t. coordinate i for point t.
        """
        x = utils.to_torch(x)
        x.requires_grad_(True)

        # shape (n, d_s)
        return utils.batch_jacobian(self.exp_glm_density.logq, x)

    def W_estimate(
        self, x: T, phi: T, lam: float, return_Vb_estimates: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Calculates the score matching estimate of W.

        Args:
            x: points to use as density for estimation. shape (n, d_s) where n
                is the number of points for estimation.
            phi: value of phi(s, a) for priors (s, a) of density x. shape (n,
                d_phi) where n is the number of points for estimation.
            lam: regularization parameter.
            return_Vb_estimates: (default False) flag to return V and b
                estimates as well.

        Returns:
            if return_Vb_estimates: W_estimate, V_estimate, b_estimate
            else: W_estimate
            where W_estimate: shape (d_psi, d_phi), V_estimate: shape
                (d_psi*d_phi, d_psi*d_phi), b_estimate: shape (d_psi*d_phi)
        """
        assert x.shape[0] == phi.shape[0]
        assert x.shape[1] == self.exp_glm_density.d_s
        n, d_s = x.shape
        d_phi = phi.shape[1]

        x = utils.to_torch(x)
        phi = utils.to_torch(phi)

        psi_grad1 = self.psi_grad1(x)  # shape (n, d_s, d_psi)
        d_psi = psi_grad1.shape[2]

        # Each row of psi_grad1 is the first gradient for coordinate i. Get
        # outer product for each row, then collapse along second dim to get C
        # for # **one** input point x.
        C = torch.sum(
            psi_grad1[:, :, :, None] @ psi_grad1[:, :, None, :], dim=1
        )  # shape (n, d_psi, d_psi)

        psi_grad2 = self.psi_grad2(x)  # shape (n, d_s, d_psi)
        logq_grad1 = self.logq_grad1(x)  # shape (n, d_s)

        xi = logq_grad1[:, :, None] * psi_grad1  # shape (n, d_s, d_psi)
        xi = torch.sum(xi + psi_grad2, dim=1)  # shape (n, d_psi)

        # TODO Implement V_estimate and b_estimate cleverly
        # Naive implementation of Phi, V_estimate, and b_estimate
        # (d_psi*d_phi, d_psi) for each data point
        Phi = torch.zeros((n, d_psi * d_phi, d_psi))
        for i in range(d_psi):
            Phi[:, i * d_phi : (i + 1) * d_phi, i] = phi

        V_estimate = torch.sum(
            (Phi @ C) @ Phi.transpose(2, 1), dim=0
        )  # shape (d_psi*d_phi, d_psi*d_phi)
        reg = lam * torch.eye(V_estimate.shape[0])
        b_estimate = torch.sum(
            (Phi @ xi[:, :, None]).squeeze(2), dim=0
        )  # shape (d_psi*d_phi)

        try:
            W_estimate = -torch.linalg.solve(V_estimate + reg, b_estimate)
        except RuntimeError as e:
            raise ValueError(
                "Density does not satisfy assumptions for Score"
                "Matching, or density functions (e.g. logq, psi"
                "etc.) are not defined correctly as functions of"
                "data points"
            ) from e

        W_estimate = W_estimate.reshape((d_psi, d_phi))

        if return_Vb_estimates:
            return W_estimate, V_estimate, b_estimate
        else:
            return W_estimate


class NonLDSScoreMatching(ScoreMatching):
    """
    x = W phi(s, a) + eps, where eps is N(0, diag(sigma))

    Score Matching Estimator for parameter W in nonLDS setting. d_psi == d_s.

    Closed form expressions for first and second derivatives of q,
    psi (useful for checking correctness of base class ScoreMatching).
    """

    def __init__(self, exp_glm_density: ExpGLMDensity):
        assert isinstance(exp_glm_density, NonLDSDensity)
        super(NonLDSScoreMatching, self).__init__(exp_glm_density)

    def psi_grad1(self, x: T) -> torch.Tensor:
        """
        Since the first gradient is independent of each point in x for nonLDS, 1
        copy of gradient is returned instead of n copies.

        Returns:
            shape (d_s, d_s) where psi_grad1[i, :] is the first gradient of
            psi w.r.t. coordinate i.
        """
        # psi_grad1[j, :] is the jth column of sigma inv
        return torch.diag(self.exp_glm_density.sigma_inv).T

    def psi_grad2(self, x: T) -> torch.Tensor:
        """
        Since the second gradient is independent of each point in x and
        coordinates of x for nonLDS, 1 copy of gradient is returned instead of n
        copies.

        Returns:
            shape (d_s) where psi_grad2[i] is the second gradient of
            psi w.r.t. coordinate i.
        """
        return torch.zeros(self.exp_glm_density.d_s)

    def logq_grad1(self, x: T) -> torch.Tensor:
        x = utils.to_torch(x)
        return -x * self.exp_glm_density.sigma_inv  # shape (n, d_s)

    def W_estimate(
        self, x: T, phi: T, lam: float, return_Vb_estimates: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        assert x.shape[0] == phi.shape[0]
        assert x.shape[1] == self.exp_glm_density.d_s
        n, d_s = x.shape
        d_phi = phi.shape[1]

        x = utils.to_torch(x)
        phi = utils.to_torch(phi)

        # In nonLDS, psi_grad1 is independent of the input point x.
        psi_grad1 = self.psi_grad1(x)  # shape (d_s, d_s)

        # Each row of psi_grad1 is the first gradient for coordinate i. Get
        # outer product for each row, then collapse along first dim to get C for
        # **one** input point x. (C is independent of input point x, so we don't
        # make n copies of C_t)
        C = torch.sum(
            psi_grad1[:, :, None] @ psi_grad1[:, None, :], dim=0
        )  # shape (d_s, d_s)

        psi_grad2 = self.psi_grad2(x)  # shape (d_s)
        logq_grad1 = self.logq_grad1(x)  # shape (n, d_s)

        # Since psi_grad1 is independent of points x but logq_grad1 is not, use
        # broadcasting to get first term of xi.
        xi = logq_grad1[:, :, None] * psi_grad1  # shape (n, d_s, d_s)
        xi = torch.sum(xi + psi_grad2, dim=1)  # shape (n, d_s)

        # TODO Implement V_estimate and b_estimate cleverly
        # Naive implementation of Phi, V_estimate, and b_estimate
        # (d_s*d_phi, d_s) for each data point
        Phi = torch.zeros((n, d_s * d_phi, d_s))
        for i in range(d_s):
            Phi[:, i * d_phi : (i + 1) * d_phi, i] = phi

        V_estimate = torch.sum(
            (Phi @ C) @ Phi.transpose(2, 1), dim=0
        )  # shape (d_s*d_phi, d_s*d_phi)
        reg = lam * torch.eye(V_estimate.shape[0])
        b_estimate = torch.sum(
            (Phi @ xi[:, :, None]).squeeze(2), dim=0
        )  # shape (d_s*d_phi)

        W_estimate = -torch.linalg.solve(V_estimate + reg, b_estimate)
        W_estimate = W_estimate.reshape((d_s, d_phi))

        if return_Vb_estimates:
            return W_estimate, V_estimate, b_estimate
        else:
            return W_estimate


class NonLDSScoreMatchingAutograd(ScoreMatching):
    """
    x = W phi(s, a) + eps, where eps is N(0, diag(sigma))

    Score Matching Estimator for parameter W in nonLDS setting. d_psi == d_s

    Implemented using `torch.autograd`.
    """

    def __init__(self, exp_glm_density: ExpGLMDensity):
        assert isinstance(exp_glm_density, NonLDSDensity)
        super(NonLDSScoreMatchingAutograd, self).__init__(exp_glm_density)

    def psi_grad1(self, x: T) -> torch.Tensor:
        """
        Since the first gradient is independent of each point in x for nonLDS, 1
        copy of gradient is returned instead of n copies.

        Returns:
            shape (d_s, d_s) where psi_grad1[i, :] is the first gradient of
            psi w.r.t. coordinate i.
        """
        x = utils.to_torch(x)
        x.requires_grad_(True)

        # Just use the first point as placeholder to get the jacobian of psi
        # shape (1, d_s, d_s)
        psi_grad1 = utils.batch_jacobian(self.exp_glm_density.psi, x[:1])
        return psi_grad1[0].T

    def psi_grad2(self, x: T) -> torch.Tensor:
        """
        Since the second gradient is independent of each point in x and
        coordinates of x for nonLDS, 1 copy of gradient is returned instead of n
        copies.

        Returns:
            shape (d_s) where psi_grad2[i] is the second gradient of
            psi w.r.t. coordinate i.
        """
        x = utils.to_torch(x)
        x.requires_grad_(True)

        # TODO If already computed psi_grad1, differentiate that instead
        # Just use the first point as placeholder to get the diag hessian of psi
        # shape (1, d_s, d_s)
        psi_grad2 = utils.batch_diag_hessian(self.exp_glm_density.psi, x[:1])
        # psi_grad2 in this case is independent of coordinates, so choose the
        # first one
        return psi_grad2[0, :, 0]  # shape (d_s)

    def logq_grad1(self, x: T) -> torch.Tensor:
        x = utils.to_torch(x)
        x.requires_grad_(True)

        # shape (n, d_s)
        return utils.batch_jacobian(self.exp_glm_density.logq, x)

    def W_estimate(
        self, x: T, phi: T, lam: float, return_Vb_estimates: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        assert x.shape[0] == phi.shape[0]
        assert x.shape[1] == self.exp_glm_density.d_s
        n, d_s = x.shape
        d_phi = phi.shape[1]

        x = utils.to_torch(x)
        phi = utils.to_torch(phi)

        # In nonLDS, psi_grad1 is independent of the input point x.
        psi_grad1 = self.psi_grad1(x)  # shape (d_s, d_s)

        # Each row of psi_grad1 is the first gradient for coordinate i. Get
        # outer product for each row, then collapse along first dim to get C for
        # **one** input point x. (C is independent of input point x, so we don't
        # make n copies of C_t)
        C = torch.sum(
            psi_grad1[:, :, None] @ psi_grad1[:, None, :], dim=0
        )  # shape (d_s, d_s)

        psi_grad2 = self.psi_grad2(x)  # shape (d_s)
        logq_grad1 = self.logq_grad1(x)  # shape (n, d_s)

        # Since psi_grad1 is independent of points x but logq_grad1 is not, use
        # broadcasting to get first term of xi.
        xi = logq_grad1[:, :, None] * psi_grad1  # shape (n, d_s, d_s)
        xi = torch.sum(xi + psi_grad2, dim=1)  # shape (n, d_s)

        # TODO Implement V_estimate and b_estimate cleverly
        # Naive implementation of Phi, V_estimate, and b_estimate
        # (d_s*d_phi, d_s) for each data point
        Phi = torch.zeros((n, d_s * d_phi, d_s))
        for i in range(d_s):
            Phi[:, i * d_phi : (i + 1) * d_phi, i] = phi

        V_estimate = torch.sum(
            (Phi @ C) @ Phi.transpose(2, 1), dim=0
        )  # shape (d_s*d_phi, d_s*d_phi)
        reg = lam * torch.eye(V_estimate.shape[0])
        b_estimate = torch.sum(
            (Phi @ xi[:, :, None]).squeeze(2), dim=0
        )  # shape (d_s*d_phi)

        W_estimate = -torch.linalg.solve(V_estimate + reg, b_estimate)
        W_estimate = W_estimate.reshape((d_s, d_phi))

        if return_Vb_estimates:
            return W_estimate, V_estimate, b_estimate
        else:
            return W_estimate

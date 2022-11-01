import abc
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

T = Union[np.ndarray, torch.Tensor]


class PhiEmbedding(abc.ABC):
    def __init__(self, d_s: int, d_a: int):
        """
        Args:
            d_s: state space dimension.
            d_a: action space dimension.
        """
        self.d_s = d_s
        self.d_a = d_a

    @property
    @abc.abstractmethod
    def d_phi(self) -> int:
        pass

    @abc.abstractmethod
    def get_phi(self, s: T, a: T) -> torch.Tensor:
        """
        Returns an embedding of priors (s, a).

        Args:
            s: prior states. shape (n, d_s) where n is the number of points.
            a: actions taken on each state. shape (n, d_a).

        Returns:
            (n, d_phi) where d_phi is embedding's dimension.
        """
        pass


class LDSPhiEmbedding(PhiEmbedding):
    """
    phi(s, a) = [s, a] so that d_phi = d_s + d_a.
    """

    def __init__(self, d_s: int, d_a: int):
        super(LDSPhiEmbedding, self).__init__(d_s, d_a)

    @property
    def d_phi(self) -> int:
        return self.d_s + self.d_a

    def get_phi(self, s: T, a: T) -> torch.Tensor:
        s = utils.to_torch(s)
        a = utils.to_torch(a)
        return torch.hstack([s, a])  # shape (n, d_s+d_a)


class ConstPhiEmbedding(PhiEmbedding):
    """
    phi(s, a) = [value]*d_s so that d_phi = d_s.
    """

    def __init__(self, d_s: int, d_a: int, value: float):
        super(ConstPhiEmbedding, self).__init__(d_s, d_a)
        self.value = value

    @property
    def d_phi(self) -> int:
        return self.d_s

    def get_phi(self, s, a) -> torch.Tensor:
        return torch.full(s.shape, self.value)  # shape (m, d_s)


class NNPhiEmbedding(PhiEmbedding, nn.Module):
    """
    phi(s, a) = NN(s, a) where the weights and bias of NN are randomly
    initialized.

    The NN has `layers` layers, each with `nunits` hidden units. Input layer
    takes in [s, a] and outputs d_phi = nunits dimensional embedding.
    """

    def __init__(self, d_s: int, d_a: int, layers: int, nunits: int):
        PhiEmbedding.__init__(self, d_s, d_a)
        nn.Module.__init__(self)
        self.layers = layers
        self.nunits = nunits

        self.fcs = []
        for i in range(layers):
            # Create a layer
            in_c = self.d_s + self.d_a if i == 0 else self.nunits
            out_c = self.nunits
            layer = nn.Linear(in_c, out_c)
            nn.init.normal_(layer.weight)
            nn.init.normal_(layer.bias)

            self.fcs.append(layer)
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        x = self.fcs[-1](x)
        x = F.normalize(x, p=2, dim=1)
        return x

    @property
    def d_phi(self) -> int:
        return self.nunits

    def get_phi(self, s: T, a: T) -> torch.Tensor:
        s = utils.to_torch(s)
        a = utils.to_torch(a)
        return self.forward(torch.hstack([s, a])).detach()  # shape (n, nunits)

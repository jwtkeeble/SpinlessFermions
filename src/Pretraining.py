import torch
import torch.nn as nn

import numpy as np
from scipy.special import factorial

from typing import Tuple
from torch import Tensor

class HermitePolynomialMatrix(nn.Module):

    def __init__(self, num_particles: int) -> None:
        r"""Constructor of class
        :param nfermions: The number of fermions in the exact solution
        :type nfermions: int

        :param device:
        :type device: device container
        """
        super(HermitePolynomialMatrix, self).__init__()
        self.num_particles = num_particles

    def log_factorial(self, n: int) -> float:
      return np.sum(np.log(np.arange(1,n+1,1)))

    def hermite(self, n: int, x: Tensor) -> Tensor:
        return torch.special.hermite_polynomial_h(x, n)
        
    def orbital(self, n: int, x: Tensor) -> Tensor:
        r"""Method class to calculate the n-th single particle orbital of
            the groundstate of the non-interacting Harmonic Oscillator.
            Tensors will be passed to the cpu to compute the Hermite Polynomials
            and subsequently passed back to `device`

        :param n: The order of the Hermite polynomial
        :type n: int

        :param x: The many-body positions
        :type x: class: `torch.Tensor`

        :return out: The values of single particle orbitals of the n-th order
                     for current many-body positions `x`.
        :type out: class: `torch.Tensor`
        """
        Hn_x = self.hermite(n, x)
        
        env = torch.exp(-0.5*x**2)
        norm = ( (2**n)*factorial(n)*(np.sqrt(np.pi)) )**(-0.5)

        orbitaln_x = Hn_x * norm * env
        return orbitaln_x

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        return torch.stack([ self.orbital(n=n, x=x) for n in range(self.num_particles) ], dim=-1).unsqueeze(1) #unsqueeze to state only one matrix at index, 1.
        
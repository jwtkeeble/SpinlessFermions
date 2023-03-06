import torch
from torch import nn, Tensor
from functorch import jacrev
from typing import Tuple

import numpy as np

class HarmonicOscillatorWithInteraction1D(nn.Module):

    def __init__(self, fnet: nn.Module, V0: float, sigma0: float, nchunks: int) -> None:
        super(HarmonicOscillatorWithInteraction1D, self).__init__()

        self.fnet = fnet
        self.V0 = V0
        self.sigma0 = sigma0
        self.nchunks = nchunks
    
        self.gauss_const = (self.V0/(np.sqrt(2*np.pi)*self.sigma0))

        def calc_logabs(params: Tuple[Tensor], x: Tensor) -> Tuple[Tensor]:
            _, logabs = self.fnet(params, x)
            return logabs

        def dlogabs_dx(params: Tuple[Tensor], x: Tensor) -> Tuple[Tensor]:
            grad_logabs = jacrev(calc_logabs, argnums=1, has_aux=False)(params, x)
            return grad_logabs, grad_logabs

        def d2logabs_dx2(params: Tuple[Tensor], x: Tensor) -> Tuple[Tensor]:
            grad2_logabs, grad_logabs = jacrev(dlogabs_dx, argnums=1, has_aux=True)(params, x)
            return -0.5*(grad2_logabs.diagonal(0,-2,-1).sum() + grad_logabs.pow(2).sum())

        #merge into a single lambda call so it can be grad/vmap'd in one go?
        self.kinetic_from_log = lambda params, x: d2logabs_dx2(params, x)
        self.potential_fn = lambda x: 0.5*(x.pow(2).sum(-1))
        self.gauss_int_fn = lambda x: self.gauss_const * ( torch.exp(-(x.squeeze(-2) - x.unsqueeze(-1))**2/(2*sigma0**2)).triu(diagonal=1).sum(dim=(-2,-1)) )

    def kinetic(self, params: Tuple[Tensor], x: Tensor) -> Tensor:
        return self.kinetic_from_log(params, x)

    def potential(self, x: Tensor) -> Tensor:
        return self.potential_fn(x)

    def gaussian_interaction(self, x: Tensor) -> Tensor:
        return self.gauss_int_fn(x)

    def forward(self, params: Tuple[Tensor], x: Tensor) -> Tensor:
        _kin = self.kinetic(params, x)
        _pot = self.potential(x)
        _int = self.gaussian_interaction(x)

        _eloc = _kin + _pot + _int
        return _eloc
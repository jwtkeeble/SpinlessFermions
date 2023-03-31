import torch
from torch import nn, Tensor
from typing import Tuple

import numpy as np

class HarmonicOscillatorWithInteraction1D(nn.Module):

    def __init__(self, net: nn.Module, V0: float, sigma0: float, nchunks: int) -> None:
        super(HarmonicOscillatorWithInteraction1D, self).__init__()

        self.net = net

        self.V0 = V0
        self.sigma0 = sigma0
    
        self.gauss_const = (self.V0/(np.sqrt(2*np.pi)*self.sigma0))

    def kinetic(self, x: Tensor) -> Tensor:
        xis = [xi.requires_grad_() for xi in x.flatten(start_dim=1).t()]
        xs_flat = torch.stack(xis, dim=1)

        _, ys = self.net(xs_flat.view_as(x))

        ones = torch.ones_like(ys)

        (dy_dxs, ) = torch.autograd.grad(ys, xs_flat, ones, retain_graph=True, create_graph=True)


        lay_ys = sum(torch.autograd.grad(dy_dxi, xi, ones, retain_graph=True, create_graph=False)[0] \
                    for xi, dy_dxi in zip(xis, (dy_dxs[..., i] for i in range(len(xis))))
        )

        ek_local_per_walker = -0.5 * (lay_ys + dy_dxs.pow(2).sum(-1)) #move const out of loop?
        return ek_local_per_walker

    def potential(self, x: Tensor) -> Tensor:
        return 0.5*(x.pow(2).sum(-1))

    def gaussian_interaction(self, x: Tensor) -> Tensor:
        return self.gauss_const * ( torch.exp(-(x.unsqueeze(-2) - x.unsqueeze(-1))**2/(2*self.sigma0**2)).triu(diagonal=1).sum(dim=(-2,-1)) )

    def forward(self, x: Tensor) -> Tensor:
        _kin = self.kinetic(x)
        _pot = self.potential(x)
        _int = self.gaussian_interaction(x)

        _eloc = _kin + _pot + _int
        return _eloc
    

class GaussianInteraction1D(nn.Module):

    def __init__(self, net: nn.Module, V0: float, sigma0: float) -> None:
        super(GaussianInteraction1D, self).__init__()

        self.net = net

        self.V0 = V0
        self.sigma0 = sigma0
    
        self.gauss_const = (self.V0/(np.sqrt(2*np.pi)*self.sigma0))

    def kinetic(self, x: Tensor) -> Tensor:
        xis = [xi.requires_grad_() for xi in x.flatten(start_dim=1).t()]
        xs_flat = torch.stack(xis, dim=1)

        _, ys = self.net(xs_flat.view_as(x))

        ones = torch.ones_like(ys)

        (dy_dxs, ) = torch.autograd.grad(ys, xs_flat, ones, retain_graph=True, create_graph=True)


        lay_ys = sum(torch.autograd.grad(dy_dxi, xi, ones, retain_graph=True, create_graph=False)[0] \
                    for xi, dy_dxi in zip(xis, (dy_dxs[..., i] for i in range(len(xis))))
        )

        ek_local_per_walker = -0.5 * (lay_ys + dy_dxs.pow(2).sum(-1)) #move const out of loop?
        return ek_local_per_walker

    def gaussian_interaction(self, x: Tensor) -> Tensor:
        return self.gauss_const * ( torch.exp(-(x.unsqueeze(-2) - x.unsqueeze(-1))**2/(2*self.sigma0**2)).triu(diagonal=1).sum(dim=(-2,-1)) )

    def forward(self, x: Tensor) -> Tensor:
        _kin = self.kinetic(x)
        _int = self.gaussian_interaction(x)

        _eloc = _kin + _int
        return _eloc
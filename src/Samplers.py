#Pytorch package
import torch
from torch import nn, Tensor

#Typecasting
from typing import Tuple, Callable

def rw_metropolis_kernel(logpdf: Callable, position: Tensor, log_prob: Tensor, sigma: float):
    #Shamelessly taken from https://rlouf.github.io/post/jax-random-walk-metropolis/
    proposal = position + sigma * torch.randn(position.shape, device=position.device)
    proposal_logprob = logpdf(proposal)

    log_uniform = torch.log(torch.rand(proposal_logprob.shape, device=position.device))
    accept = log_uniform < proposal_logprob - log_prob
    
    acceptance_rate = accept.float().mean()

    position = accept[:, None]*proposal + (~accept[:,None])*position
    log_prob = torch.where(accept, proposal_logprob, log_prob)
    return position, log_prob, acceptance_rate

class MetropolisHastings(nn.Module):

    def __init__(self, network: nn.Module, dof: int, nwalkers: int, target_acceptance: float) -> None:
        super(MetropolisHastings, self).__init__()

        self.network = network
        self.dof = dof
        self.nwalkers = nwalkers
        self.target_acceptance = target_acceptance

        self.device = next(self.network.parameters()).device

        self.sigma = torch.tensor(1.0, device=self.device)
        self.acceptance_rate = torch.tensor(0.0, device=self.device)

        self.chains = torch.randn(size=(self.nwalkers, self.dof),
                                        device=self.device,
                                        requires_grad=False) #isotropic initialisation.
        
        _pretrain = self.network.pretrain
        if(_pretrain==True):
            self.network.pretrain=False
        self.log_prob = self.network(self.chains)[1].mul(2)
        self.network.pretrain=_pretrain

    def log_pdf(self, x: Tensor) -> Tensor:
      _pretrain = self.network.pretrain
      if(_pretrain==True):
        self.network.pretrain=False
      _, logabs = self.network(x)
      self.network.pretrain=_pretrain
      return 2.*logabs

    def _update_sigma(self, acceptance_rate: Tensor) -> Tensor: 
        #Shamelessly taken from https://github.com/deepqmc/deepqmc
        scale_factor = self.target_acceptance / max(acceptance_rate, 0.05)
        return self.sigma / scale_factor

    @torch.no_grad()
    def forward(self, n_sweeps: int) -> Tuple[Tensor, Tensor]:
        for _ in range(n_sweeps):
            self.chains, self.log_prob, self.acceptance_rate = rw_metropolis_kernel(logpdf=self.log_pdf,
                                                                                    position=self.chains,
                                                                                    log_prob=self.log_prob,
                                                                                    sigma=self.sigma)
            if(self.target_acceptance is not None):
              self.sigma = self._update_sigma(self.acceptance_rate)
            else:
              self.sigma = self.sigma
        return self.chains, self.log_prob 
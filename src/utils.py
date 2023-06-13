import torch
from torch import nn, Tensor
import time
from tqdm import tqdm
from typing import Callable, Tuple
import numpy as np
import os, warnings
from Writers import WriteToFile


def unsqueeze_to_size(m: Tensor, s: int) -> Tensor:
   cur = len(m.shape)
   for _ in range(cur, s, 1): #while loop instead?
     m=m.unsqueeze(-1)
   return m

def count_parameters(model: nn.Module) -> int:
    r""" Method to count the number of parameters of a Deep Neural Network.
        :param model: An `nn.Module` object representing a Deep Neural Network
        :type model: class `nn.Module`
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_params(net, requires_grad: bool=False):
    #_, params = make_functional(net)
    #return [p.requires_grad_(requires_grad) for p in params]
    raise ValueError("")

def sync_time() -> None:
    torch.cuda.synchronize()
    return time.perf_counter()

def clip(elocal: Tensor, clip_factor: int):
    r""" Method for clipping a Tensor's elements in accordance to its l1-norm.
        The :math:`l_{1}`-norm is defined as,

        .. math::

            l_{1} = \frac{1}{N}\sum_{i=1}^{N} \vert E_{L,i} - \mathbb{E}_{x \sim \vert\Psi\vert^2} \left[\mathrm{E}_{L}\right] \vert

        The local energies are now clamped within a range of :math:`[\mathbb{E}_{x \sim \vert\Psi\vert^2} \left[\mathrm{E}_{L}\right] - c l_{1}, \mathbb{E}_{x \sim \vert\Psi\vert^2} \left[\mathrm{E}_{L}\right] + c l_{1}]` where :math:`c` is the `clip_factor` which defaults to 5.
    """
    median = elocal.median()
    variation = torch.mean( torch.abs(elocal - median) )
    window = clip_factor*variation
    minima, maxima = (median-window).item(), (median+window).item()
    return torch.clip(elocal, minima, maxima)

def calc_pretraining_loss(net_matrix: Tensor, target_matrix: Tensor) -> Tuple[Tensor,Tensor]:
    r"""Calculates the pre-training loss of the Deep Neural Network when
        pre-training towards the non-interacting groundstate for D Generalised Slater Matrices

        :param net_matrix: A Tensor which contains the slater matrices that are outputed from the Deep Neural Network
        :type net_matrix: class: `torch.Tensor`

        :param target_matrix: A Tensor which contains the exact groundstate slater matrix
        :type target_matrix: class: `torch.Tensor`

        :return out: returns a Tuple containing the mean and std. dev. of the pre-training loss.
        :type out: `Tuple[torch.Tensor, torch.Tensor]`
    """
    loss_values = (net_matrix-target_matrix).pow(2).sum(dim=(-3,-2,-1)) #sum over dets, particles, orbitals
    #var, mean = torch.var_mean(loss_values, unbiased=True)
    mean = torch.mean(loss_values)
    var = torch.var(loss_values)
    return mean, var.sqrt()

def get_groundstate(A: int, V0: int, datapath: str) -> float:
    r"""Method to get the analytical groundstate for the Harmonic Oscillator.
        It currently supports up to :math:`2 \leq A \leq 5` and :math:`-20 \leq V_{0} \leq +20`
        (in steps in 1) and :math:`\sigma_{0} = 0.5`

        :param A: The number of fermions
        :type A: int

        :param V0: The interaction strength
        :type V0: int

        :param datapath: The datapath of the groundstate energies file
        :type: str
    """
    if(A<2 or A>5):
        #raise ValueError("Only have energies for 2 <= A <= 5") 
        warnings.warn("Only have energies for 2 <= A <= 5. Returning non-interacting value.")
        return A**2/2
    if(V0 < -20 or V0 > 20):
        #raise ValueError("Only have energies for -20 <= V0 <= +20")
        warnings.warn("Only have energies for -20 <= V0 <= +20. Returning non-interacting value.")
        return A**2/2


    filestr = "%s%i%s" % (datapath, A, "p_20modes.txt")

    data = np.genfromtxt(filestr)

    idx = int(V0+20)

    gs = data[idx, 1] #get gs values from .txt file 
    return gs

def load_dataframe(filename: str) -> WriteToFile:
    r"""Loads an WriterObject with an existing .csv file if it exists, but if it doesn't a new one is instantiated.

        :param filename: The filename to which we load/write data
    """
    if(os.path.isfile(filename)):
        print("Dataframe already exists %s - appending" % (filename))
        writer = WriteToFile(load=filename, filename=filename)
    else:
        print("Saving data to %s - new" % (filename))
        writer = WriteToFile(load=None, filename=filename)
    return writer


def load_model(model_path: str, device: torch.device, net: nn.Module, optim: torch.optim.Optimizer, sampler: nn.Module) -> dict:
        #, net: nn.Module, optim: torch.optim.Optimizer, sampler: nn.Module):
    r"""A function to load in an object saved from `torch.save` if the file exists already. The method returns a dict 
    """
    if(os.path.isfile(model_path)):
        print("Model already exists %s - transferring" % (model_path))
        state_dict = torch.load(f=model_path, map_location=device)

        start=state_dict['epoch']+1 #start at next epoch
        net.load_state_dict(state_dict['model_state_dict'])
        optim.load_state_dict(state_dict['optim_state_dict'])
        optim._steps = start        #update epoch in optim too!
        loss = state_dict['loss']
        sampler.chains = state_dict['chains']
        sampler.log_prob = state_dict['log_prob'] #cache log_prob too!
        sampler.sigma = state_dict['sigma'] #optimal sigma for proposal distribution!
        print("Model resuming at epoch %6i with energy %6.4f MeV" % (start, loss))
    else:
        print("Saving model to %s - new" % (model_path))
        start=0
    return {'start':start, 'device':device, 'net':net, 'optim':optim, 'sampler':sampler}

def get_batches_of_measurements(nbatches: int, nwalkers: int, sampler: nn.Module, burn_in: int, calc_local_energy: Callable):
  r"""Simple function to get batches of measurements with ease.
      Pass in sampler and function of which you want measurements with the number of walkers and burn_in.

        :param nbatches: The number of batches of measurements
        :type nbatches: int

        :param nwalkers: The number of walkers (or particles in MCMC languages) per batch of measurements
        :type nwalkers: int

        :param sampler: The Metropolis-Hastings Sampler class that returns many-body positions
        :type sampler: class: `nn.Module`

        :param burn_in: The burn-in or 'thermalisation' time of the Metropolis-Hastings Sampler
        :type burn_in: int

        :param calc_local_energy: A callable which returns a vector of samples for a single batch of walkers applied to given Callable
        :type calc_local_energy: `Callable`

        :return samples: Returns a Tensor containing all samples of the given function
        :type samples: class: `torch.Tensor`
  """
  #measure the groundstate
  samples = torch.zeros((nbatches, nwalkers), device=sampler.device) #store local energy values

  for batch in tqdm(range(nbatches)):
    X, acceptance = sampler(burn_in)
    X=X.requires_grad_(True)
    samples[batch, :] = calc_local_energy(X).detach_() #just save mean values? compare with autocorrelation method?

  return samples
"""
def get_batches_of_local_energies(calc_elocal: Callable,
                                  sampler: nn.Module,
                                  nbatches: int,
                                  chunks: int,
                                  n_sweeps: int,
                                  storage_device: torch.device):
    
    sampler.target_acceptance = None
   
    _, params = make_functional(sampler.network) #get params of current network
    nwalkers = sampler.nwalkers
    samples = torch.zeros((nbatches, nwalkers), device=storage_device) #store local energy values
    print("done")
    for batch in tqdm(range(nbatches)):
      x, _ = sampler(n_sweeps)

      samples[batch, :] = xmap(calc_elocal, in_dims=(None, 0), chunks=chunks)(params, x).detach_().to(storage_device)
    return samples

"""
def round_to_err(x, dx):
    """
    Rounds x to first significant figure of dx (i.e. +- 1 sig fig of error)

    :param x: value
    :param dx: uncertainty in value
    :return x, dx: value and its uncertainty rounded appropriately
    """

    #If uncertainty is zero, round to some sensible number of sfs
    if np.isclose(dx, 0.0):
        first_sf = 6
    else:
        # First, get number of dp needed to round dx to 1 sf
        first_sf = -int(np.floor(np.log10(abs(dx))))

    # Then, return x and dx rounded to this number of dp
    return round(x, first_sf), round(dx, first_sf)

#https://stackoverflow.com/questions/34031402/
def str_with_err(value, error):
    digits = -int(np.floor(np.log10(error)))
    return "{0:.{2}f}({1:.0f})".format(value, error*10**digits, digits)

def eyes_like(M: Tensor) -> Tensor:
    return torch.ones(*M.shape[0:-1], device=M.device, dtype=M.dtype).diag_embed()
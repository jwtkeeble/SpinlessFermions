import torch
from torch.autograd import Function
from torch import Tensor, nn

def exclusive_cumsum(x: Tensor, dim: int):
  res = x.cumsum(dim=dim).roll(1)
  res[...,0]=0.
  return res
  
def reverse_exclusive_cumsum(x: Tensor, dim: int):
  res = x.flip(dim).cumsum(dim=dim)
  res[...,-1]=0.
  res=res.roll(1).flip(-1) 
  return res

def get_log_gamma(log_sigma: Tensor):
  lower = exclusive_cumsum(log_sigma, dim=-1)
  upper = reverse_exclusive_cumsum(log_sigma, dim=-1)
  log_gamma = lower + upper

  return log_gamma
  
def get_log_rho(log_sigma: Tensor):
  lower_cumsum = exclusive_cumsum(log_sigma, -1)
  upper_cumsum = reverse_exclusive_cumsum(log_sigma, -1)
  v = lower_cumsum.unsqueeze(-1) + upper_cumsum.unsqueeze(-2)

  s_mat = torch.transpose(torch.tile(log_sigma[..., None], [1] * len(log_sigma.shape) + [log_sigma.shape[-1]]), -2, -1)
  triu_s_mat = torch.triu(s_mat, 1)

  z = exclusive_cumsum(triu_s_mat, -1)

  r = torch.exp(z + v) #no shift?  
  r = torch.triu(r, 1)
  return torch.log(r + torch.transpose(r, -2, -1))
  
def get_off_diagonal_elements(M: Tensor) -> Tensor:
  """
  returns a clone of the input Tensor with its diagonal elements zeroed.
  """
  res = M.clone()
  res.diagonal(dim1=-2, dim2=-1).zero_() #diag=0?
  return res
  
def get_Xi_diag(M: Tensor, R: Tensor) -> Tensor:
  """
  A specific use case function which takes the diagonal of M and the rho matrix
  and does,

  Xi_ii = sum_{j != i} M_jj \prod_{k != ij} \sigma_k

  over a batch of matrices.
  """

  diag_M = torch.diagonal(M, offset=0, dim1=-2, dim2=-1).unsqueeze(-2)
  idx=[1] * len(M.shape) #move to relative dims (or not?)
  idx[-2]=M.shape[-1] #defines [1, 1, N, 1]
  diag_M_repeat = diag_M.repeat(*idx) #vmap with tile/len statement? (repeat_like)
  MR = diag_M_repeat*R
  return get_off_diagonal_elements(MR).sum(dim=-1).diag_embed()
  
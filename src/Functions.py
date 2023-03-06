import torch
import torch.nn as nn
from torch.autograd import Function

from torch import Tensor
from typing import Any, Tuple

from Functions_utils import get_log_gamma, get_log_rho
from Functions_utils import get_off_diagonal_elements, get_Xi_diag

from utils import unsqueeze_to_size

#We use the old-style of torch.autograd.Function 

class GeneralisedLogSumExpEnvLogDomainStable(Function):

  generate_vmap_rule = True

  @staticmethod
  def forward(matrices: Tensor, log_envs: Tensor) -> Tuple[Tensor, Tensor]:
    """The forward call (or evaluation) of the custom `torch.autograd.Function` which computes a Signed-Logabs value for a LogSumExp function (with max substraction)

    :param matrices: A Tensor representing the set of D matrcies from the last Equivariant Layer
    :type matrices: class: `torch.Tensor`

    :param log_envs: A Tensor representing the set of D matrcies from the Log-Envelope function
    :type log_envs: class: `torch.Tensor`

    :return out: A Tuple containing the global sign and global logabs value of the Signed-Log Summed Determinant function
    :type out: `Tuple[torch.Tensor, torch.Tensor]`
    """
    sgns, logabss = torch.slogdet(matrices * torch.exp(log_envs))

    max_logabs_envs = torch.max(logabss, keepdim=True, dim=-1)[0] #grab the value only (no indices)

    scaled_dets = sgns*torch.exp( logabss - max_logabs_envs )  #max subtraction

    summed_scaled_dets = scaled_dets.sum(keepdim=True, dim=-1) #sum

    global_logabs = (max_logabs_envs + (summed_scaled_dets).abs().log()).squeeze(-1) #add back in max value and take logabs

    global_sgn = summed_scaled_dets.sign().squeeze(-1) #take sign

    #ctx.mark_non_differentiable(global_sgn)    #mark sgn non-differientable?
    #ctx.save_for_backward(matrices, log_envs, global_sgn, global_logabs)
    return global_sgn, global_logabs, matrices, log_envs

  @staticmethod
  def setup_context(ctx, inputs, output):
    matrices, log_envs = inputs
    global_sgn, global_logabs, matrices, log_envs = output
    ctx.save_for_backward(global_sgn, global_logabs, matrices, log_envs)

  @staticmethod
  def backward(ctx: Any, grad_global_sgn: Tensor, grad_global_logabs: Tensor, grad_matrices: Tensor, grad_log_envs: Tensor) -> Tuple[Tensor, Tensor]:
    global_sgn, global_logabs, matrices, log_envs = ctx.saved_tensors
    
    dLoss_dA, dLoss_dS, U, S, VT, matrices, log_envs, detU, detVT, normed_G, sgn_prefactor, \
    U_normed_G_VT_exp_log_envs, grad_global_logabs,  global_sgn, global_logabs, \
    = GeneralisedLogSumExpEnvLogDomainStableBackward.apply(matrices, log_envs, global_sgn, global_logabs, grad_global_sgn, grad_global_logabs)

    return dLoss_dA, dLoss_dS

class GeneralisedLogSumExpEnvLogDomainStableBackward(Function):

  generate_vmap_rule = True

  @staticmethod
  def forward(matrices: Tensor, log_envs: Tensor, global_sgn: Tensor, global_logabs: Tensor, grad_global_sgn: Tensor, grad_global_logabs: Tensor) -> Tuple[Tensor, Tensor]:

    U, S, VT = torch.linalg.svd(matrices * torch.exp(log_envs))  #all shape [B, D, A, A]
    detU, detVT = torch.linalg.det(U), torch.linalg.det(VT)      #both shape [B,D]
    log_G = get_log_gamma(S.log())                               #shape [B, D, A] (just the diagonal)

    #print("log_G/global_logabs/globl_log unsq: ",log_G.shape, print(global_logabs.shape), unsqueeze_to_size(global_logabs, len(log_G.shape)).shape)
    #normed_G = torch.exp( log_G - global_logabs[:,None,None] )   #shape [B,D,A]
    normed_G = torch.exp( log_G - global_logabs[...,None,None] ) #without batch? #potential broadcast ERROR

    U_normed_G_VT = U @ normed_G.diag_embed() @ VT
    U_normed_G_VT_exp_log_envs = torch.exp(log_envs)*U_normed_G_VT

    #sgn_prefactor = ((grad_global_logabs * global_sgn)[:,None] * detU * detVT)[:,:,None,None] 
    #print("logabs/sgn/U/VT: ",grad_global_logabs.shape, global_sgn.shape, detU.shape, detVT.shape)
    sgn_prefactor = ((grad_global_logabs * global_sgn)[...,None] * detU * detVT)[...,None,None]
    #sgn_prefactor = unsqueeze_to_size(m=(unsqueeze_to_size( m=(grad_global_logabs * global_sgn), s=len(detU.shape)) * detU * detVT), s=len(matrices.shape))
    
    #without batch? #potential broadcast ERROR
    #print(sgn_prefactor.shape, U_normed_G_VT_exp_log_envs.shape)
    dLoss_dA = sgn_prefactor * U_normed_G_VT_exp_log_envs
    dLoss_dS = matrices * dLoss_dA

    #ctx.save_for_backward(U, S, VT, matrices, log_envs, detU, detVT, normed_G, sgn_prefactor, \
    #                      U_normed_G_VT_exp_log_envs, grad_global_logabs,  global_sgn, global_logabs)

    return dLoss_dA, dLoss_dS, U, S, VT, matrices, log_envs, detU, detVT, normed_G, sgn_prefactor, \
                          U_normed_G_VT_exp_log_envs, grad_global_logabs,  global_sgn, global_logabs

  @staticmethod
  def setup_context(ctx: Any, inputs, output):
    matrices, log_envs, global_sgn, global_logabs, grad_global_sgn, grad_global_logabs = inputs
    dLoss_dA, dLoss_dS, U, S, VT, matrices, log_envs, detU, detVT, normed_G, sgn_prefactor, \
                          U_normed_G_VT_exp_log_envs, grad_global_logabs,  global_sgn, global_logabs = output
    ctx.save_for_backward(U, S, VT, matrices, log_envs, detU, detVT, normed_G, sgn_prefactor, \
                          U_normed_G_VT_exp_log_envs, grad_global_logabs,  global_sgn, global_logabs)

  @staticmethod
  def backward(ctx: Any, grad_G: Tensor, grad_H: Tensor, grad_U: Tensor, grad_S: Tensor, grad_VT: Tensor, grad_matrices: Tensor, grad_log_envs: Tensor,\
               grad_detU: Tensor, grad_detVT: Tensor, grad_normed_G: Tensor, grad_sgn_prefactor: Tensor, grad_U_normed_G_VT_exp_log_envs: Tensor, \
               grad_grad_global_logabs: Tensor,  grad_global_sgn: Tensor, grad_global_logabs: Tensor             
               ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    U, S, VT, matrices, log_envs, detU, detVT, normed_G, sgn_prefactor, \
    U_normed_G_VT_exp_log_envs, grad_global_logabs, global_sgn, global_logabs = ctx.saved_tensors #get cached Tensors

    log_envs_max = torch.max(torch.max(log_envs, keepdim=True, dim=-2)[0], keepdim=True, dim=-1)[0]
    grad_K = (grad_G + grad_H * matrices) * torch.exp(log_envs - log_envs_max)
    M = VT @ grad_K.transpose(-2,-1) @ U

    #Calculate normed rho matrices
    log_rho = get_log_rho(S.log())
    #normed_rho = torch.exp( log_rho - global_logabs[:,None,None,None] + log_envs_max)
    #print("log_rho/global_log/global_log unsq/log_envs_max): ",log_rho.shape, global_logabs.shape, unsqueeze_to_size(global_logabs, s=len(log_rho.shape)).shape, log_envs_max.shape)
    #normed_rho = torch.exp( log_rho - unsqueeze_to_size(global_logabs, s=len(log_rho.shape)) + log_envs_max) #without batch? #potential broadcast ERROR
    normed_rho = torch.exp( log_rho - global_logabs[...,None,None,None] + log_envs_max)

    #Calculate normed Xi matrices
    Xi_diag = get_Xi_diag(M, normed_rho)
    Xi_off_diag = get_off_diagonal_elements(-M*normed_rho)
    normed_Xi = Xi_diag + Xi_off_diag #perhaps 1 operation?

    #calculate c constant; sum(dim=(-2,-1)) is summing over kl or mn ; sum(..., dim=-1) is summing over determinants
    c = global_sgn * torch.sum( detU * detVT * torch.sum( (grad_G + matrices * grad_H ) * U_normed_G_VT_exp_log_envs, dim=(-2,-1) ), dim=-1)

    #normed_Xi_minus_c_normed_G = (normed_Xi - c[:,None,None,None]*normed_G.diag_embed()) 
    #print("c/norm XI: ",c.shape, normed_Xi.shape)
    #normed_Xi_minus_c_normed_G = (normed_Xi - unsqueeze_to_size(m=c, s=len(normed_Xi.shape))*normed_G.diag_embed()) #without batch? #potential broadcast ERROR
    #print("c: ",c.shape, c[...,None,None,None].shape)
    normed_Xi_minus_c_normed_G = (normed_Xi - c[...,None,None,None]*normed_G.diag_embed()) 

    U_Xi_c_G_VT =  U @ normed_Xi_minus_c_normed_G @ VT
    U_Xi_c_G_VT_exp_log_envs = U_Xi_c_G_VT * torch.exp(log_envs)

    dF_dA = sgn_prefactor * (U_Xi_c_G_VT_exp_log_envs + grad_H * U_normed_G_VT_exp_log_envs)

    dF_dS = sgn_prefactor * (matrices * U_Xi_c_G_VT_exp_log_envs + (grad_G + grad_H * matrices)*U_normed_G_VT_exp_log_envs)

    dF_dsgn_Psi = None
    dF_dlogabs_Psi = None

    dF_dgrad_sgn_Psi = None
    dF_dgrad_logabs_Psi = c

    return dF_dA, dF_dS, dF_dsgn_Psi, dF_dlogabs_Psi, dF_dgrad_sgn_Psi, dF_dgrad_logabs_Psi


#we define a naive version of the above function here for easier comparision.
def NaiveLogSumExpEnvLogDomainStable(matrices: Tensor, log_envs: Tensor) -> Tuple[Tensor, Tensor]:
  sgns, logabss = torch.slogdet(matrices * torch.exp(log_envs))

  max_logabs_envs = torch.max(logabss, keepdim=True, dim=-1)[0] #grab the value only (no indices)

  scaled_dets = sgns*torch.exp( logabss - max_logabs_envs )  #max subtraction

  summed_scaled_dets = scaled_dets.sum(keepdim=True, dim=-1) #sum

  global_logabs = (max_logabs_envs + (summed_scaled_dets).abs().log()).squeeze(-1) #add back in max value and take logabs

  global_sgn = summed_scaled_dets.sign().squeeze(-1) #take sign

  return global_sgn, global_logabs
import torch
_ = torch.manual_seed(0)    #fix psuedorandom seed
from torch import nn, Tensor
from torch.autograd import Function

from functorch import make_functional, vmap, jacrev
from typing import Tuple, Any
from copy import deepcopy

#######################################################################################################
# util functions for our custom function 
#######################################################################################################

def print_shape(x):
  if(x is None):
    return None
  else:
    return x.shape
  
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

#######################################################################################################
# Our custom function, which is essentially a sum of multiple determinants 
# (with some numerical stability tricks)
#######################################################################################################

#Our custom function!
class custom_function(Function): #GeneralisedLogSumExpEnvLogDomainStable

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
    
    ctx.mark_non_differentiable(global_sgn)
    ctx.save_for_backward(global_sgn, global_logabs, matrices, log_envs)

  @staticmethod
  def backward(ctx: Any, grad_global_sgn: Tensor, grad_global_logabs: Tensor, grad_matrices: Tensor, grad_log_envs: Tensor) -> Tuple[Tensor, Tensor]:
    global_sgn, global_logabs, matrices, log_envs = ctx.saved_tensors
    
    dLoss_dA, dLoss_dS, U, S, VT, matrices, log_envs, detU, detVT, normed_G, sgn_prefactor, \
    U_normed_G_VT_exp_log_envs, grad_global_logabs,  global_sgn, global_logabs, \
    = custom_functionBackward.apply(matrices, log_envs, global_sgn, global_logabs, grad_global_sgn, grad_global_logabs)

    return dLoss_dA, dLoss_dS
  
  """
  @staticmethod
  def vmap(info, in_dims, matrices: Tensor, log_envs: Tensor):
    print("Backward: ",info, in_dims) #VmapInfo(batch_size=4096, randomness='error') (0, 0)
    matrices_bdim, log_envs_bdim = in_dims

    matrices = matrices.movedim(matrices_bdim, 0)
    log_envs = log_envs.movedim(log_envs_bdim, 0)

    #https://pytorch.org/docs/master/notes/extending.func.html 
    #result is re-calc'd in tutorial (not sure why)
    result = custom_function.apply(matrices, log_envs)

    return result, (0, 0, 0, 0)
  """
  

class custom_functionBackward(Function):

  #generate_vmap_rule = True

  @staticmethod
  def forward(matrices: Tensor, log_envs: Tensor, global_sgn: Tensor, global_logabs: Tensor, grad_global_sgn: Tensor, grad_global_logabs: Tensor) -> Tuple[Tensor, Tensor]:

    U, S, VT = torch.linalg.svd(matrices * torch.exp(log_envs))  #all shape [B, D, A, A]
    detU, detVT = torch.linalg.det(U), torch.linalg.det(VT)      #both shape [B,D]
    log_G = get_log_gamma(S.log())                               #shape [B, D, A] (just the diagonal)

    normed_G = torch.exp( log_G - global_logabs.unsqueeze(-1).unsqueeze(-1) )   #shape [B,D,A]

    U_normed_G_VT = U @ normed_G.diag_embed() @ VT
    U_normed_G_VT_exp_log_envs = torch.exp(log_envs)*U_normed_G_VT

    sgn_prefactor = ((grad_global_logabs * global_sgn.unsqueeze(-1)) * detU * detVT)#[:,:,None,None]


    dLoss_dA = sgn_prefactor.unsqueeze(-1).unsqueeze(-1) * U_normed_G_VT_exp_log_envs
    dLoss_dS = matrices * dLoss_dA

    return dLoss_dA, dLoss_dS, U, S, VT, matrices, log_envs, detU, detVT, normed_G, sgn_prefactor, \
                          U_normed_G_VT_exp_log_envs, grad_global_logabs,  global_sgn, global_logabs

  @staticmethod
  def setup_context(ctx: Any, inputs, output):
    matrices, log_envs, global_sgn, global_logabs, grad_global_sgn, grad_global_logabs = inputs

    dLoss_dA, dLoss_dS, U, S, VT, matrices, log_envs, detU, detVT, normed_G, sgn_prefactor, \
                          U_normed_G_VT_exp_log_envs, grad_global_logabs,  global_sgn, global_logabs = output
    
    ctx.mark_non_differentiable(grad_global_logabs, global_sgn)
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
    normed_rho = torch.exp( log_rho - global_logabs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + log_envs_max)

    #Calculate normed Xi matrices
    Xi_diag = get_Xi_diag(M, normed_rho)
    Xi_off_diag = get_off_diagonal_elements(-M*normed_rho)
    normed_Xi = Xi_diag + Xi_off_diag #perhaps 1 operation?

    #calculate c constant; sum(dim=(-2,-1)) is summing over kl or mn ; sum(..., dim=-1) is summing over determinants
    c = global_sgn * torch.sum( detU * detVT * torch.sum( (grad_G + matrices * grad_H ) * U_normed_G_VT_exp_log_envs, dim=(-2,-1) ), dim=-1)
  
    normed_Xi_minus_c_normed_G = (normed_Xi - c.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*normed_G.diag_embed())  

    U_Xi_c_G_VT =  U @ normed_Xi_minus_c_normed_G @ VT
    U_Xi_c_G_VT_exp_log_envs = U_Xi_c_G_VT * torch.exp(log_envs)

    dF_dA = sgn_prefactor.unsqueeze(-1).unsqueeze(-1) * (U_Xi_c_G_VT_exp_log_envs + grad_H * U_normed_G_VT_exp_log_envs)

    dF_dS = sgn_prefactor.unsqueeze(-1).unsqueeze(-1) * (matrices * U_Xi_c_G_VT_exp_log_envs + (grad_G + grad_H * matrices)*U_normed_G_VT_exp_log_envs)

    dF_dsgn_Psi = None
    dF_dlogabs_Psi = None

    dF_dgrad_sgn_Psi = None
    dF_dgrad_logabs_Psi = c

    print(dF_dA.shape, dF_dS.shape, dF_dgrad_logabs_Psi.shape)
    return dF_dA, dF_dS, dF_dsgn_Psi, dF_dlogabs_Psi, dF_dgrad_sgn_Psi, dF_dgrad_logabs_Psi

def pytorch_function(matrices: Tensor, log_envs: Tensor) -> Tuple[Tensor, Tensor]:
  sgns, logabss = torch.slogdet(matrices * torch.exp(log_envs))

  max_logabs_envs = torch.max(logabss, keepdim=True, dim=-1)[0] #grab the value only (no indices)

  scaled_dets = sgns*torch.exp( logabss - max_logabs_envs )  #max subtraction

  summed_scaled_dets = scaled_dets.sum(keepdim=True, dim=-1) #sum

  global_logabs = (max_logabs_envs + (summed_scaled_dets).abs().log()).squeeze(-1) #add back in max value and take logabs

  global_sgn = summed_scaled_dets.sign().squeeze(-1) #take sign

  return global_sgn, global_logabs

#######################################################################################################
# Our base model class, which is an antisymmetric function w.r.t input permutations 
#######################################################################################################

class model(nn.Module):

  def __init__(self, num_input, num_hidden):
    super(model, self).__init__()
    
    self.num_input = num_input
    self.num_hidden = num_hidden
    
    self.layer1 = nn.Linear(2, num_hidden)
    self.layer2 = nn.Linear(num_hidden, num_input)
    
    self.log_env = nn.Linear(1, self.num_input)
    
  def summed_dets(self, matrices, log_envs):
     pass

  def forward(self, x):
    x = x.unsqueeze(-1)

    g = x.mean(dim=0, keepdim=True).repeat(self.num_input, 1)
    f = torch.cat((x,g), dim=1)
    
    y1 = self.layer1(f)
    y2 = self.layer2(y1)

    log_e = self.log_env(x)

    matrices = y2.unsqueeze(-3)
    log_envs = log_e.unsqueeze(-3)

    sgn, logabs = self.summed_dets(matrices, log_envs)
    return sgn, logabs    

#pytorch function vs our custom function
def pytorch_func(matrices, log_envs):
   sgn, logabs = pytorch_function(matrices=matrices, log_envs=log_envs)
   return sgn, logabs

def custom_func(matrices, log_envs):
   sgn, logabs, _, _ = custom_function.apply(matrices, log_envs)
   return sgn, logabs

#######################################################################################################
# Here's the start of the program. We define a simple model, monkey-patch our pytorch and custom functions
# in to the network (used deepcopy to ensure they're identical, same parameters etc.)
#
# GOAL: network must have same output, first derivative w.r.t input, and second derivative w.r.t input
#######################################################################################################

pytorch_net = model(2, 32)

custom_net = deepcopy(pytorch_net) #model(2,32)

pytorch_net.summed_dets = pytorch_func  #2 networks (same weight, different function)
custom_net.summed_dets = custom_func

x = torch.randn(4096, 2, requires_grad=True)

fnet, params = make_functional(pytorch_net)

def calc_logabs(params: Tuple[Tensor], x: Tensor) -> Tuple[Tensor]:
    _, logabs = fnet(params, x)
    return logabs

def dlogabs_dx(params: Tuple[Tensor], x: Tensor) -> Tuple[Tensor]:
    grad_logabs = jacrev(calc_logabs, argnums=1, has_aux=False)(params, x)
    return grad_logabs, grad_logabs

def d2logabs_dx2(params: Tuple[Tensor], x: Tensor) -> Tuple[Tensor]:
    grad2_logabs, grad_logabs = jacrev(dlogabs_dx, argnums=1, has_aux=True)(params, x)
    #return -0.5*(grad2_logabs.diagonal(0,-2,-1).sum() + grad_logabs.pow(2).sum())
    return grad2_logabs.diagonal(0,-2,-1)

pytorch_logabs = vmap(calc_logabs, in_dims=(None, 0))(params, x)
pytorch_grad_logabs, _ = vmap(dlogabs_dx, in_dims=(None, 0))(params, x)
pytorch_grad2_logabs = vmap(d2logabs_dx2, in_dims=(None, 0))(params, x)

fnet, params = make_functional(custom_net) #now repeat with custom function

def calc_logabs(params: Tuple[Tensor], x: Tensor) -> Tuple[Tensor]:
    _, logabs = fnet(params, x)
    return logabs

def dlogabs_dx(params: Tuple[Tensor], x: Tensor) -> Tuple[Tensor]:
    grad_logabs = jacrev(calc_logabs, argnums=1, has_aux=False)(params, x)
    return grad_logabs, grad_logabs

def d2logabs_dx2(params: Tuple[Tensor], x: Tensor) -> Tuple[Tensor]:
    grad2_logabs, grad_logabs = jacrev(dlogabs_dx, argnums=1, has_aux=True)(params, x)
    #return -0.5*(grad2_logabs.diagonal(0,-2,-1).sum() + grad_logabs.pow(2).sum())
    return grad2_logabs.diagonal(0,-2,-1)

custom_logabs = vmap(calc_logabs, in_dims=(None, 0))(params, x)
custom_grad_logabs, _ = vmap(dlogabs_dx, in_dims=(None, 0))(params, x)
custom_grad2_logabs = vmap(d2logabs_dx2, in_dims=(None, 0))(params, x)

print("Check")
print("logabs: ",torch.allclose(pytorch_logabs, custom_logabs))              #Output matches
print("grad:   ",torch.allclose(pytorch_grad_logabs, custom_grad_logabs))    #1st derivative fails, although small errors
print("grad2:  ",torch.allclose(pytorch_grad2_logabs,custom_grad2_logabs))   #2nd derivative fails, huge errors

"""
returns:
      Check
      logabs:  True
      grad:    False
      grad2:   False
"""

print(pytorch_logabs-custom_logabs)
print(pytorch_grad_logabs-custom_grad_logabs)
print(pytorch_grad2_logabs-custom_grad2_logabs)


#######################################################################################################
# Now compare the custom function with pytorch's grad-check and grad-grad-check, which somehow equals
#######################################################################################################

nsamples=1     #gradchecks only support batch_size=1
num_dets=1     #1 torch.linalg.slogdet
num_inputs=2

matrices = torch.randn(nsamples, num_dets, num_inputs, num_inputs, dtype=torch.float64, requires_grad=True)
log_envs = torch.randn(nsamples, num_dets, num_inputs, num_inputs, dtype=torch.float64, requires_grad=True) 

custom_out = pytorch_func(matrices, log_envs)
naive_out = custom_func(matrices, log_envs)

sign_forward_check = torch.allclose(naive_out[0], custom_out[0])
logabs_forward_check = torch.allclose(naive_out[1], custom_out[1])

backward_check = torch.autograd.gradcheck(func=custom_func, inputs=(matrices, log_envs), raise_exception=False)

double_backward_check = torch.autograd.gradgradcheck(func=custom_func, inputs=(matrices, log_envs), raise_exception=False)

print("forward check (sign): ", sign_forward_check)
print("forward check (logabs): ",logabs_forward_check)
print("gradcheck: ",backward_check)
print("gradgradcheck: ",double_backward_check)

"""
returns:
    forward check (sign):  True
    forward check (logabs):  True
    gradcheck:  True
    gradgradcheck:  True
"""
import torch
import torch.nn as nn
from torch.autograd import Function
from torch import Tensor

torch.manual_seed(0)
torch.set_default_dtype(torch.float64) #need float64 for gradgradcheck (even though code is float32)

from Functions import GeneralisedLogSumExpEnvLogDomainStable  #Custom
from Functions import NaiveLogSumExpEnvLogDomainStable        #Standard

def naive_summed_det(A: Tensor, log_envs: Tensor) -> Tensor:
  global_sgn, global_logabs = NaiveLogSumExpEnvLogDomainStable(A, log_envs)
  return global_sgn, global_logabs

def custom_summed_det(A: Tensor, log_envs: Tensor) -> Tensor:
  global_sgn, global_logabs = GeneralisedLogSumExpEnvLogDomainStable.apply(A, log_envs)
  return global_sgn, global_logabs

nsamples = 1 #can only work for 1 sample (for gradgrad checks)
num_dets = 1
num_inputs = 6

A = torch.randn(nsamples, num_dets, num_inputs, num_inputs, requires_grad=True)
log_envs = torch.rand(nsamples, num_dets, num_inputs, num_inputs, requires_grad=True)

custom_out = custom_summed_det(A, log_envs)
naive_out = naive_summed_det(A, log_envs)

sign_forward_check = torch.allclose(naive_out[0], custom_out[0])
logabs_forward_check = torch.allclose(naive_out[1], custom_out[1])

backward_check = torch.autograd.gradcheck(func=custom_summed_det, inputs=(A, log_envs), raise_exception=False)

double_backward_check = torch.autograd.gradgradcheck(func=custom_summed_det, inputs=(A, log_envs), raise_exception=False)


print("forward check (sign): ", sign_forward_check)
print("forward check (logabs): ",logabs_forward_check)
print("gradcheck: ",backward_check)
print("gradgradcheck: ",double_backward_check)

def naive_summed_det(A: Tensor, log_envs: Tensor) -> Tensor:
  global_sgn, global_logabs = NaiveLogSumExpEnvLogDomainStable(A, log_envs)
  return global_logabs

def custom_summed_det(A: Tensor, log_envs: Tensor) -> Tensor:
  global_sgn, global_logabs = GeneralisedLogSumExpEnvLogDomainStable.apply(A, log_envs)
  return global_logabs

naive_jac = torch.autograd.functional.jacobian(func=naive_summed_det, inputs=(A, log_envs))
naive_hess = torch.autograd.functional.hessian(func=naive_summed_det, inputs=(A, log_envs))

custom_jac = torch.autograd.functional.jacobian(func=custom_summed_det, inputs=(A, log_envs))
custom_hess = torch.autograd.functional.hessian(func=custom_summed_det, inputs=(A, log_envs))

print("\nCheck custom function vs naive function")
print("For larger matrices, in D or N, the difference will become more apparent\n")

for i in range(2):
  print("Jacobian check: ", i, torch.allclose(naive_jac[i], custom_jac[i]))

for i in range(2):
  for j in range(2):
    print("Hessian check: ",i,j,torch.allclose(naive_hess[i][j], custom_hess[i][j]))    
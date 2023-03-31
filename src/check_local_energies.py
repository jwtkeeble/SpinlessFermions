import torch
torch.manual_seed(0)
from torch import nn

from Models import vLogHarmonicNet 
from Samplers import MetropolisHastings
from Hamiltonian import HarmonicOscillatorWithInteraction1D as HOw1D

from functorch import vmap, make_functional
from functorch.experimental import chunk_vmap as xmap

import argparse

parser = argparse.ArgumentParser(description='benis')
#https://stackoverflow.com/questions/14117415/in-python-using-argparse-allow-only-positive-integers/14117567

parser.add_argument("-N", "--num_fermions", type=int,   default=2,     help="Number of fermions in physical system")
parser.add_argument("-H", "--num_hidden",   type=int,   default=32,    help="Number of hidden neurons per layer")
parser.add_argument("-L", "--num_layers",   type=int,   default=1,     help="Number of layers within the network")
parser.add_argument("-D", "--num_dets",     type=int,   default=1,     help="Number of determinants within the network's final layer")
parser.add_argument("-V", "--V0",           type=float, default=0.,    help="Interaction strength (in harmonic units)")
parser.add_argument("-S", "--sigma0",       type=float, default=0.5,   help="Interaction distance (in harmonic units")
parser.add_argument("--preepochs",          type=int,   default=10000, help="Number of pre-epochs for the pretraining phase")
parser.add_argument("--epochs",             type=int,   default=10000, help="Number of epochs for the energy minimisation phase")
parser.add_argument("-C", "--chunks",       type=int,   default=1,     help="Number of chunks for vectorized operations")

args = parser.parse_args()

nfermions = args.num_fermions #number of input nodes
num_hidden = args.num_hidden  #number of hidden nodes per layer
num_layers = args.num_layers  #number of layers in network
num_dets = args.num_dets      #number of determinants (accepts arb. value)
func = nn.Tanh()  #activation function between layers
pretrain = False   #pretraining output shape?

nwalkers=4096
n_sweeps=10 #n_discard
std=1.#0.02#1.
target_acceptance=0.5

V0 = args.V0
sigma0 = args.sigma0

pt_save_every_ith=500
em_save_every_ith=500
nchunks=1

preepochs=args.preepochs
epochs=args.epochs

gs=nfermions**2/2.

device='cuda'

net = vLogHarmonicNet(num_input=nfermions,
                      num_hidden=num_hidden,
                      num_layers=num_layers,
                      num_dets=num_dets,
                      func=func,
                      pretrain=pretrain)
net=net.to(device)
#[print(name, p) for name, p in net.named_parameters()]
sampler = MetropolisHastings(network=net, dof=nfermions, nwalkers=nwalkers, target_acceptance=target_acceptance)
#try diff. samplers? enforce accept after nth samples?

net.pretrain=False
fnet, params = make_functional(net)

calc_elocal = HOw1D(fnet=fnet, V0=V0, sigma0=sigma0, nchunks=nchunks)

optim = torch.optim.Adam(params=net.parameters(), lr=1e-4)

x, _ = sampler(n_sweeps)

#elocal = xmap(calc_elocal, in_dims=(None, 0), chunks=nchunks)(params, x)

#print("elocal: ",elocal)

from torch import Tensor
from typing import Tuple
from functorch import jacrev

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

#merge into a single lambda call so it can be grad/vmap'd in one go?
kinetic_from_log = lambda params, x: d2logabs_dx2(params, x)
print("logabs calc")
logabs = vmap(calc_logabs, in_dims=(None, 0))(params, x)
print("grad_logabs calc")
grad_logabs, _ = vmap(dlogabs_dx, in_dims=(None, 0))(params, x)
print("grad2_logabs calc")
grad2_logabs = vmap(d2logabs_dx2, in_dims=(None, 0))(params, x)

print(logabs)
print(grad_logabs)
print(grad2_logabs)
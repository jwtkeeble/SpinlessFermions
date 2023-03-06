import torch
from torch import nn, Tensor
from functorch import make_functional
from functorch.experimental import chunk_vmap as xmap
import os, sys, time

torch.manual_seed(0)
torch.set_printoptions(4)
torch.backends.cudnn.benchmark=True
torch.set_default_dtype(torch.float32)

device = torch.device('cuda')
dtype = str(torch.get_default_dtype()).split('.')[-1]

sys.path.append("./src/")

from Models import vLogHarmonicNet
from Samplers import MetropolisHastings
from Hamiltonian import HarmonicOscillatorWithInteraction1D as HOw1D
from Pretraining import HermitePolynomialMatrix 

from utils import load_dataframe, load_model, count_parameters, get_groundstate
from utils import get_params, sync_time, clip, calc_pretraining_loss

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import argparse

parser = argparse.ArgumentParser(prog="SpinlessFermions",
                                 usage='%(prog)s [options]',
                                 description="A Neural Quantum State (NQS) solution to one-dimensional fermions interacting in a Harmonic trap",
                                 epilog="and fin")

parser.add_argument("-N", "--num_fermions", type=int,   default=2,     help="Number of fermions in physical system")
parser.add_argument("-H", "--num_hidden",   type=int,   default=64,    help="Number of hidden neurons per layer")
parser.add_argument("-L", "--num_layers",   type=int,   default=2,     help="Number of layers within the network")
parser.add_argument("-D", "--num_dets",     type=int,   default=1,     help="Number of determinants within the network's final layer")
parser.add_argument("-V", "--V0",           type=float, default=0.,    help="Interaction strength (in harmonic units)")
parser.add_argument("-S", "--sigma0",       type=float, default=0.5,   help="Interaction distance (in harmonic units")
parser.add_argument("--preepochs",          type=int,   default=10000, help="Number of pre-epochs for the pretraining phase")
parser.add_argument("--epochs",             type=int,   default=10000, help="Number of epochs for the energy minimisation phase")
parser.add_argument("-C", "--chunks",       type=int,   default=1,     help="Number of chunks for vectorized operations")

parser.add_argument("-B","--num_batches",   type=int,   default=10000, help="Number of batches of samples (effectively the length of the chain)")
parser.add_argument("-W","--num_walkers",   type=int,   default=4096,  help="Number of walkers used to generate configuration")
parser.add_argument("--num_sweeps",         type=int,   default=10,    help="Number of sweeped/discard proposed configurations between accepted batches (The equivalent of thinning constant)")

args = parser.parse_args()

nfermions = args.num_fermions #number of input nodes
num_hidden = args.num_hidden  #number of hidden nodes per layer
num_layers = args.num_layers  #number of layers in network
num_dets = args.num_dets      #number of determinants (accepts arb. value)
func = nn.Tanh()  #activation function between layers
pretrain = True   #pretraining output shape?

nbatches = args.num_batches
nwalkers=args.num_walkers
n_sweeps=args.num_sweeps #n_discard
std=1.#0.02#1.
target_acceptance=0.5

V0 = args.V0
sigma0 = args.sigma0

pt_save_every_ith=1000
em_save_every_ith=1000

nchunks=1

preepochs=args.preepochs
epochs=args.epochs

net = vLogHarmonicNet(num_input=nfermions,
                      num_hidden=num_hidden,
                      num_layers=num_layers,
                      num_dets=num_dets,
                      func=func,
                      pretrain=pretrain)
net=net.to(device)

sampler = MetropolisHastings(network=net,
                             dof=nfermions,
                             nwalkers=nwalkers,
                             target_acceptance=target_acceptance)

###############################################################################################################################################
#####                                             GENERATE MANY-BODY DATA                                                                 #####
###############################################################################################################################################

net.pretrain = False #check it's false
optim = torch.optim.Adam(params=net.parameters(), lr=1e-4) #new optimizer

model_path = "results/energy/checkpoints/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s_chkp.pt" % \
                (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, \
                 optim, False, device, dtype)
filename = "results/energy/data/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s.csv" % \
                (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, \
                 optim, False, device, dtype)

writer = load_dataframe(filename)
output_dict = load_model(model_path=model_path, device=device, net=net, optim=optim, sampler=sampler)

configurations = torch.zeros(size=(nbatches, nwalkers, nfermions), dtype=torch.get_default_dtype(), device='cpu') #store on CPU 

xmin=-6
xmax=+6

xdata = torch.zeros([nbatches, nwalkers])
ydata = torch.zeros([nbatches, nwalkers])
zdata = torch.zeros([nbatches, nwalkers])

with torch.no_grad():
    for batch in tqdm(range(nbatches), desc="Generating configurations"):
        x, _ = sampler(n_sweeps=n_sweeps)

        s = torch.rand_like(x)*(xmax-xmin) + xmin #uniform rand in range [xmin, xmax)

        configurations[batch, ...] = x.detach() #check clone

        xp = x.clone() #ghost-particle method for computing one-body density matrix
        xp[:,0] = s[:,0]

        sgn_p, logabs_p = net(xp)
        sgn, logabs = net(x)

        xdata[batch, ...] = x[:,0]
        ydata[batch, ...] = xp[:, 0]
        zdata[batch, ...] = ((xmax-xmin) * sgn_p*sgn * torch.exp(logabs_p-logabs))

def to_numpy(x: Tensor) -> np.ndarray:
    x=x.detach().cpu().numpy()
    return x

plt.hist(configurations[...,0])

#counts, bins = np.histogram(a=to_numpy(xdata.flatten()), range=(-5,5), bins=100, density=True)
#plt.stairs(counts, bins)
plt.hist(x=to_numpy(xdata[:,0].flatten()), range=(-5,5), bins=100, density=True)
plt.show()

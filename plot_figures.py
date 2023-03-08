import torch
from torch import nn
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

#nbatches = args.num_batches
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

optim = "Adam"

device='cuda'
dtype='float32'

analysis_datapath = "analysis/PHYS_A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s.npz" % \
                (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, \
                 optim, False, device, dtype)

loaded = np.load(analysis_datapath)

density_xx = loaded['density_xx']
density_psi = loaded['density_psi']

plt.plot(density_xx, density_psi)
plt.show()

h_obdm = loaded['h_obdm']           #rho-matrix (OBDM)
xedges_obdm = loaded['xedges_obdm'] 
yedges_obdm = loaded['yedges_obdm']

cmap=plt.cm.bwr
norm=colors.TwoSlopeNorm(vmin=np.min(h_obdm), vmax=np.max(h_obdm), vcenter=0.)

sc=plt.pcolormesh(xedges_obdm, yedges_obdm, h_obdm, cmap=cmap, norm=norm)
#plt.contour(xedges, yedges, rho_matrix, color='black')
plt.colorbar(sc)
plt.show()

eigenvalues = loaded['eigenvalues']
eigenvectors = loaded['eigenvectors']

xvals = xedges_obdm[:-1] + np.diff(xedges_obdm)/2.

for i in range(nfermions):
    plt.plot(xvals, eigenvectors[:,i], label="%i" % (i))
plt.legend()
plt.show()

h_tbd = loaded['h_tbd']
xedges_tbd = loaded['xedges_tbd']
yedges_tbd = loaded['yedges_tbd']

sc_tbdm=plt.pcolormesh(xedges_tbd, yedges_tbd, h_tbd)
plt.colorbar(sc_tbdm)
plt.show()
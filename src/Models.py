import torch
import torch.nn as nn

from torch import Tensor
from typing import Callable, Tuple

from Layers import EquivariantLayer, SlaterMultiDet, LogEnvelope, MatrixToSLogDeterminant

class vLogHarmonicNet(nn.Module):

    def __init__(self, num_input: int, num_hidden: int, num_layers: int, num_dets: int, func: nn.Module, pretrain: bool):
        """Permutational Equivariant Neural Network which takes the one-dimensional positions
        of the system (represented by a matrix of [A,1]) and returns the log. abs. determinant
        (and its sign) of D Generalised Slater Matrices which are subsequently merged together
        via a Generalised LogSumExp function which return the global logabs values (along with 
        a global sign).
        
        :param num_input: The number of input nodes (representing fermions)
                        for the network to have
        :type num_input: int
        
        :param num_hidden: The number of hidden nodes the network will have per each
                        equivariant layer
        :type num_hidden: int
        
        :param num_layers: The number of hidden equivariant layers within the network
        :type num_layers: int
        
        :param num_dets: The number of Generalised Slater Matrices within the network 
        :type num_dets: int
        
        :param func: The non-linear activation function between each permutation 
                    equivariant layer
        :type func: class: `torch.nn.Module`
        
        :param pretrain: A Flag variable to determine the output of the neural network
                        if True, it returns a set of D Generalised Slater Matrices
                        if False, it returns a global logabs and sign values
        :type pretrain: bool   
        """
        super(vLogHarmonicNet, self).__init__()
        
        self.num_input=num_input
        self.num_hidden=num_hidden
        self.num_layers=num_layers
        self.num_dets=num_dets
        self.func=func
        self.pretrain=pretrain
        
        layers = []
        layers.append(EquivariantLayer(in_features=2,
                                    out_features=self.num_hidden,
                                    num_particles=self.num_input,
                                    bias=True))
        for i in range(1, self.num_layers):
            layers.append(EquivariantLayer(in_features=2*self.num_hidden,
                                            out_features=self.num_hidden,
                                            num_particles=self.num_input,
                                            bias=True)
                    )

        layers.append(SlaterMultiDet(in_features=self.num_hidden, 
                                    num_particles=self.num_input,
                                    num_dets=self.num_dets,
                                    bias=True))

        self.layers = nn.ModuleList(layers)

        self.log_envelope = LogEnvelope(num_particles=self.num_input,
                                        num_dets=self.num_dets)

        self.slog_slater_det = MatrixToSLogDeterminant(num_particles=self.num_input)



    def forward(self, x0: Tensor):
        """The call method of the class (is the equivalent of evaluating the network's current output)
        
        :param x0: The input positions of the A fermions being studied
        :type x0: class: `torch.Tensor`
        
        :return out: returns a tuple whose content depends on class attribute self.pretrain.
                    If `Models.MultiDetLogHarmonicNet.pretrain` is True, the Tuple contains the Generalised Slater Matrices
                    If `Models.MultiDetLogHarmonicNet.pretrain` is False, the Tuple contains the global sign and global logabs values of Generalised Slater Matrices     
        :rtype out: `Tuple[torch.Tensor, torch.Tensor]`
        """

        h=x0.unsqueeze(-1)                #add feature dim (1d-systems only)
        x = self.func(self.layers[0](h))  #equivariant layers here... 
        for l in self.layers[1:-1]:       #(with residual connections)
            x = self.func(l(x)) + x
        matrices = self.layers[-1](x)     #slater multi-det layer
        log_envs = self.log_envelope(x0)  #log-envelopes 

        if(self.pretrain):
            generalised_matrices = matrices * torch.exp(log_envs)
            return generalised_matrices
        else:
            sign, logabsdet = self.slog_slater_det(matrices, log_envs)
            return sign, logabsdet

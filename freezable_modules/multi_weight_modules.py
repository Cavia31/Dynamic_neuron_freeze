import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import grad
from typing import Tuple

class FreezableConv2d(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int,int], stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 padding_mode='zeros', device=None, dtype=None, freezing_matrix=None):
        """
        A convolution layer with separated kernels that can be individually frozen.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (Tuple[int,int]): size of the kernels (MUST be a tuple of 2 ints!)
            stride (int): step between each cross-correlation
            padding (int): size of the padding
            dilation (int): space between each convolution point
            groups (int): number of groups of kernels (must divide in_channels and out_channels)
            bias (bool): use bias or not
            padding_mode (str): controls padding mode (not implemented)
            device (str): device on which to send the parameters
            dtype (type): type of the parameters
            freezing_matrix (Tensor): the matrix controlling wich kernel is frozen or not (defaulting to all kernels unfrozen)
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype

        super().__init__()

        if in_channels%groups > 0:
            raise Exception
        
        if padding_mode != 'zeros':
            raise NotImplementedError

        # Use the provided freezing Matrix, or generate one
        if freezing_matrix is not None:
            self.freezing_matrix = freezing_matrix
        else:
            self.freezing_matrix = torch.full((in_channels//groups,out_channels), True, dtype=bool)
        
        # Generate the kernels (separated so the flag requires_grad can be set up individually)
        self.neurons = nn.ParameterList([nn.Parameter(torch.randn((1,in_channels//groups,kernel_size[0],kernel_size[0])), requires_grad=True) for _ in range(out_channels)])

        # Generate bias (always trained)
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels), requires_grad=True) #[torch.randn(1, requires_grad=True) for _ in range(out_channels)]
        else:
            self.bias = None #[torch.zeros(1) for _ in range(out_channels)]

    def forward(self, x: torch.Tensor):
        outputs = []
        for idx,param in enumerate(self.neurons):
            outputs.append(F.conv2d(x, param, self.bias[idx:idx+1],self.stride,self.padding,self.dilation,1))
        out = torch.cat(outputs,dim=1)
        return out

    def set_freezing_matrix(self, freezing_matrix):
        self.freezing_matrix = freezing_matrix
        self.reset_parameters_frozen_state()

    def reset_parameters_frozen_state(self):
        for i,kernel_list in enumerate(self.kernels):
            for j,kernel in enumerate(kernel_list):
                kernel.requires_grad_(bool(self.freezing_matrix[i,j]))


class FreezableLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, freeze_matrix=None):
        """
        A linear layer with separated neurons that can be individually frozen.

        Args:
            in_features (int): number of input features
            out_features (int): number of output features
            bias (bool): use bias or not
            device (str): device on which to send the parameters
            dtype (type): type of the parameters
            freezing_matrix (Tensor): the matrix controlling wich kernel is frozen or not (defaulting to all kernels unfrozen)
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        if freeze_matrix is not None:
            self.freeze_matrix = freeze_matrix
        else:
            self.freeze_matrix = torch.full((out_features,), True, dtype=bool)
        
        self.neurons = nn.ParameterList([nn.Parameter(torch.randn(in_features), requires_grad=bool(self.freeze_matrix[i])) for i in range(out_features)])
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features), requires_grad=True)

    def forward(self, x: torch.Tensor):
        neuron_tensor = torch.empty((self.out_features,self.in_features), device=self.neurons[0].device)
        for i in range(self.out_features):
            neuron_tensor[i] = self.neurons[i]
        return F.linear(x, neuron_tensor, self.bias)



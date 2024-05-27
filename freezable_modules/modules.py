import torch
import torch.nn as nn
import torch.nn.functional as F
import freezable_modules.bw_functions as bwf
from random import sample

class FreezableConv2d(nn.Conv2d):
    def __init__(self, input_features, output_features, kernel_size1, kernel_size2, bias=True, padding=0, stride=1, dilation=1, groups=1):
        super().__init__(input_features, output_features, kernel_size1, kernel_size2, bias)
        self.input_features = input_features
        self.output_features = output_features
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        # a dict may not be necessary as NEq updates this each epoch... 
        # will investigate once everything works
        self.freeze = list(range(output_features))
        self.weight = nn.Parameter(torch.empty(output_features, input_features//groups, kernel_size1, kernel_size2))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        # Not a very smart way to initialize weights
        self.reset_parameters()

    def forward(self, input):
        return bwf.FreezableConv2dFunction.apply(input, self.weight, self.bias, self.freeze, self.padding, self.stride, self.dilation, self.groups)

    def extra_repr(self):
        return f'input_features={self.input_features}, output_features={self.output_features}, bias={self.bias is not None}'

    def set_freeze(self, freeze:list):
        if freeze is None:
            self.freeze = []
            return
        self.freeze = freeze

    def get_default_freeze(self):
        return list(range(self.output_features))
    
    def get_new_freeze(self, ratio):
        k = max(1, int(self.output_features*ratio))
        return sample(self.get_default_freeze(), k)

    def get_total_parameters(self) -> int:
        s = self.weight.size()
        sb = 0 if self.bias is None else self.bias.size()[0]
        return s[0]*s[1]*s[2]*s[3] + sb
    
    def get_unfrozen_parameters(self) -> int:
        s = self.weight.size()
        sb = 0 if self.bias is None else self.bias.size()[0]
        return len(self.freeze)*s[1]*s[2]*s[3] + sb

class FreezableLinear(nn.Linear):
    def __init__(self, input_features, output_features, bias=True):
        super(FreezableLinear, self).__init__(input_features, output_features, bias)
        self.input_features = input_features
        self.output_features = output_features
        # a dict may not be necessary as NEq updates this each epoch... 
        # will investigate once everything works
        self.freeze = list(range(output_features))
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        # Not a very smart way to initialize weights
        self.reset_parameters()

    def forward(self, input):
        return bwf.FreezableLinearFunction.apply(input, self.weight, self.bias, self.freeze)

    def extra_repr(self):
        return f'input_features={self.input_features}, output_features={self.output_features}, bias={self.bias is not None}' 
    
    def set_freeze(self, freeze:list):
        if freeze is None:
            self.freeze = []
            return
        self.freeze = freeze
    
    def get_default_freeze(self):
        return list(range(self.output_features))

    def get_new_freeze(self, ratio):
        k = max(1, int(self.output_features*ratio))
        return sample(self.get_default_freeze(), k)

    def get_total_parameters(self) -> int:
        s = self.weight.size()
        sb = 0 if self.bias is None else self.bias.size()[0]
        return s[0]*s[1] + sb
    
    def get_unfrozen_parameters(self) -> int:
        s = self.weight.size()
        sb = 0 if self.bias is None else self.bias.size()[0]
        return len(self.freeze)*s[1] + sb


class SequentialF(nn.Sequential):
    def __init__(self, *args: nn.Module):
        super(SequentialF,self).__init__(*args)
    
    def set_freezing_matrix(self, freezing_matrix:dict):
        for module in self.named_children():
            if isinstance(module[1], FreezableConv2d):
                module[1].set_freeze(freezing_matrix.get(module[0]))
            elif isinstance(module[1], FreezableLinear):
                module[1].set_freeze(freezing_matrix.get(module[0]))
            elif isinstance(module[1], FreezableModule):
                module[1].set_freezing_matrix(freezing_matrix.get(module[0]))
            elif isinstance(module[1], SequentialF):
                module[1].set_freezing_matrix(freezing_matrix.get(module[0]))
        
    def get_default_freezing_matrix(self):
        module_freeze_matrix = dict()
        for module in self.named_children():
            if isinstance(module[1], FreezableConv2d):
                module_freeze_matrix[module[0]] = module[1].get_default_freeze()
            elif isinstance(module[1], FreezableLinear):
                module_freeze_matrix[module[0]] = module[1].get_default_freeze()
            elif isinstance(module[1], FreezableModule):
                module_freeze_matrix[module[0]] = module[1].get_default_freezing_matrix()
            elif isinstance(module[1], SequentialF):
                module_freeze_matrix[module[0]] = module[1].get_default_freezing_matrix()
        return module_freeze_matrix
    
    def get_empty_freezing_matrix(self):
        module_freeze_matrix = dict()
        for module in self.named_children():
            if isinstance(module[1], FreezableConv2d):
                module_freeze_matrix[module[0]] = []
            elif isinstance(module[1], FreezableLinear):
                module_freeze_matrix[module[0]] = []
            elif isinstance(module[1], SequentialF):
                module_freeze_matrix[module[0]] = module[1].get_empty_freezing_matrix()
            elif isinstance(module[1], FreezableModule):
                module_freeze_matrix[module[0]] = module[1].get_empty_freezing_matrix()
        return module_freeze_matrix
    
    def get_total_parameters(self):
        total = 0
        for module in self.named_children():
            if isinstance(module[1], FreezableConv2d):
                total += module[1].get_total_parameters()
            elif isinstance(module[1], FreezableLinear):
                total += module[1].get_total_parameters()
            elif isinstance(module[1], FreezableModule):
                total += module[1].get_total_parameters()
            elif isinstance(module[1], SequentialF):
                total += module[1].get_total_parameters()
        return total
    
    def get_unfrozen_parameters(self):
        total = 0
        for module in self.named_children():
            if isinstance(module[1], FreezableConv2d):
                total += module[1].get_unfrozen_parameters()
            elif isinstance(module[1], FreezableLinear):
                total += module[1].get_unfrozen_parameters()
            elif isinstance(module[1], FreezableModule):
                total += module[1].get_unfrozen_parameters()
            elif isinstance(module[1], SequentialF):
                total += module[1].get_unfrozen_parameters()
        return total
    
    def _neuron_list(self, extended_name=""):
        neuron_list = []
        for module in self.named_children():
            if isinstance(module[1],FreezableConv2d):
                for neuron in range(module[1].output_features):
                    neuron_list.append((extended_name+module[0],neuron))
            elif isinstance(module[1],FreezableLinear):
                for neuron in range(module[1].output_features):
                    neuron_list.append((extended_name+module[0],neuron))
            elif isinstance(module[1],FreezableModule):
                neuron_list.extend(module[1]._neuron_list(extended_name+module[0]+"-"))
            elif isinstance(module[1],SequentialF):
                neuron_list.extend(module[1]._neuron_list(extended_name+module[0]+"-"))
        return neuron_list
    
    
class FreezableModule(nn.Module):
    def __init__(self):
        super(FreezableModule,self).__init__()
        self.module_freeze_matrix = dict()
    
    def set_freezing_matrix(self, freezing_matrix:dict):
        self.module_freeze_matrix = freezing_matrix
        for module in self.named_children():
            if isinstance(module[1], FreezableConv2d):
                module[1].set_freeze(self.module_freeze_matrix.get(module[0]))
            elif isinstance(module[1], FreezableLinear):
                module[1].set_freeze(self.module_freeze_matrix.get(module[0]))
            elif isinstance(module[1], SequentialF):
                module[1].set_freezing_matrix(self.module_freeze_matrix.get(module[0]))
            elif isinstance(module[1], FreezableModule):
                module[1].set_freezing_matrix(self.module_freeze_matrix.get(module[0]))
    
    def get_default_freezing_matrix(self):
        module_freeze_matrix = dict()
        for module in self.named_children():
            if isinstance(module[1], FreezableConv2d):
                module_freeze_matrix[module[0]] = module[1].get_default_freeze()
            elif isinstance(module[1], FreezableLinear):
                module_freeze_matrix[module[0]] = module[1].get_default_freeze()
            elif isinstance(module[1], SequentialF):
                module_freeze_matrix[module[0]] = module[1].get_default_freezing_matrix()
            elif isinstance(module[1], FreezableModule):
                module_freeze_matrix[module[0]] = module[1].get_default_freezing_matrix()
        return module_freeze_matrix

    def get_empty_freezing_matrix(self):
        module_freeze_matrix = dict()
        for module in self.named_children():
            if isinstance(module[1], FreezableConv2d):
                module_freeze_matrix[module[0]] = []
            elif isinstance(module[1], FreezableLinear):
                module_freeze_matrix[module[0]] = []
            elif isinstance(module[1], SequentialF):
                module_freeze_matrix[module[0]] = module[1].get_empty_freezing_matrix()
            elif isinstance(module[1], FreezableModule):
                module_freeze_matrix[module[0]] = module[1].get_empty_freezing_matrix()
        return module_freeze_matrix
    
    def get_total_parameters(self):
        total = 0
        for module in self.named_children():
            if isinstance(module[1], FreezableConv2d):
                total += module[1].get_total_parameters()
            elif isinstance(module[1], FreezableLinear):
                total += module[1].get_total_parameters()
            elif isinstance(module[1], SequentialF):
                total += module[1].get_total_parameters()
            elif isinstance(module[1], FreezableModule):
                total += module[1].get_total_parameters()
        return total
    
    def get_unfrozen_parameters(self):
        total = 0
        for module in self.named_children():
            if isinstance(module[1], FreezableConv2d):
                total += module[1].get_unfrozen_parameters()
            elif isinstance(module[1], FreezableLinear):
                total += module[1].get_unfrozen_parameters()
            elif isinstance(module[1], SequentialF):
                total += module[1].get_unfrozen_parameters()
            elif isinstance(module[1], FreezableModule):
                total += module[1].get_unfrozen_parameters()
        return total
    
    def _neuron_list(self, extended_name=""):
        neuron_list = []
        for module in self.named_children():
            if isinstance(module[1],FreezableConv2d):
                for neuron in range(module[1].output_features):
                    neuron_list.append((extended_name+module[0],neuron))
            elif isinstance(module[1],FreezableLinear):
                for neuron in range(module[1].output_features):
                    neuron_list.append((extended_name+module[0],neuron))
            elif isinstance(module[1],FreezableModule):
                neuron_list.extend(module[1]._neuron_list(extended_name+module[0]+"-"))
            elif isinstance(module[1],SequentialF):
                neuron_list.extend(module[1]._neuron_list(extended_name+module[0]+"-"))
        return neuron_list
    
    @staticmethod
    def _build_freezing_matrix(matrix:dict, keys:list, neuron:int, order=True):
        if len(keys) > 1:
            key = keys.pop(0)
            matrix.get
            if matrix.get(key) is None:
                matrix[key] = dict()
                FreezableModule._build_freezing_matrix(matrix[key], keys, neuron)
            else:
                FreezableModule._build_freezing_matrix(matrix[key], keys, neuron)
        elif len(keys) == 1:
            key = keys.pop()
            if matrix.get(key) is None:
                matrix[key] = [neuron]
            else:
                matrix[key].append(neuron)
                if order:
                    matrix[key].sort()
        else:
            return
    
    def random_freezing_matrix(self, ratio, order=True):
        """
        Returns a random freezing matrix to be used to freeze neurons of the module this method is called on.
            Args:
                ratio (float): the ratio of frozen neurons to be set (float between 0. and 1.)
        """
        neuron_list = self._neuron_list()
        n_unfrozen = int(len(neuron_list)*ratio)
        sampled_list = sample(neuron_list, n_unfrozen)
        freezing_matrix = self.get_empty_freezing_matrix()
        for key,neuron in sampled_list:
            key_parts = key.split('-')
            FreezableModule._build_freezing_matrix(freezing_matrix, key_parts, neuron, order=order)            
        return freezing_matrix, n_unfrozen
    
    def n_random_freezing_matrixes(self, ratio, order=True):
        """
        Returns a list of tuples containing a freezing matrix and the number of active neurons for this matrix.
        The matrixes in the list cover all the neurons of the model.
            Args:
                ratio (float): the ratio of frozen neurons to be set (float between 0. and 1.)
        """
        # List all neurons
        neuron_list = self._neuron_list()
        # Get the number of neurons, and the size of a sample with respect to the ratio
        freezable_neurons = len(neuron_list)
        step_size = int(freezable_neurons*ratio)
        freezing_matrixes = []
        # Sample the neurons, and build the freezing matrixes
        while len(neuron_list) > step_size:
            sampled_list = sample(neuron_list, int(freezable_neurons*ratio))
            freezing_matrix = self.get_empty_freezing_matrix()
            for elem in sampled_list:
                neuron_list.remove(elem)
                key,neuron = elem
                key_parts = key.split('-')
                FreezableModule._build_freezing_matrix(freezing_matrix, key_parts, neuron, order=order)
            freezing_matrixes.append((freezing_matrix,len(sampled_list)))
        # Build the last matrix, possibly smaller than the other ones
        sampled_list = neuron_list
        freezing_matrix = self.get_empty_freezing_matrix()
        for elem in sampled_list:
            key,neuron = elem
            key_parts = key.split('-')
            FreezableModule._build_freezing_matrix(freezing_matrix, key_parts, neuron, order=order)
        freezing_matrixes.append((freezing_matrix,len(sampled_list)))
        return freezing_matrixes

    def n_proportional_matrixes(self, ratio, order=True):
        matrix = []


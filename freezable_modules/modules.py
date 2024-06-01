import torch
import torch.nn as nn
import torch.nn.functional as F
import freezable_modules.bw_functions as bwf
from random import sample
from utils import normalize_output

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
        self.neq = False
        self.last_output = None
        self.last_cosim = None
        self.velocities = torch.ones(output_features)
        self.neq_momentum = 0
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
        if not self.neq:
            return bwf.FreezableConv2dFunction.apply(input, self.weight, self.bias, self.freeze, self.padding, self.stride, self.dilation, self.groups)
        else:
            y = bwf.FreezableConv2dFunction.apply(input, self.weight, self.bias, self.freeze, self.padding, self.stride, self.dilation, self.groups)
            yn = y.detach()
            yn = normalize_output(yn)
            if self.last_output is None:
                self.last_output = yn
                return y
            cosim = (yn*self.last_output).sum(dim=1) #compute the cosim over the flattened vector, and sum over the batch
            self.last_output = yn
            if self.last_cosim is None:
                self.last_cosim = cosim
                return y
            delta = cosim - self.last_cosim
            self.last_cosim = cosim
            self.velocities = delta - self.neq_momentum*self.velocities.to(delta.device)
            return y     

    def extra_repr(self):
        return f'input_features={self.input_features}, output_features={self.output_features}, bias={self.bias is not None}'

    def set_freeze(self, freeze:list):
        if freeze is None:
            self.freeze = []
            return
        self.freeze = freeze

    def get_default_freeze(self):
        return list(range(self.output_features))
    
    def get_proportinal_freeze(self, n):
        f = self.get_default_freeze()
        k, m = divmod(self.output_features, n)
        return (f[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    def get_total_parameters(self) -> int:
        s = self.weight.size()
        sb = 0 if self.bias is None else self.bias.size()[0]
        return s[0]*s[1]*s[2]*s[3] + sb
    
    def get_unfrozen_parameters(self) -> int:
        s = self.weight.size()
        sb = 0 if self.bias is None else self.bias.size()[0]
        return len(self.freeze)*s[1]*s[2]*s[3] + sb
    
    def get_param_per_neuron(self) -> int:
        s = self.weight.size()
        sb = 0 if self.bias is None else 1
        return s[1]*s[2]*s[3] + sb
    
    def update_neq_freeze(self, eps):
        self.freeze = []
        for i in range(self.output_features):
            if abs(self.velocities[i].item()) >= eps:
                self.freeze.append(i)
        return len(self.freeze)

class FreezableLinear(nn.Linear):
    def __init__(self, input_features, output_features, bias=True):
        super(FreezableLinear, self).__init__(input_features, output_features, bias)
        self.input_features = input_features
        self.output_features = output_features
        self.neq = False
        self.last_output = None
        self.last_cosim = None
        self.velocities = torch.ones(output_features)
        self.neq_momentum = 0
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
        if not self.neq:
            return bwf.FreezableLinearFunction.apply(input, self.weight, self.bias, self.freeze)
        else:
            y = bwf.FreezableLinearFunction.apply(input, self.weight, self.bias, self.freeze)
            yn = y.detach()
            yn = normalize_output(yn)
            if self.last_output is None:
                self.last_output = yn
                return y
            cosim = (yn*self.last_output).sum(dim=1) #compute the cosim
            self.last_output = yn
            if self.last_cosim is None:
                self.last_cosim = cosim
                return y
            delta = cosim - self.last_cosim
            self.last_cosim = cosim
            self.velocities = delta - self.neq_momentum*self.velocities.to(delta.device)
            return y
        
    def extra_repr(self):
        return f'input_features={self.input_features}, output_features={self.output_features}, bias={self.bias is not None}' 
    
    def set_freeze(self, freeze:list):
        if freeze is None:
            self.freeze = []
            return
        self.freeze = freeze
    
    def get_default_freeze(self):
        return list(range(self.output_features))

    def get_proportinal_freeze(self, n):
        f = self.get_default_freeze()
        k, m = divmod(self.output_features, n)
        return (f[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    def get_total_parameters(self) -> int:
        s = self.weight.size()
        sb = 0 if self.bias is None else self.bias.size()[0]
        return s[0]*s[1] + sb
    
    def get_unfrozen_parameters(self) -> int:
        s = self.weight.size()
        sb = 0 if self.bias is None else self.bias.size()[0]
        return len(self.freeze)*s[1] + sb
    
    def get_param_per_neuron(self) -> int:
        s = self.weight.size()
        sb = 0 if self.bias is None else 1
        return s[1] + sb
    
    def update_neq_freeze(self, eps):
        self.freeze = []
        for i in range(self.output_features):
            if abs(self.velocities[i].item()) >= eps:
                self.freeze.append(i)
        return len(self.freeze)


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
    
    def _neuron_list_with_velocities(self, extended_name=""):
        neuron_list = []
        velocities = torch.tensor([])
        for module in self.named_children():
            if isinstance(module[1],FreezableConv2d):
                for neuron in range(module[1].output_features):
                    neuron_list.append((extended_name+module[0],neuron,module[1].get_param_per_neuron()))
                velocities = torch.cat((velocities, module[1].velocities.cpu()))
            elif isinstance(module[1],FreezableLinear):
                for neuron in range(module[1].output_features):
                    neuron_list.append((extended_name+module[0],neuron,module[1].get_param_per_neuron()))
                velocities = torch.cat((velocities, module[1].velocities.cpu()))
            elif isinstance(module[1],FreezableModule):
                n_list, v_tensor = module[1]._neuron_list_with_velocities(extended_name+module[0]+"-")
                neuron_list.extend(n_list)
                velocities = torch.cat((velocities, v_tensor))
            elif isinstance(module[1],SequentialF):
                n_list, v_tensor = module[1]._neuron_list_with_velocities(extended_name+module[0]+"-")
                neuron_list.extend(n_list)
                velocities = torch.cat((velocities, v_tensor))
        return neuron_list, velocities
    
    def neq_mode(self, b=True, mu=None):
        if b:
            self.eval()
        for module in self.named_children():
            if isinstance(module[1], FreezableConv2d):
                module[1].neq = b
                if mu:
                    module[1].neq_momentum = mu
            elif isinstance(module[1], FreezableLinear):
                module[1].neq = b
                if mu:
                    module[1].neq_momentum = mu
            elif isinstance(module[1], FreezableModule):
                module[1].neq_mode(b,mu)
            elif isinstance(module[1], SequentialF):
                module[1].neq_mode(b,mu)
                
    def update_neq(self, eps):
        total = 0
        for module in self.named_children():
            if isinstance(module[1], FreezableConv2d):
                total += module[1].update_neq_freeze(eps)
            elif isinstance(module[1], FreezableLinear):
                total += module[1].update_neq_freeze(eps)
            elif isinstance(module[1], FreezableModule):
                total += module[1].update_neq(eps)
            elif isinstance(module[1], SequentialF):
                total += module[1].update_neq(eps)
        return total
        
    
    
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
        
    def _neuron_list_with_velocities(self, extended_name=""):
        neuron_list = []
        velocities = torch.tensor([])
        for module in self.named_children():
            if isinstance(module[1],FreezableConv2d):
                for neuron in range(module[1].output_features):
                    neuron_list.append((extended_name+module[0],neuron,module[1].get_param_per_neuron()))
                velocities = torch.cat((velocities, module[1].velocities.cpu()))
            elif isinstance(module[1],FreezableLinear):
                for neuron in range(module[1].output_features):
                    neuron_list.append((extended_name+module[0],neuron,module[1].get_param_per_neuron()))
                velocities = torch.cat((velocities, module[1].velocities.cpu()))
            elif isinstance(module[1],FreezableModule):
                n_list, v_tensor = module[1]._neuron_list_with_velocities(extended_name+module[0]+"-")
                neuron_list.extend(n_list)
                velocities = torch.cat((velocities, v_tensor))
            elif isinstance(module[1],SequentialF):
                n_list, v_tensor = module[1]._neuron_list_with_velocities(extended_name+module[0]+"-")
                neuron_list.extend(n_list)
                velocities = torch.cat((velocities, v_tensor))
        return neuron_list, velocities
    
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
    
    def n_random_freezing_matrices(self, ratio, order=True):
        """
        Returns a list of tuples containing a freezing matrix and the number of active neurons for this matrix.
        The matrices in the list cover all the neurons of the model.
            Args:
                ratio (float): the ratio of frozen neurons to be set (float between 0. and 1.)
        """
        # List all neurons
        neuron_list = self._neuron_list()
        # Get the number of neurons, and the size of a sample with respect to the ratio
        freezable_neurons = len(neuron_list)
        step_size = int(freezable_neurons*ratio)
        freezing_matrices = []
        # Sample the neurons, and build the freezing matrices
        while len(neuron_list) > step_size:
            sampled_list = sample(neuron_list, int(freezable_neurons*ratio))
            freezing_matrix = self.get_empty_freezing_matrix()
            for elem in sampled_list:
                neuron_list.remove(elem)
                key,neuron = elem
                key_parts = key.split('-')
                FreezableModule._build_freezing_matrix(freezing_matrix, key_parts, neuron, order=order)
            freezing_matrices.append((freezing_matrix,len(sampled_list)))
        # Build the last matrix, possibly smaller than the other ones
        sampled_list = neuron_list
        freezing_matrix = self.get_empty_freezing_matrix()
        for elem in sampled_list:
            key,neuron = elem
            key_parts = key.split('-')
            FreezableModule._build_freezing_matrix(freezing_matrix, key_parts, neuron, order=order)
        freezing_matrices.append((freezing_matrix,len(sampled_list)))
        return freezing_matrices

    def n_proportional_matrices(self, ratio):
        n = int(round(1/ratio))
        print("building " + str(n) + " proportional matrices")
        freezing_matrices = [[self.get_empty_freezing_matrix(),0] for _ in range(n)]
        base_matrix = self.get_default_freezing_matrix()
        
        def split(a, n):
            k, m = divmod(len(a), n)
            return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]
        
        def change_field(d1:dict, d2:dict, i):
            for k in d1:
                if isinstance(d1[k], dict):
                    change_field(d1[k],  d2[k], i)
                else:
                    l = split(d2[k], n)[i]
                    d1[k] = l
                    freezing_matrices[i][1] += len(l)
                    
        for i in range(n):
            change_field(freezing_matrices[i][0], base_matrix, i)
        
        return freezing_matrices
    
    def neq_mode(self, b=True, mu=None):
        if b:
            self.eval()
        for module in self.named_children():
            if isinstance(module[1], FreezableConv2d):
                module[1].neq = b
                if mu:
                    module[1].neq_momentum = mu
            elif isinstance(module[1], FreezableLinear):
                module[1].neq = b
                if mu:
                    module[1].neq_momentum = mu
            elif isinstance(module[1], FreezableModule):
                module[1].neq_mode(b,mu)
            elif isinstance(module[1], SequentialF):
                module[1].neq_mode(b,mu)
        
    def update_neq(self, eps):
        total = 0
        for module in self.named_children():
            if isinstance(module[1], FreezableConv2d):
                total += module[1].update_neq_freeze(eps)
            elif isinstance(module[1], FreezableLinear):
                total += module[1].update_neq_freeze(eps)
            elif isinstance(module[1], FreezableModule):
                total += module[1].update_neq(eps)
            elif isinstance(module[1], SequentialF):
                total += module[1].update_neq(eps)
        return total
    
    def velocity_freezing_matrix(self, ratio, order=True):
        """
        Returns a freezing matrix using velocity as a metric to sort neurons. After beeing sorted, the neurons are added to the matrix
        as long as their parameters fit in the budget.
            Args:
                ratio (float): the ratio of frozen neurons to be set (float between 0. and 1.)
        """
        neuron_list, velocities = self._neuron_list_with_velocities()
        n_unfrozen = 0
        max_param = int(self.get_total_parameters()*ratio)
        vel,indices = torch.sort(torch.abs(velocities), descending=True)
        sorted_neurons = []
        for i in range(len(neuron_list)):
            print(vel[i].item())
            if abs(vel[i].item()) > 1:
                raise Exception
            idx = indices[i].item()
            sorted_neurons.append(neuron_list[idx])
        freezing_matrix = self.get_empty_freezing_matrix()
        n_params = 0
        for key,neuron,params in sorted_neurons:
            if n_params+params > max_param:
                continue
            n_params += params
            n_unfrozen += 1
            key_parts = key.split('-')
            FreezableModule._build_freezing_matrix(freezing_matrix, key_parts, neuron, order=order)            
        return freezing_matrix, n_unfrozen


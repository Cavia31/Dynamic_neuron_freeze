import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_input,conv2d_weight

class FreezableConv2dFunction(torch.autograd.Function):
    '''An implementation of a 2d convolution function that supports passing 
    a subset of the weights to the gradient computation. To compute the
    backward step, the functions provided in torch.nn.grad are used.'''
    @staticmethod
    def forward(ctx, input, weight, bias, freeze:list, padding, stride, dilation, groups) -> torch.Any:
        ctx.freeze = freeze
        ctx.padding = padding
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.save_for_backward(input, weight, bias)
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx: torch.Any, grad_output: torch.Any) -> torch.Any:
        input, weight, bias = ctx.saved_tensors
        grad_input = None
        grad_weight = torch.zeros(weight.shape, device=weight.device)
        if bias is not None:
            grad_bias = torch.zeros(bias.shape, device=bias.device)
        else:
            grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = conv2d_input(input.shape, weight, grad_output, ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        if ctx.needs_input_grad[1]:
            if len(ctx.freeze) > 0:
                if ctx.groups != 1: #TODO Depthwise convolution with out_channels = input_channels and other sizes of groups will use false sparse backward
                    temp_grad = conv2d_weight(input, weight.shape, grad_output, ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
                    grad_weight[ctx.freeze] = temp_grad[ctx.freeze]
                    #ctx.groups = len(ctx.freeze)
                    # inputs can be separated because this is depthwise convolution (all inputs separated)
                    #grad_weight[ctx.freeze] = conv2d_weight(input[:,ctx.freeze], weight[ctx.freeze].shape, grad_output[:,ctx.freeze], ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
                else:
                    grad_weight[ctx.freeze] = conv2d_weight(input, weight[ctx.freeze].shape, grad_output[:,ctx.freeze], ctx.stride, ctx.padding, ctx.dilation, ctx.groups) 
        if bias is not None and ctx.needs_input_grad[2]:
            if len(ctx.freeze) > 0:
                grad_bias[ctx.freeze] = grad_output[:,ctx.freeze].sum((0,2,3)).squeeze(0)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None
    
class FreezableLinearFunction(torch.autograd.Function):
    '''An implementation of a linear function that supports passing a 
    subset of the weights to the gradient computation. To implement 
    the freeze a dictionary, freeze, is passed as a variable to the forward. 
    This is instantiated inside the module and since it is a dictionary, 
    changes to the value associated with a key are accessible to the 
    backward allowing to determine the number of neurons at equilibrium 
    after the forward.'''
    @staticmethod
    def forward(ctx, input, weight, bias, freeze:list):
        ctx.freeze = freeze
        ctx.save_for_backward(input, weight, bias)
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = None
        grad_weight = torch.zeros(weight.shape, device=weight.device)
        if bias is not None:
            grad_bias = torch.zeros(bias.shape, device=bias.device)
        else:
            grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight)
        if ctx.needs_input_grad[1]:
            if len(ctx.freeze) > 0:
                grad_weight[ctx.freeze] = torch.matmul(grad_output[:,ctx.freeze].t(), input)
        if bias is not None and ctx.needs_input_grad[2]:
            if len(ctx.freeze) > 0:
                grad_bias[ctx.freeze] = grad_output[:,ctx.freeze].sum(0)
        return grad_input, grad_weight, grad_bias, None, 
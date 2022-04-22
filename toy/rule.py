from os import stat
import torch
import torch.nn as nn

from graph import Graph
from device import device

class Clipping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        output = input_.clamp(min=0, max=1)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input

class MaxCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        return torch.max(input_, dim=1)[0]
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        shape = input.shape
        temp = torch.ones(shape)
        return temp.T.mul(grad_output).T

class MaxWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        return torch.max(input_, dim=2)[0]
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        shape = input.shape
        temp = torch.ones(shape)
        return temp.T.mul(grad_output.T).T

class Step(nn.Module):
    def __init__(self, n_predicate: int, randinit=False):
        super(Step, self).__init__()
        if (randinit):
            data = torch.randn(n_predicate)
        else:
            data = torch.zeros(n_predicate)
        self.R = nn.Parameter(data=data, requires_grad=True)
    
    def _regularized_R(self, power=1.0):
        return torch.pow(torch.softmax(self.R, dim=0), power)

    def forward(self, status: torch.Tensor, graph: Graph, power=1.0):
        R = self._regularized_R(power)
        status = graph.A.matmul(status).T.matmul(R).T
        # status = torch.max(graph.A.matmul(status).T.mul(R), dim=1)[0]
        # status = torch.max(MaxWithGrad.apply(graph.A.mul(status)).T.mul(R), dim=1)[0]
        return status

class Rule(nn.Module):
    def __init__(self, n_predicate: int, n_step: int, randinit=False):
        super(Rule, self).__init__()
        self.R = nn.ModuleList()
        for i in range(n_step):
            self.R.append(Step(n_predicate, randinit))
        self.n_predicate = n_predicate
        self.n_step = n_step

    def _regularized_R(self, power=1.0):
        ret = torch.zeros(self.n_predicate).to(device)
        for step in self.R:
            ret = ret + step._regularized_R(power) / self.n_step
        return ret

    def forward(self, graph: Graph, power=1.0):
        status = torch.zeros((graph.n_vertex, graph.n_vertex)).to(device)
        for i in range(graph.n_vertex):
            status[i][i] = 1.0

        for step in range(self.n_step):
            status = Clipping.apply(self.R[step](status, graph, power))
            # status = torch.clamp(self.R[step](status, graph, power), min=0.0, max=1.0)
            # status = graph.A.matmul(status).T.matmul(R[step])

        return status

    
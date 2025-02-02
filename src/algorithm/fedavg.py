import torch

from .basealgorithm import BaseOptimizer



class FedavgOptimizer(BaseOptimizer):
    def __init__(self, params, **kwargs):
        # self.lr = kwargs.get('lr')
        # self.momentum = kwargs.get('momentum', 0.)
        # defaults = dict(lr=self.lr, momentum=self.momentum)
        # BaseOptimizer.__init__(self); torch.optim.Optimizer.__init__(self, params=params, defaults=defaults)
        self.params = params

    def zero_grad(self, set_to_none=False):
        for name, param in self.params.items():
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    if param.grad.grad_fn is not None:
                        param.grad.detach_()
                    else:
                        param.grad.requires_grad_(False)
                    param.grad.zero_()
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for name, param in self.params.items():
            if param.grad is None:
                continue
            delta = param.grad.data
            param.data.sub_(delta)

        return self.params

    def accumulate(self, mixing_coefficient, local_layers_iterator, check_if=lambda name: 'num_batches_tracked' in name):
        # for group in self.param_groups:
        for server_param, (name, local_signals) in zip(self.params.values(), local_layers_iterator):
            if check_if(name) or name not in mixing_coefficient:
                # server_param.data.zero_()
                # server_param.data.grad = torch.zeros_like(server_param)
                # local_delta = torch.zeros_like(server_param)
                continue
            if mixing_coefficient[name] == 0 or local_signals is None:
                local_delta = torch.zeros_like(server_param)
            else:
                local_delta = (server_param - local_signals).mul(mixing_coefficient[name]).data.type(server_param.dtype)
            if server_param.grad is None: # NOTE: grad buffer is used to accumulate local updates!
                server_param.grad = local_delta
            else:
                server_param.grad.data.add_(local_delta)

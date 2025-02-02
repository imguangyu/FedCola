import torch

from .fedavg import FedavgOptimizer



class VectorOptimizer(FedavgOptimizer):
    def __init__(self, params, **kwargs):
        super(VectorOptimizer, self).__init__(params=params, **kwargs)

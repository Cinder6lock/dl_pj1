from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model) -> None:
        super().__init__(init_lr, model)
    
    def step(self) -> None:
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu) -> None:
        super().__init__(init_lr, model)
        self.mu = mu
        self.momentums = {}
        for layer in self.model.layers:
            if layer.optimizable == True:
                self.momentums[layer] = {}
                for key in layer.params.keys():
                    self.momentums[layer][key] = np.zeros_like(layer.params[key])
    
    def step(self) -> None:
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    self.momentums[layer][key] = self.mu * self.momentums[layer][key] - self.init_lr * layer.grads[key]
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] += self.momentums[layer][key]